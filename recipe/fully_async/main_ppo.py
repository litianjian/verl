import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type

from .message_queue import MessageQueue, MessageQueueClient
from .async_trainer import FullyAsyncTrainer
from .async_rollouter import FullyAsyncRollouter
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

def create_role_worker_mappings(config):
    """Create role worker mappings based on the actor strategy."""
    from verl.single_controller.ray import RayWorkerGroup
    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        from .fsdp_workers import (
            CriticWorker,
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from .megatron_workers import (
            CriticWorker,
            DetachActorWorker,
            DetachAsyncRolloutWorker,
        )
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")


    role_worker_mapping = {
        Role.Actor: ray.remote(DetachActorWorker),
        Role.Rollout: ray.remote(DetachAsyncRolloutWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp2":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"Unsupported reward model strategy: {config.reward_model.strategy}")

        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    # 添加reference policy（如果需要KL loss或reward）
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls
    

def create_resource_pool_mgr(config, roles: list) -> ResourcePoolManager:
    """Create resource pool manager based on the roles and configuration."""
    # 初始化资源池管理器
    resource_pool_spec = {}
    mapping = {}

    # Actor/Critic resource pool
    if any(role in roles for role in [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]):
        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        # Map training-related roles to the same resource pool
        for role in [Role.Actor, Role.Critic, Role.RefPolicy, Role.RewardModel]:
            if role in roles:
                mapping[role] = "trainer_pool"

    if Role.Rollout in roles:
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
        resource_pool_spec["rollout_pool"] = rollout_pool
        mapping[Role.Rollout] = "rollout_pool"
    # 初始化资源池管理器
    resource_pool_mgr = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    return resource_pool_mgr
    
@hydra.main(config_path="config", config_name="fully_async", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)



@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)


    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"
    
    def init_envs(self, config):

        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        self.config = config
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.tokenizer = tokenizer
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.processor = processor

        self.role_worker_mapping, self.ray_worker_group_cls = create_role_worker_mappings(config)

        # Load the reward manager for training and validation.
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        print("Creating FullyAsyncRollouter...")

        self._create_rollout_worker(config)

        print("Creating Trainer...")
        self._create_actor_worker(config)

        max_queue_size = 1000
        message_queue = MessageQueue.remote(config, queue_size=max_queue_size)
        self.message_queue_client = MessageQueueClient(message_queue)

        ray.get(self.rollouter.set_message_queue_client.remote(self.message_queue_client))
        ray.get(self.trainer.set_message_queue_client.remote(self.message_queue_client))

        from .parameter_server import ParamterServer

        param_server = ParamterServer.remote(
            config=self.config,
            trainer=self.trainer,
            rollouter=self.rollouter,
            mq_client=self.message_queue_client,
        )
        self.param_server = param_server

        ray.get(self.rollouter.set_parameter_server.remote(param_server))
        ray.get(self.trainer.set_parameter_server.remote(param_server))

        ray.get(param_server.sync_weights.remote(0))

    def _create_rollout_worker(self,config):
        rollout_role_mapping = {
            Role.Rollout: self.role_worker_mapping[Role.Rollout],
        }
        resource_mgr = create_resource_pool_mgr(config, roles=list(rollout_role_mapping.keys()))

        rollouter = FullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.tokenizer,
            role_worker_mapping=rollout_role_mapping,
            resource_pool_manager=resource_mgr,
            ray_worker_group_cls=self.ray_worker_group_cls,
            processor=self.processor,
            device_name=config.trainer.device,
        )
        ray.get(rollouter.init_workers.remote())
        self.rollouter = rollouter

    def _create_actor_worker(self, config):

        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.role_worker_mapping.items()
            if role != Role.Rollout
        }

        resource_mgr = create_resource_pool_mgr(config, roles=list(trainer_role_mapping.keys()))
        trainer = FullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.tokenizer,
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=resource_mgr,
            ray_worker_group_cls=self.ray_worker_group_cls,
            processor=self.processor,
            reward_fn=self.reward_fn,
            val_reward_fn=self.val_reward_fn,
            device_name=config.trainer.device,
        )
        ray.get(trainer.init_workers.remote())
        self.trainer = trainer
        print("Creating FullyAsyncRollouter...")

    def _running_loop(self):
        self.running = True

        print("Starting _running_loop")

        feature_rollout = self.rollouter.fit.remote()
        feature_trainer = self.trainer.fit.remote()

        ray.get(feature_rollout)
        ray.get(feature_trainer)

        self.message_queue_client.clear_queue()

        print("Finished _running_loop")
    
    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        
        self.init_envs(config)
        self._running_loop()


if __name__ == "__main__":
    main()