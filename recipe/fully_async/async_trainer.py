
import logging
import time
from typing import Any, Optional
from dataclasses import dataclass

import ray
import torch

from omegaconf import OmegaConf

import numpy as np
from .detach_utils import (
    assemble_batch_from_rollout_samples,
    calculate_one_step_size,
)
from .message_queue import MessageQueueClient
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import Tracking
from verl.utils.config import omega_conf_to_dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Simplified configuration wrapper"""
    trigger_parameter_sync_step: int
    balance_batch: bool
    project_name: str
    experiment_name: str
    logger_backend: str


@dataclass 
class TrainerStats:
    """Training statistics tracker"""
    processed_samples: int = 0
    stale_samples_processed: int = 0
    current_param_version: int = 0
    local_trigger_step: int = 1


class SampleCollector:
    """Handles sample collection from message queue"""
    
    def __init__(self, message_queue_client: MessageQueueClient, required_samples: int):
        self.message_queue_client = message_queue_client
        self.required_samples = required_samples
    
    def collect_samples(self) -> Optional[list]:
        """Collect required number of samples from queue"""
        print(f"[SampleCollector] Requesting {self.required_samples} samples")
        
        samples = []
        start_time = time.time()
        
        while len(samples) < self.required_samples:
            sample = self.message_queue_client.get_sample_sync()
            
            if sample is None:
                print(f"[SampleCollector] Termination signal received. Collected {len(samples)}/{self.required_samples}")
                break
                
            samples.append(sample)

        if not samples or len(samples) < self.required_samples:
            logger.warning("Insufficient samples collected")
            return None
            
        elapsed = time.time() - start_time
        print(f"[SampleCollector] Completed: {len(samples)} samples in {elapsed:.2f}s")
        
        return [ray.cloudpickle.loads(sample) for sample in samples]


class BatchProcessor:
    """Handles batch processing and metrics calculation"""
    
    def __init__(self, tokenizer, config, balance_batch_fn=None):
        self.tokenizer = tokenizer
        self.config = config
        self.balance_batch_fn = balance_batch_fn
    
    def process_samples(self, samples: list, stats: TrainerStats) -> tuple[Any, dict]:
        """Process samples into training batch and calculate metrics"""
        # Assemble batch
        batch = assemble_batch_from_rollout_samples(
            samples, 
            self.tokenizer, 
            self.config, 
            self.balance_batch_fn
        )
        
        # Calculate metrics
        metrics = {}
        if hasattr(batch, "meta_info") and batch.meta_info:
            metrics.update(self._calculate_staleness_metrics(batch, stats))
            metrics.update(self._extract_processing_metrics(batch))
        
        return batch, metrics
    
    def _calculate_staleness_metrics(self, batch: Any, stats: TrainerStats) -> dict:
        """Calculate staleness-related metrics"""
        rollout_versions = batch.meta_info["rollout_param_versions"]
        stale_count = sum(1 for v in rollout_versions if stats.current_param_version - v > 1)
        stats.stale_samples_processed += stale_count
        
        return {
            "fully_async/stale_samples_ratio": stale_count / len(rollout_versions),
            "fully_async/stale_samples_processed": stats.stale_samples_processed,
            "fully_async/current_param_version": stats.current_param_version,
        }
    
    def _extract_processing_metrics(self, batch: Any) -> dict:
        """Extract processing time metrics"""
        metrics = {}
        for metric_name in [
            "avg_processing_time", "max_processing_time", "min_processing_time",
            "tp50_processing_time", "tp99_processing_time", "tp95_processing_time",
            "param_version_diversity"
        ]:
            metrics[f"fully_async/{metric_name}"] = batch.meta_info.get(metric_name, 0)
        return metrics


class ParameterSynchronizer:
    """Handles parameter synchronization logic"""
    
    def __init__(self, param_synchronizer, trigger_step: int):
        self.param_synchronizer = param_synchronizer
        self.trigger_step = trigger_step
    
    def maybe_sync(self, stats: TrainerStats) -> None:
        """Trigger parameter sync if needed"""
        if stats.local_trigger_step >= self.trigger_step:
            stats.local_trigger_step = 1
            stats.current_param_version += 1
            
            print(f"[ParameterSynchronizer] Syncing version {stats.current_param_version}")
            ray.get(self.param_synchronizer.sync_weights.remote(stats.current_param_version))
        else:
            stats.local_trigger_step += 1


@ray.remote(num_cpus=2)
class FullyAsyncTrainer(RayPPOTrainer):
    """
    Simplified fully asynchronous PPO trainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Basic setup
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        
        # Validate config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine, "Hybrid engine not supported"
        
        # Initialize parent
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name or config.trainer.device
        
        # Model configuration
        self._setup_model_config()
        self._validate_config()
        
        # Core components
        self.message_queue_client = None
        self.param_server = None
        
        # Calculate required samples
        self.required_samples = calculate_one_step_size(
            self.minimal_bsz, 
            config.actor_rollout_ref.actor.ppo_mini_batch_size
        )
        
        # Training state
        self.stats = TrainerStats()
        
        # Configuration
        self.trainer_config = TrainerConfig(
            trigger_parameter_sync_step=config.async_training.trigger_parameter_sync_step,
            balance_batch=config.trainer.balance_batch,
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            logger_backend=config.trainer.logger
        )
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)


        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        # create actor and rollout
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        self.resource_pool_to_cls[resource_pool]["actor"] = actor_cls
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        
        self.actor_rollout_wg = all_wg["actor"]
        self.actor_rollout_wg.init_model()
        self.actor_wg = self.actor_rollout_wg

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )
        self.minimal_bsz = minimal_bsz

        super()._validate_config()

    def _setup_model_config(self):
        """Setup model-related configuration"""
        self.use_reference_policy = Role.RefPolicy in self.role_worker_mapping
        self.use_rm = Role.RewardModel in self.role_worker_mapping
        self.ref_in_actor = self.config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        
        # Critic configuration
        if self.config.critic.enable is not None:
            self.use_critic = bool(self.config.critic.enable)
        else:
            from verl.trainer.ppo.core_algos import AdvantageEstimator
            self.use_critic = self.config.algorithm.adv_estimator == AdvantageEstimator.GAE
    
    def set_message_queue_client(self, client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = client
    
    def set_parameter_server(self, param_server):
        """Set parameter synchronizer"""
        self.param_server = param_server
    
    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg
    
    def _create_actor_rollout_classes(self):
        """Create only actor classes for training"""
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        role_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        self.resource_pool_to_cls[resource_pool]["actor"] = role_cls
    
    def _init_models(self):
        """Initialize required model worker groups"""
        model_roles = []
        
        if self.use_critic:
            model_roles.append(Role.Critic)
        if self.use_reference_policy and not self.ref_in_actor:
            model_roles.append(Role.RefPolicy)
        if self.use_rm:
            model_roles.append(Role.RewardModel)
        
        # Initialize model worker groups
        for role in model_roles:
            wg = self.all_wg[str(role)]
            wg.init_model()
            setattr(self, f"{role.value.lower()}_wg", wg)
        
        # Initialize actor (required)
        self.actor_wg = self.all_wg["actor"]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg
    
    def _init_async_rollout_manager(self):
        """No async rollout manager needed for trainer"""
        pass
    
    def fit(self):
        """Main training loop"""
        self._validate_setup()
        
        # Initialize components
        logger = self._setup_logger()
        sample_collector = SampleCollector(self.message_queue_client, self.required_samples)
        batch_processor = BatchProcessor(
            self.tokenizer, 
            self.config, 
            self._balance_batch if self.trainer_config.balance_batch else None
        )
        param_sync = ParameterSynchronizer(
            self.param_server, 
            self.trainer_config.trigger_parameter_sync_step
        )
        
        # Load checkpoint and setup
        self._load_checkpoint()
        self.global_steps = 1
        
        print("[FullyAsyncTrainer] Starting training loop...")
        
        # Main training loop
        while True:
            timing_raw = {}
            
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # Collect samples
                with marked_timer("gen", timing_raw, color="red"):
                    samples = sample_collector.collect_samples()
                    if samples is None:
                        break
                
                # Process batch
                batch, metrics = batch_processor.process_samples(samples, self.stats)
                
                # Train
                batch, reward_extra_infos = self._process_batch_common(batch, metrics, timing_raw)
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()            
            # Log metrics
            print(f"[FullyAsyncTrainer] Step {self.global_steps}: {metrics}")
            
            # Sync parameters
            param_sync.maybe_sync(self.stats)
            self.global_steps += 1

    def _process_batch_common(self, batch, metrics, timing_raw):
        with marked_timer("reward", timing_raw, color="yellow"):
            # compute reward model score
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                # TODO: we may want to add diff of probs too.
                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                actor_old_log_probs = batch.batch["old_log_probs"]
                attention_mask = batch.batch["attention_mask"]
                responses = batch.batch["responses"]
                response_length = responses.size(1)
                response_mask = attention_mask[:, -response_length:]

                rollout_probs = torch.exp(rollout_old_log_probs)
                actor_probs = torch.exp(actor_old_log_probs)
                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                metrics.update(
                    {
                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    }
                )
        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        # compute values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        with marked_timer("adv", timing_raw, color="brown"):
            # we combine with rule-based rm
            reward_extra_infos_dict: dict[str, list]
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # compute advantages, executed on the driver process

            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)
        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return batch, reward_extra_infos_dict
    
    def _validate_setup(self):
        """Validate trainer setup"""
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set")
        if self.param_server is None:
            raise ValueError("Parameter server not set")
    
    def _setup_logger(self) -> Tracking:
        """Setup training logger"""
        return Tracking(
            project_name=self.trainer_config.project_name,
            experiment_name=self.trainer_config.experiment_name,
            default_backend=self.trainer_config.logger_backend,
            config=OmegaConf.to_container(self.config, resolve=True),
        )