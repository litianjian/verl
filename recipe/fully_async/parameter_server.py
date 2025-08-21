
import ray
import logging

from typing import Any, Optional

logger = logging.getLogger(__name__)

@ray.remote
class ParamterServer:
    def __init__(
        self,
        config: Any,
        trainer: Any,
        rollouter: Any,
        mq_client: Any
    ):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter

        self.mq_client = mq_client

        self.weight_infos = None
        self._is_initialized = False
        self.sync_group_name = "actor_rollout"
        self.backend = "nccl"
        self.current_version = 0

        self._initialize()

    def _initialize(self):
        """
        Initialize the parameter server.
        """
        try:
            self._init_worker_groups()
            self._init_weights_info()
            self._init_sync_group()
            self._is_initialized = True
        except Exception as e:
            logging.error(f"Failed to initialize the parameter server: {e}")
            raise

    def _init_worker_groups(self):
        """
        Initialize the worker groups.
        """
        self.actor_wg = ray.get(self.trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(self.rollouter.get_rollout_wg.remote())

    def _init_weights_info(self):
        """
        Initialize the weights info.
        """
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    
    def _init_sync_group(self):
        """
        Initialize the sync group.
        """
        print("init sync group")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        from ray.util.collective import collective
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend=self.backend,
            group_name=self.sync_group_name,
        )

    def _pause_rollout(self):
        """
        Pause the rollout.
        """
        ray.get(self.rollouter.pause.remote())

    def _resume_rollout(self):
        """
        Resume the rollout.
        """
        ray.get(self.rollouter.resume.remote())

    def _update_mq_version(self, version):
        """
        Update the mq version.
        """
        
        self.mq_client.update_param_version_sync(version)
        self.current_version = version

    def _update_actor_weights(self):
        """
        Update the actor weights.
        """
        self.actor_wg.sync_rollout_weights()
        ray.get(self.rollout_wg.sync_rollout_weights())

    def _update_rollout_weigts_version(self, version):
        """
        Update the rollout weights.
        """
        ray.get(self.rollouter.update_param_version.remote(version))

    def sync_weights(self, version):
        """
        Sync the rollout weights.
        操作顺序：
        1. 暂停Rollout Worker
        2. Actor更新权值
        3. Rollouter更新权值
        4. 重启Rollout Worker

        """
        logger.info(f"sync rollout weights for versions: {version}")

        try:
            self._pause_rollout()
            print("pause rollout")
            self._update_mq_version(version)
            print("update mq version")
            self._update_actor_weights()
            print("update actor weights")
            self._update_rollout_weigts_version(version)
            print("update rollout weights")
            self._resume_rollout()
            print("resume rollout")
        except Exception as e:
            logger.error(f"Failed to sync rollout weights: {e}")
            self._resume_rollout()
            raise