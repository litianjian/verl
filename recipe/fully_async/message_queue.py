
import ray
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
from enum import Enum
from collections import deque
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


@dataclass
class QueueStats:
    """
    统计消息队列信息
    """
    total_produced: int = 0
    total_consumed: int = 0
    current_size: int = 0
    param_version: int = 0
    dropped_sample: int = 0
    dropped_full: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            'total_produced': self.total_produced,
            'total_consumed': self.total_consumed,
            'current_size': self.current_size,
            'param_version': self.param_version,
            'dropped_sample': self.dropped_sample,
            'dropped_full': self.dropped_full,
            }


class QueueState(Enum):
    RUNNING="running"
    PAUSED="paused"
    STOPPED="stopped"


@ray.remote(num_cpus=2, max_concurrency=50)
class MessageQueue:
    """
    用于连接TrainerWorker和RolloutWorker的消息队列,实现异步训练

    特性：
    1. 完全异步操作, 支持高并发
    2. 支持StepOff threshold,支持动态调整

    """

    def __init__(
        self,
        config: Optional[DictConfig] = None,
        queue_size: int = 1000,
        step_off_threshold: int = 3,
        log_interval: int = 100,
    ):
        self.config = config
        self.max_queue_size = queue_size
        self.queue = deque(maxlen=self.max_queue_size)
        self._sample_id_counter = 0
        self.log_interval = log_interval

        try:
            if hasattr(config, "async_training") and config.async_training is not None:
                self.step_off_threshold = getattr(config.async_training, "step_off_threshold", 3)
            else:
                self.step_off_threshold = 3
        except (AttributeError, RecursionError):
            self.step_off_threshold = 3
    
        self.current_param_version = 0
        
        self.state = QueueState.RUNNING
        self.stats = QueueStats()

        self._lock = asyncio.Lock()
        self._producer_condition = asyncio.Condition(self._lock)
        self._consumer_condition = asyncio.Condition(self._lock)

        self._last_log_time = time.time()
        logger.info(f"MessageQueue initialized with max_queue_size={self.max_queue_size}")
        logger.info(f"MessageQueue initialized with step_off_threshold={self.step_off_threshold}")


    async def put_sample(
        self,
        sample_data: Any,
        param_version: int,
        sample_id: str = "",
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        异步将样本放入队列中
        """
        async with self._lock:
            if self.state != QueueState.RUNNING:
                logger.warning(f"MessageQueue is {self.state.value}, cannot put_sample")
                return False
            step_off = self.current_param_version - param_version
            if step_off > self.step_off_threshold:
                self.satts.dropped_sample += 1
                logger.warning(f"MessageQueue step_off={step_off}, cannot put_sample",
                               f"current_param_version={self.current_param_version}, param_version={param_version}")
                return False

            if sample_id is None:
                self._sample_id_counter += 1
                sample_id = f"sample_{self._sample_id_counter}"

            #处理队列满的情况
            if len(self.queue) >= self.max_queue_size:
                #队列会去除掉最老的数据，但是我们可以添加统计信息
                self.stats.dropped_full += 1

            self.queue.append(sample_data)
            self.stats.total_produced += 1
            self.stats.current_size = len(self.queue)

            #通知等待的消费者
            self._consumer_condition.notify_all()
            #打印统计信息
            await self._maybe_log_stats()

            return True

    async def _maybe_log_stats(self):
        if self.stats.total_produced % 100 == 0:
            logger.info(f"MessageQueue stats: {self.stats.to_dict()}")

    async def get_batch(
        self,
        batch_size: int,
        timeout: Optional[float] = None,
        min_batch_size: int = 1
    ):
        """
        异步从队列中获取样本
        """
        async with self._lock:
            batch = []
            start_time = time.time()
            while len(self.queue) < min_batch_size and self.state == QueueState.RUNNING:
                # 检查超时
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = timeout - elapsed
                    if remaining_timeout <= 0:
                        logger.debug(f"get_batch timeout, returning {len(batch)} samples")
                        break
                    try:
                        await asyncio.wait_for(
                            self._consumer_condition.wait(),
                            timeout=remaining_timeout
                            )
                    except asyncio.TimeoutError:
                        break
                else:
                    await self._consumer_condition.wait()
            # 获取样本
            actual_batch_size = min(batch_size, len(self.queue))
            for _ in range(actual_batch_size):
                if self.queue:
                    sample = self.queue.popleft()
                    batch.append(sample)
                    self.stats.total_consumed += 1
            self.stats.current_size = len(self.queue)
            # 通知等待的生产者（如果有背压控制）
            self._producer_condition.notify_all()
            if batch:
                logger.debug(f"Retrieved batch of {len(batch)} samples")
            return batch

    async def get_sample(
        self,
        timeout: Optional[float] = None,
    ):
        """
        异步从队列中获取单个样本
        """
        batch = await self.get_batch(batch_size=1, timeout=timeout, min_batch_size=1)
        if batch:
            return batch[0]
        else:
            return None

    async def update_param_version(self, param_version: int) -> None:
        """
        异步更新参数版本
        """
        async with self._lock:
            old_param_version = self.current_param_version
            self.current_param_version = param_version
            self.stats.param_version = param_version

            #清理过期样本（是否需要在这里清理）
            cleaned_count = self._clean_up_stale_samples()

    async def _clean_up_stale_samples(self) -> int:
        """
        异步清理过期样本
        """
        if not self.queue:
            return 0
        cleaned_queue = deque(maxlen=self.max_queue_size)
        cleaned_count = 0

        for sample in self.queue:
            step_off = self.current_param_version - sample.param_version
            if step_off <= self.step_off_threshold:
                cleaned_queue.append(sample)
            else:
                cleaned_count += 1
        #TODO: 是否需要清空原来的Queue
        self.queue = cleaned_queue
        self.stats.dropped_sample += cleaned_count
        self.stats.current_size = len(self.queue)

        return cleaned_count
        
    async def pause(self) -> None:
        """
        异步暂停队列
        """
        async with self._lock:
            self.state = QueueState.PAUSED
            logger.info(f"MessageQueue paused")

    async def resume(self) -> None:
        """
        异步恢复队列
        """
        async with self._lock:
            self.state = QueueState.RUNNING
            logger.info(f"MessageQueue resumed")

    async def stop(self) -> None:
        """
        异步停止队列
        """
        async with self._lock:
            self.state = QueueState.STOPPED
            self._consumer_condition.notify_all()
            self._producer_condition.notify_all()
            logger.info(f"MessageQueue stopped")

    async def clear_queue(self) -> None:
        """
        异步清空队列
        """
        async with self._lock:
            self.queue.clear()
            self.stats.current_size = len(self.queue)
            logger.info(f"MessageQueue cleared")

    async def get_stats(self) -> dict[str, Any]:
        """
        异步获取队列统计信息
        """
        async with self._lock:
            return self.stats.to_dict()

    
    async def get_queue_size(self) -> int:
        """
        异步获取队列大小
        """
        async with self._lock:
            return len(self.queue)

    async def is_empty(self) -> bool:
        """
        异步判断队列是否为空
        """
        async with self._lock:
            return len(self.queue) == 0
    
    async def is_full(self) -> bool:
        """
        异步判断队列是否满
        """
        async with self._lock:
            return len(self.queue) == self.max_queue_size

    async def _maybe_log_stats(self) -> None:
        """定期打印统计信息（必须在锁内调用）"""
        current_time = time.time()
        if (self.stats.total_produced % self.log_interval == 0 or current_time - self._last_log_time > 30):
            logger.info(f"MessageQueue stats: "
                        f"produced={self.stats.total_produced}, "
                        f"consumed={self.stats.total_consumed}, "
                        f"current_size={self.stats.current_size}, "
                        f"dropped_sample={self.stats.dropped_sample}, "
                        f"dropped_full={self.stats.dropped_full}, "
                        f"param_version={self.stats.param_version}")
            self._last_log_time = current_time



class MessageQueueClient:
    """
    MessageQueue的客户端
    """
    def __init__(self, message_queue_ref) -> None:
        self.mq_ref = message_queue_ref
    
    async def put_sample(
        self,
        sample_data: Any,
        param_version: int,
        **kwargs
    ) -> None:
        """
        异步将样本放入队列中
        """
        await self.mq_ref.put_sample.remote(sample_data, param_version, **kwargs)

    async def get_batch(
        self,
        batch_size: int,
        **kwargs
    ):
        """
        异步从队列中获取样本
        """
        return await self.mq_ref.get_batch.remote(batch_size, **kwargs)
    
    async def get_sample(
        self,
        **kwargs
    ):
        """
        异步从队列中获取单个样本
        """
        return await self.mq_ref.get_sample.remote(**kwargs)

    async def update_param_version(
        self,
        param_version: int,
    ) -> None:
        """
        异步更新参数版本
        """
        await self.mq_ref.update_param_version.remote(param_version)
    async def pause(
        self,
    ) -> None:
        """
        异步暂停队列
        """
        await self.mq_ref.pause.remote()
    async def resume(
        self,
    ) -> None:
        """
        异步恢复队列
        """
        await self.mq_ref.resume.remote()
    
    async def stop(
        self,
    ) -> None:
        """
        异步停止队列
        """
        await self.mq_ref.stop.remote()

    async def clear_queue(
        self,
    ) -> None:
        """
        异步清空队列
        """
        await self.mq_ref.clear_queue.remote()

    async def get_stats(
        self,
    ) -> dict[str, Any]:
        """
        异步获取队列统计信息
        """
        return await self.mq_ref.get_stats.remote()
    
    async def get_queue_size(
        self,
    ) -> int:
        """
        异步获取队列大小
        """
        return await self.mq_ref.get_queue_size.remote()
    
    async def is_empty(
        self,
    ) -> bool:
        """
        异步判断队列是否为空
        """
        return await self.mq_ref.is_empty.remote()

    async def is_full(
        self,
    ) -> bool:
        """
        异步判断队列是否满
        """
        return await self.mq_ref.is_full.remote()

    def get_sample_sync(self) -> Any | None:
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.mq_ref.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.mq_ref.get_stats.remote())

    def update_param_version_sync(self, version: int):
        """Update parameter version (async)"""
        return ray.get(self.mq_ref.update_param_version.remote(version))


async def example_usage():
    
    mq = MessageQueue.remote(
        queue_size=1000,
        step_off_threshold=3,
        log_interval=30
    )

    client = MessageQueueClient(mq)

    async def consumer():
        for i in range(100):
            sample_data = {"data": f"sample_{i}"}
            success = await client.put_sample(sample_data=sample_data, param_version=1)
            if success:
                print(f"Producer put sample {i}")
            await asyncio.sleep(0.1)

    async def producer():
        while True:
            batch = await client.get_batch(batch_size=10, timeout=1.0)
            if batch:
                print(f"Consumer got batch of size {len(batch)}")
            else:
                print("No samples available")
            await asyncio.sleep(0.5)
    await asyncio.gather(consumer(), producer())
if __name__ == "__main__":
    ray.init()
    asyncio.run(example_usage())