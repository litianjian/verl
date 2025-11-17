# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import torch
from copy import deepcopy

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table


from verl.models.mcore.util import postprocess_packed_seqs, preprocess_packed_seqs
from verl.utils.megatron.router_replay_patch import RouterReplay, RoutingMode

def merge_router_topk_indices(attention_mask, input_ids, mini_layer_topk_idx_list, tf_config, vp_rank=None):
    """
    Merge recorded topk indices from all router instances.

    Args:
        attention_mask: Attention mask tensor
        input_ids: Input IDs tensor
        mini_layer_topk_idx_list: List to append the merged topk indices

    Returns:
        None (appends result to mini_layer_topk_idx_list)
    """
    with torch.no_grad():
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        layers_topk_idx = []
        for router in router_instances_list:
            layers_topk_idx.append(router.recorded_topk_idx)  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).cuda()
        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = gather_from_sequence_parallel_region(
            layers_topk_idx, tensor_parallel_output_grad=False
        ).unsqueeze(0)

        batch_size, seq_len = attention_mask.shape[:2]
        _, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        layers_topk_idx = postprocess_packed_seqs(
            layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
        )
        mini_layer_topk_idx_list.append(layers_topk_idx)


def set_router_replay_data(layers_topk_idx, attention_mask, tf_config, vp_rank=None):
    """
    Set router replay data for replay mode.

    Args:
        layers_topk_idx: Topk indices tensor to be set as replay data
        attention_mask: Attention mask tensor

    Returns:
        None (sets replay data in RouterReplay)
    """
    with torch.no_grad():
        layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.cuda().squeeze(dim=0)
        ).unsqueeze(dim=0)

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        for i,router in enumerate(router_instances_list):
            router.set_target_indices(layers_topk_idx_reshape[i].to(torch.int64))

def reorder_and_merge_vpp_layers(
    t: torch.Tensor,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    将形状为 [bs*vpp_size, max_token_len, layer_num/vpp_size, topk] 的张量，
    先根据调度表 schedule_table 将第0维（virtual_microbatch_id 顺序）重排，
    使同一 model_chunk_id 的条目在第0维上变得连续；随后按层维度合并，
    得到形状 [bs, max_token_len, layer_num, topk]。

    参数:
    - t: 输入张量，形状 [bs*vpp_size, max_token_len, layer_num/vpp_size, topk]
    - num_microbatches: bs（每个流水阶段的微批次数）
    - vpp_size: 虚拟流水并行（模型分块）数，即 len(model)
    - microbatch_group_size_per_vp_stage: 每个虚拟阶段连续处理的微批数量

    返回:
    - 输出张量，形状 [bs, max_token_len, layer_num, topk]
    """

    if t.dim() != 4:
        raise ValueError(f"expect a 4D tensor, got dim={t.dim()} and shape={tuple(t.shape)}")
    bs_vpp, max_token_len, layer_per_vpp, topk = t.shape

    expected_bs_vpp = num_microbatches * vpp_size
    if bs_vpp != expected_bs_vpp:
        raise ValueError(
            f"first dim (bs*vpp_size={bs_vpp}) must equal num_microbatches*vpp_size={expected_bs_vpp}"
        )
    if vpp_size <= 0:
        raise ValueError(f"vpp_size must be positive, got {vpp_size}")

    # 1) 构建调度表: 每个 virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)
    if len(schedule_table) != expected_bs_vpp:
        raise RuntimeError(
            f"schedule_table length {len(schedule_table)} mismatch total virtual microbatches {expected_bs_vpp}"
        )

    # 2) 依据 model_chunk_id 分组，生成重排索引，使同一 chunk 的条目在第0维上连续
    indices_by_chunk = [[] for _ in range(vpp_size)]
    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        indices_by_chunk[chunk_id].append(vidx)
    reorder_indices = [idx for chunk_id in range(vpp_size) for idx in indices_by_chunk[chunk_id]]

    index_tensor = torch.tensor(reorder_indices, dtype=torch.long, device=t.device)
    t_reordered = torch.index_select(t, dim=0, index=index_tensor)

    # 3) 重排后 reshape 并在层维上合并
    bs = num_microbatches
    # 视图: [vpp_size, bs, max_token_len, layer_per_vpp, topk]
    t_view = t_reordered.contiguous().view(vpp_size, bs, max_token_len, layer_per_vpp, topk)
    # 交换维度 -> [bs, max_token_len, vpp_size, layer_per_vpp, topk]
    t_perm = t_view.permute(1, 2, 0, 3, 4).contiguous()
    # 合并 (vpp_size, layer_per_vpp) -> layer_num
    out = t_perm.view(bs, max_token_len, vpp_size * layer_per_vpp, topk)

    # 形状校验
    if out.shape != (bs, max_token_len, vpp_size * layer_per_vpp, topk):
        raise RuntimeError(
            f"unexpected output shape {tuple(out.shape)}; "
            f"expected ({bs}, {max_token_len}, {vpp_size * layer_per_vpp}, {topk})"
        )
    return out


def compute_pipeline_layer_assignment(tf_config):
    #Todo: 应该可以使用megatron官方API替代
    """
    计算每个流水并行 rank（以及每个虚拟流水段）的全局层索引区间与层数。
    返回: dict[(pp_rank, vp_stage)] = {"start": int, "end": int, "count": int}
    """
    num_layers = tf_config.num_layers
    pp_size = tf_config.pipeline_model_parallel_size
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    first_stage_layers = tf_config.num_layers_in_first_pipeline_stage
    last_stage_layers = tf_config.num_layers_in_last_pipeline_stage

    assignments = {}

    if pp_size is None or pp_size <= 1:
        # 单段流水，无需切分。
        if vp_size is not None and vp_size > 1:
            # 虚拟流水要求在每物理段内部再均分。
            assert num_layers % vp_size == 0, "num_layers 必须可被虚拟流水大小整除"
            per_vp = num_layers // vp_size
            offset = 0
            for j in range(vp_size):
                assignments[(0, j)] = {"start": offset, "end": offset + per_vp, "count": per_vp}
                offset += per_vp
        else:
            assignments[(0, 0)] = {"start": 0, "end": num_layers, "count": num_layers}
        return assignments

    # 非布局模式下的平均/不均匀切分
    if first_stage_layers is None and last_stage_layers is None:
        # 平均切分到各物理段
        assert num_layers % pp_size == 0, "num_layers 必须可被 pipeline_model_parallel_size 整除"
        per_rank_total = num_layers // pp_size
        if vp_size is None:
            # 无虚拟流水
            for i in range(pp_size):
                start_i = i * per_rank_total
                end_i = start_i + per_rank_total
                assignments[(i, 0)] = {"start": start_i, "end": end_i, "count": per_rank_total}
        else:
            # 有虚拟流水: 每物理段再均分到 vp_size 个子段
            assert per_rank_total % vp_size == 0, "每段层数必须可被虚拟流水大小整除"
            per_vp = per_rank_total // vp_size
            for i in range(pp_size):
                base = i * per_rank_total
                for j in range(vp_size):
                    start_ij = base + j * per_vp
                    assignments[(i, j)] = {"start": start_ij, "end": start_ij + per_vp, "count": per_vp}
        return assignments
    else:
        # 不均匀首/末段配置
        assert first_stage_layers is not None and last_stage_layers is not None, "必须同时提供首段与末段层数"
        assert pp_size >= 2, "不均匀切分至少需要两个物理段"
        mid_ranks = pp_size - 2
        remaining = num_layers - first_stage_layers - last_stage_layers
        if mid_ranks > 0:
            assert remaining % mid_ranks == 0, "中间段总层数必须能平均分配到各中间物理段"
            per_mid_total = remaining // mid_ranks
        else:
            per_mid_total = 0

        # 计算每物理段的总层区间
        start = 0
        phys_ranges = []  # [(start, end)] for i in [0..pp_size-1]
        for i in range(pp_size):
            if i == 0:
                total_i = first_stage_layers
            elif i == pp_size - 1:
                total_i = last_stage_layers
            else:
                total_i = per_mid_total
            phys_ranges.append((start, start + total_i))
            start += total_i

        # 虚拟流水均分每物理段
        if vp_size is None:
            for i in range(pp_size):
                s, e = phys_ranges[i]
                assignments[(i, 0)] = {"start": s, "end": e, "count": e - s}
        else:
            for i in range(pp_size):
                s, e = phys_ranges[i]
                total_i = e - s
                assert total_i % vp_size == 0, "物理段的层数必须可被虚拟流水大小整除"
                per_vp = total_i // vp_size
                for j in range(vp_size):
                    start_ij = s + j * per_vp
                    assignments[(i, j)] = {"start": start_ij, "end": start_ij + per_vp, "count": per_vp}
        return assignments


def get_current_rank_layer_info(tf_config, vp_rank = None):
    #设置为none时，vp_rank为0
    """返回当前进程 (pp_rank, vp_stage) 的层区间与层数，以及整体分配表。"""
    assignments = compute_pipeline_layer_assignment(tf_config)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_rank == None:
        vp_rank = mpu.get_virtual_pipeline_model_parallel_rank() if vp_size is not None else 0

    local = assignments[(pp_rank, vp_rank)]
    return local, assignments


def build_local_router_map(global_layers_router_map, local_range):
    """
    根据当前 rank 的层区间，将全局路由表切分为本地路由表。
    支持的输入类型：
    - torch.Tensor: 1D 或以层为第一维的 2D/ND 张量，按 [start:end) 切片第一维
    - list/tuple: 按 [start:end) 切片
    - dict: 保留键在 [start:end) 的条目
    返回与输入类型一致的本地子集。
    """
    s = local_range["start"]
    e = local_range["end"]

    if isinstance(global_layers_router_map, torch.Tensor):
        if global_layers_router_map.dim() == 0:
            raise ValueError("global_layers_router_map 为标量，无法按层切分")
        # 将第一维视为层维
        global_layers_router_map = global_layers_router_map.permute(2, 0, 1, 3)
        slc = [slice(None)] * global_layers_router_map.dim()
        slc[0] = slice(s, e)
        return global_layers_router_map[tuple(slc)].clone().permute(1, 2, 0, 3)
    else:
        raise TypeError("不支持的 global_layers_router_map 类型: {}".format(type(global_layers_router_map)))

def pp_gather(local_layers_router_map, tf_config):
        #Todo:考虑不均匀分配模式
        """
        收集所有 pp_rank 的本地路由表到全局路由表。
        """
        pp_size = tf_config.pipeline_model_parallel_size
        if pp_size <= 1:
            return local_layers_router_map

        pp_group = mpu.get_pipeline_model_parallel_group()
        world_size = torch.distributed.get_world_size(pp_group)

        local_layers_router_map = local_layers_router_map.cuda()
        # layers_topk_idx = layers_topk_idx.permute(2,0,1,3).cuda().contiguous() #local_layer_num, bs, max_seq_len,topk
        layers_topk_idx_global_list = [torch.empty(size=local_layers_router_map.shape, \
                                                    dtype=local_layers_router_map.dtype,   \
                                                    device=local_layers_router_map.device ) \
                                                    for _ in range(world_size) ] 
        torch.distributed.all_gather(
            tensor=local_layers_router_map,
            tensor_list=layers_topk_idx_global_list,
            group=pp_group,
            async_op=False,
        )
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")
        return global_router_map

def pp_dispatch(global_layers_router_map, tf_config):
    '''
        global_layers_router_map:全局路由表,[bs, max_seq_len, num_layers, top-k]

        return:
            local_router_map:本pp_rank的路由表,[bs, max_seq_len, pp_rank_layers, top-k]

    '''
    pp_size = tf_config.pipeline_model_parallel_size
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if pp_size <= 1:
        return global_layers_router_map
    
    # 根据 tf_config 计算当前 rank 的层信息，并切分路由表
    local_info, all_assignments = get_current_rank_layer_info(tf_config)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    if  vp_size is None or vp_size<=1:
        local_router_map = build_local_router_map(global_layers_router_map, local_info)
    else:
        local_info = all_assignments[(pp_rank, 0)]
        count = local_info["count"]
        sta = local_info["start"]
        pp_layer_count = count * vp_size
        local_info["end"] = sta + pp_layer_count
        local_router_map = build_local_router_map(global_layers_router_map, local_info)
    return local_router_map



class RouterReplayHelper:
    """Helper class to simplify router replay mode checking."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        local, all_assignments = get_current_rank_layer_info(tf_config, vp_rank)
        vp_size = tf_config.virtual_pipeline_model_parallel_size
        pp_size = tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_local = deepcopy(local)
        #开启vpp
        if not (pp_size <= 1 or vp_size is None or vp_size<=1):
            local_t = all_assignments[(pp_rank, 0)]
            pp_local = deepcopy(local_t)

            count = local_t["count"]
            sta = local_t["start"]
            pp_layer_count = count * vp_size

            pp_local["start"] = sta
            pp_local["end"] = sta + pp_layer_count
            pp_local["count"] = pp_layer_count
        sta, end = local["start"] - pp_local["start"], local["end"] - pp_local["start"]
        router_instances_list = RouterReplay.router_instances[sta:end]
        return router_instances_list

    @staticmethod
    def is_r2_record_mode(tf_config, vp_rank=None) -> bool:
        """Check if current mode is R2 RECORD."""
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].routing_mode == RoutingMode.RECORD
        )

    @staticmethod
    def is_replay_forward_mode(tf_config, vp_rank=None) -> bool:
        """Check if current mode is REPLAY_FORWARD."""
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].routing_mode == RoutingMode.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_mode(tf_config, vp_rank=None) -> bool:
        """Check if current mode is REPLAY_BACKWARD."""
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].routing_mode == RoutingMode.REPLAY_BACKWARD
        )

    @staticmethod
    def is_r2_or_r3_mode(router_replay) -> bool:
        """Check if current mode is R2 or R3."""
        return router_replay.mode in ["R2", "R3"]
