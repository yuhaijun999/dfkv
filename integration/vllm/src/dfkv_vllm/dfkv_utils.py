# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm (mooncake/mooncake_utils.py): only the
# DP-engine-index helper is needed; dfkv handles its own RDMA bootstrap, so the
# Mooncake transfer-engine / bootstrap-server machinery is dropped.
"""Small helpers for the dfkv vLLM connector."""
from vllm.config import ParallelConfig


def get_dp_engine_index(parallel_config: ParallelConfig) -> int:
    """Return the per-engine DP index used for connector side channels."""
    if parallel_config.local_engines_only:
        assert parallel_config.data_parallel_rank_local is not None
        return parallel_config.data_parallel_rank_local
    return parallel_config.data_parallel_index
