import copy
from typing import List, Optional, Sequence, Tuple

from omegaconf import DictConfig


def resolve_rollout_chunk_size(config) -> Optional[int]:
    es_cfg = getattr(config, "es_manager", None)
    val_cfg = getattr(es_cfg, "val", None) if es_cfg is not None else None
    if val_cfg is None:
        return None
    raw_chunk_size = getattr(val_cfg, "rollout_chunk_size", None)
    if raw_chunk_size in (None, ""):
        return None
    chunk_size = int(raw_chunk_size)
    return chunk_size if chunk_size > 0 else None


def iter_val_rollout_chunks(
    total_env_groups: int,
    start_group_index: int,
    chunk_size: Optional[int],
) -> List[Tuple[int, int, int]]:
    total_env_groups = int(total_env_groups)
    start_group_index = int(start_group_index)
    if chunk_size is None:
        return [(0, start_group_index, total_env_groups)]

    chunk_size = int(chunk_size)
    if total_env_groups <= 0 or chunk_size <= 0 or chunk_size >= total_env_groups:
        return [(0, start_group_index, total_env_groups)]

    chunks = []
    chunk_offset = 0
    while chunk_offset < total_env_groups:
        current_env_groups = min(chunk_size, total_env_groups - chunk_offset)
        chunks.append(
            (
                chunk_offset,
                start_group_index + chunk_offset,
                current_env_groups,
            )
        )
        chunk_offset += current_env_groups
    return chunks


def slice_env_group_config(
    tags: Sequence[str],
    n_groups: Sequence[int],
    chunk_offset: int,
    chunk_env_groups: int,
) -> Tuple[List[str], List[int]]:
    if len(tags) != len(n_groups):
        raise ValueError(
            f"env_configs.tags and env_configs.n_groups must have the same length, got "
            f"{len(tags)} and {len(n_groups)}"
        )

    chunk_tags: List[str] = []
    chunk_n_groups: List[int] = []
    chunk_end = chunk_offset + chunk_env_groups
    cursor = 0

    for tag, raw_n_group in zip(tags, n_groups):
        n_group = int(raw_n_group)
        next_cursor = cursor + n_group
        overlap_start = max(cursor, chunk_offset)
        overlap_end = min(next_cursor, chunk_end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            chunk_tags.append(str(tag))
            chunk_n_groups.append(int(overlap))
        cursor = next_cursor

    if sum(chunk_n_groups) != chunk_env_groups:
        raise ValueError(
            "Chunked env_config slicing produced an inconsistent group count: "
            f"expected {chunk_env_groups}, got {sum(chunk_n_groups)} from tags {chunk_tags}."
        )
    return chunk_tags, chunk_n_groups


def clone_config_for_val_chunk(
    config: DictConfig,
    *,
    chunk_offset: int,
    chunk_start_group_index: int,
    chunk_env_groups: int,
) -> DictConfig:
    chunk_config = copy.deepcopy(config)
    base_val_cfg = config.es_manager.val
    chunk_val_cfg = chunk_config.es_manager.val

    chunk_tags, chunk_n_groups = slice_env_group_config(
        tags=list(base_val_cfg.env_configs.tags),
        n_groups=list(base_val_cfg.env_configs.n_groups),
        chunk_offset=chunk_offset,
        chunk_env_groups=chunk_env_groups,
    )

    chunk_val_cfg.start_group_index = int(chunk_start_group_index)
    chunk_val_cfg.env_groups = int(chunk_env_groups)
    chunk_val_cfg.env_configs.tags = chunk_tags
    chunk_val_cfg.env_configs.n_groups = chunk_n_groups
    return chunk_config
