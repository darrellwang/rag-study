"""
多路召回结果融合。

向量检索和 BM25 的得分不在同一个量纲上（余弦距离 vs BM25 分值，
后者上界还随语料变化），直接加权相加是错的。两种正确做法：

1. RRF（Reciprocal Rank Fusion）—— 只用名次，不用原始分值
       score(d) = Σ_通道 1 / (k + rank_通道(d))
   k 默认 60（原论文取值）。它的作用是压缩头部差距：
   rank=1 与 rank=2 的贡献差 (1/61 vs 1/62) 很小，
   因此「被多个通道同时召回」比「在单个通道排第一」更重要——
   这正是我们想要的鲁棒性。RRF 无需调参，是默认首选。

2. 加权分数融合 —— 各通道先归一化到 [0,1] 再加权
   能表达「向量通道更可信」这类先验，但归一化对异常值敏感，
   且权重需要用评估集调。在有标注数据时才值得用。
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]], k: int = 60
) -> List[Tuple[Hashable, float]]:
    """对多个「已排序的 id 列表」做 RRF 融合，返回 (id, 融合分) 降序列表。"""
    fused: Dict[Hashable, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _min_max(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:  # 全部相同，退化为均等
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def weighted_score_fusion(
    scored_lists: Sequence[Sequence[Tuple[Hashable, float]]],
    weights: Sequence[float],
) -> List[Tuple[Hashable, float]]:
    """各通道 min-max 归一化后加权求和。

    scored_lists: 每个通道的 [(id, 原始分), ...]，分值越大越相关
    weights:      与通道一一对应的权重
    """
    if len(scored_lists) != len(weights):
        raise ValueError("scored_lists 与 weights 长度必须一致")

    fused: Dict[Hashable, float] = {}
    for channel, weight in zip(scored_lists, weights):
        ids = [i for i, _ in channel]
        norm = _min_max([s for _, s in channel])
        for doc_id, s in zip(ids, norm):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * s
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
