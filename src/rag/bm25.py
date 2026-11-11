"""
BM25 稀疏检索（纯 Python 实现，无第三方依赖）。

为什么向量检索之外还需要 BM25？
向量检索擅长语义近似，但对「精确串」很弱：
  - 专有名词、错误码（ERR_5021）、API 名（build_or_load_kb）、版本号
  - 训练语料里没见过的内部术语
这些场景下 embedding 会把它们映射到一个模糊的位置，而 BM25 靠字面命中，
稳定得多。所以生产 RAG 几乎都是「向量 + BM25」双通道召回后融合。

BM25 打分公式（Okapi BM25）：

    score(q, d) = Σ_{t∈q} IDF(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

    IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

    f(t,d) 词 t 在文档 d 中的频次；|d| 文档长度；avgdl 平均文档长度
    k1 控制词频饱和速度（词出现 10 次并不比 5 次相关一倍）
    b  控制长度归一化强度（b=0 不归一，b=1 完全归一）

中文分词：不引入 jieba 等依赖，改用「ASCII 词 + 汉字 unigram/bigram」。
bigram 能捕捉到「向量-量数-数据-据库」这类组合，对检索召回已足够好，
且天然规避了分词器切错词的问题。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_ASCII_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """把文本切成检索词：ASCII 词（小写）+ 汉字 unigram + 汉字 bigram。"""
    tokens = [w.lower() for w in _ASCII_WORD.findall(text)]
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)  # unigram
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigram
    return tokens


class BM25Index:
    """内存 BM25 索引。

    适用规模：数万 chunk 以内（本地知识库的典型量级）。
    再大就该换 Elasticsearch / OpenSearch 的倒排索引了。
    """

    def __init__(self, corpus: Sequence[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus)
        self.term_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        doc_freq: Counter = Counter()

        for text in corpus:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            doc_freq.update(tf.keys())  # 每篇只计一次，得到文档频次

        self.avg_doc_len = (sum(self.doc_lens) / self.doc_count) if self.doc_count else 0.0
        # 预计算 IDF，查询时只做乘加
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.doc_count - n + 0.5) / (n + 0.5))
            for term, n in doc_freq.items()
        }

    def score(self, query: str) -> List[float]:
        """返回查询对每篇文档的 BM25 得分。"""
        scores = [0.0] * self.doc_count
        if not self.doc_count:
            return scores

        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:  # 语料里不存在该词，贡献为 0
                continue
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * (self.doc_lens[i] / self.avg_doc_len)
                scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)
        return scores

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """返回 top-k 的 (文档下标, 得分)，已按得分降序，过滤 0 分。"""
        scored = [(i, s) for i, s in enumerate(self.score(query)) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
