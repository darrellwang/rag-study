"""
RAG 核心逻辑的离线单元测试。

关键点：用 FakeEmbeddings 替代真实 embedding 模型，
因此这些测试不需要启动 LMStudio，也不需要联网，可以放进 CI。

生产经验：RAG 系统里真正容易出错的不是模型调用，而是
「去重逻辑」「按来源删除」「metadata 传递」「知识库枚举」这些管道细节。
它们恰好都是可以用假模型完全覆盖的，值得写死在测试里防回归。

运行:
  python -m unittest discover -s tests -v
  # 或 pytest tests -v
"""
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

# 必须在 import config 之前指定，config 在导入时就读取环境变量。
# 临时目录建在项目内而非 /tmp：部分 CI / 沙箱环境对 /tmp 只给读权限，
# Chroma 写 sqlite 会直接报 "readonly database"。
_TMP_DIR = tempfile.mkdtemp(prefix=".rag_test_", dir=str(BASE_DIR))
os.environ["CHROMA_PERSIST_DIR"] = _TMP_DIR
os.environ["DOCS_DIR"] = str(BASE_DIR / "data")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from rag.bm25 import BM25Index, tokenize  # noqa: E402
from rag.chunking import ChunkStrategy, describe_chunks, split_documents  # noqa: E402
from rag.evaluate import hit_at_k, mrr, ndcg_at_k, recall_at_k, keyword_coverage  # noqa: E402
from rag.fusion import reciprocal_rank_fusion, weighted_score_fusion  # noqa: E402
from rag.loaders import load_directory, load_file  # noqa: E402
from rag.pipeline import build_context  # noqa: E402
from rag.store import KnowledgeBase, chunk_id, delete_knowledge_base, list_knowledge_bases  # noqa: E402


class FakeEmbeddings(Embeddings):
    """确定性哈希嵌入。相同文本得到相同向量，够用来验证存储与检索管道。"""

    dim = 32

    def _vec(self, text: str):
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self.dim)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class TestTokenizeAndBM25(unittest.TestCase):
    def test_mixed_language_tokenize(self):
        tokens = tokenize("BM25 的 k1 参数")
        self.assertIn("bm25", tokens)        # ASCII 转小写
        self.assertIn("k1", tokens)
        self.assertIn("参", tokens)          # 汉字 unigram
        self.assertIn("参数", tokens)        # 汉字 bigram

    def test_bm25_ranks_relevant_doc_first(self):
        corpus = [
            "BM25 的 k1 参数控制词频饱和速度，b 参数控制长度归一化。",
            "向量检索使用 embedding 模型把文本映射为高维向量。",
            "今天天气不错，适合出门散步。",
        ]
        index = BM25Index(corpus)
        top = index.search("k1 和 b 参数", k=3)
        self.assertTrue(top, "应至少命中一条")
        self.assertEqual(top[0][0], 0)

    def test_bm25_ignores_unknown_terms(self):
        index = BM25Index(["检索增强生成"])
        self.assertEqual(index.search("完全不相干的词ZZZ"), [])

    def test_bm25_empty_corpus(self):
        self.assertEqual(BM25Index([]).search("任意查询"), [])


class TestFusion(unittest.TestCase):
    def test_rrf_favours_multi_channel_agreement(self):
        """只被单通道排第一的 x，应该输给被两个通道都召回的 b。"""
        fused = dict(reciprocal_rank_fusion([["x", "b"], ["b", "y"]]))
        self.assertGreater(fused["b"], fused["x"])

    def test_rrf_is_rank_only(self):
        """RRF 不看分值，只看名次，因此同名次贡献相同。"""
        a = dict(reciprocal_rank_fusion([["a", "b"]]))
        b = dict(reciprocal_rank_fusion([["c", "d"]]))
        self.assertAlmostEqual(a["a"], b["c"])

    def test_weighted_fusion_normalizes_scales(self):
        """BM25 分值(0~20) 与向量相似度(0~1) 量纲差异不应主导结果。"""
        fused = dict(
            weighted_score_fusion(
                [[("a", 0.9), ("b", 0.1)], [("b", 20.0), ("a", 1.0)]], [0.5, 0.5]
            )
        )
        self.assertAlmostEqual(fused["a"], fused["b"], places=6)


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(
                page_content="# 标题\n\n正文一。\n\n## 小节\n\n正文二。" + "填充内容。" * 80,
                metadata={"source": "doc.md"},
            )
        ]

    def test_recursive_respects_chunk_size(self):
        chunks = split_documents(self.docs, ChunkStrategy.RECURSIVE, chunk_size=120, chunk_overlap=20)
        self.assertTrue(chunks)
        # 递归切分在无可用分隔符时才会超限，留 1.5 倍余量
        self.assertLessEqual(max(len(c.page_content) for c in chunks), 180)

    def test_markdown_injects_heading_path(self):
        chunks = split_documents(self.docs, ChunkStrategy.MARKDOWN, chunk_size=200, chunk_overlap=20)
        self.assertTrue(any(c.metadata.get("heading_path") for c in chunks))
        with_path = [c for c in chunks if c.metadata.get("heading_path")][0]
        self.assertTrue(with_path.page_content.startswith("["))

    def test_non_markdown_falls_back(self):
        """.txt 文件里的 # 不该被当成标题。"""
        txt = [Document(page_content="C# 是一门语言。\n" * 20, metadata={"source": "a.txt"})]
        chunks = split_documents(txt, ChunkStrategy.MARKDOWN)
        self.assertTrue(chunks)
        self.assertFalse(any(c.metadata.get("heading_path") for c in chunks))

    def test_chunk_index_is_sequential(self):
        chunks = split_documents(self.docs, ChunkStrategy.RECURSIVE, chunk_size=120)
        self.assertEqual([c.metadata["chunk_index"] for c in chunks], list(range(len(chunks))))

    def test_describe_empty(self):
        self.assertEqual(describe_chunks([]), {"count": 0})


class TestLoaders(unittest.TestCase):
    def test_loads_project_data_dir(self):
        docs = load_directory(str(BASE_DIR / "data"))
        self.assertTrue(docs)
        for d in docs:
            self.assertIn("source", d.metadata)
            self.assertIn("file_name", d.metadata)

    def test_rejects_unsupported_ext(self):
        with self.assertRaises(ValueError):
            load_file("whatever.docx")

    def test_csv_becomes_one_doc_per_row(self):
        path = os.path.join(_TMP_DIR, "t.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("name,desc\nBM25,稀疏检索\nHNSW,向量索引\n")
        docs = load_file(path)
        self.assertEqual(len(docs), 2)
        self.assertIn("BM25", docs[0].page_content)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase("testkb", embeddings=FakeEmbeddings())
        self.kb.clear()
        self.docs = [
            Document(page_content="向量数据库用于存储 embedding。", metadata={"source": "a.md"}),
            Document(page_content="BM25 属于稀疏检索方法。", metadata={"source": "b.md"}),
        ]

    def tearDown(self):
        delete_knowledge_base("testkb")

    def test_chunk_id_is_content_addressed(self):
        self.assertEqual(chunk_id("kb", "a.md", "正文"), chunk_id("kb", "a.md", "正文"))
        self.assertNotEqual(chunk_id("kb", "a.md", "正文"), chunk_id("kb2", "a.md", "正文"))
        self.assertNotEqual(chunk_id("kb", "a.md", "正文"), chunk_id("kb", "a.md", "正文2"))

    def test_ingest_is_idempotent(self):
        first = self.kb.add_documents(self.docs, source_label="a.md")
        self.assertEqual(first.added, first.chunks)
        self.assertEqual(first.skipped, 0)

        second = self.kb.add_documents(self.docs, source_label="a.md")
        self.assertEqual(second.added, 0, "重复入库不应新增")
        self.assertEqual(second.skipped, second.chunks)
        self.assertEqual(self.kb.count(), first.total_in_kb)

    def test_delete_by_source(self):
        self.kb.add_documents([self.docs[0]], source_label="a.md")
        self.kb.add_documents([self.docs[1]], source_label="b.md")
        before = self.kb.count()
        removed = self.kb.delete_by_source("a.md")
        self.assertGreater(removed, 0)
        self.assertEqual(self.kb.count(), before - removed)
        self.assertNotIn("a.md", self.kb.list_sources())
        self.assertIn("b.md", self.kb.list_sources())

    def test_update_flow_replaces_content(self):
        """先删后插 = 增量更新，不应留下旧内容。"""
        self.kb.add_documents([Document(page_content="旧版本内容", metadata={})], source_label="c.md")
        self.kb.delete_by_source("c.md")
        self.kb.add_documents([Document(page_content="新版本内容", metadata={})], source_label="c.md")
        texts = [d.page_content for d in self.kb.all_documents()]
        self.assertIn("新版本内容", texts)
        self.assertNotIn("旧版本内容", texts)

    def test_metadata_scalars_only(self):
        """Chroma 不接受 list/dict metadata，入库前必须过滤掉。"""
        doc = Document(page_content="内容", metadata={"source": "d.md", "tags": ["a", "b"]})
        self.kb.add_documents([doc], source_label="d.md")
        for stored in self.kb.all_documents():
            for value in stored.metadata.values():
                self.assertIsInstance(value, (str, int, float, bool))

    def test_stats_shape(self):
        self.kb.add_documents(self.docs, source_label="a.md")
        st = self.kb.stats()
        self.assertEqual(st["kb_id"], "testkb")
        self.assertGreater(st["chunks"], 0)
        self.assertGreater(st["avg_chunk_chars"], 0)

    def test_uses_langchain_default_collection(self):
        """collection 名必须沿用 langchain-chroma 的默认值。
        改名会让旧知识库的数据读不到——Chroma 只会静默新建一个空 collection，
        表现为「库还在但内容没了」。"""
        from rag.store import DEFAULT_COLLECTION

        self.assertEqual(DEFAULT_COLLECTION, "langchain")
        self.assertEqual(self.kb.vs._collection.name, DEFAULT_COLLECTION)

    def test_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            KnowledgeBase("../evil", embeddings=FakeEmbeddings())

    def test_list_ignores_chroma_segment_dirs(self):
        """Chroma 的 UUID 段目录不含 chroma.sqlite3，不该被当成知识库。"""
        KnowledgeBase("realkb", embeddings=FakeEmbeddings())
        fake_segment = os.path.join(_TMP_DIR, "94f920a5-77ea-48a0-860e-d6e7f3c5d925")
        os.makedirs(fake_segment, exist_ok=True)
        open(os.path.join(fake_segment, "data_level0.bin"), "wb").close()
        try:
            kbs = list_knowledge_bases()
            self.assertIn("realkb", kbs)
            self.assertNotIn("94f920a5-77ea-48a0-860e-d6e7f3c5d925", kbs)
        finally:
            shutil.rmtree(fake_segment, ignore_errors=True)
            delete_knowledge_base("realkb")


class TestDocKeyAndScoreParsing(unittest.TestCase):
    def test_doc_key_ignores_internal_id(self):
        """向量通道的 Document 没有 _id，BM25 通道的有。同一 chunk 必须算同一个 key，
        否则融合后 top-k 会出现重复片段。"""
        from rag.retriever import _doc_key

        from_vector = Document(page_content="同一段正文", metadata={"source": "a.md"})
        from_bm25 = Document(page_content="同一段正文", metadata={"source": "a.md", "_id": "abc123"})
        self.assertEqual(_doc_key(from_vector), _doc_key(from_bm25))

    def test_doc_key_separates_different_sources(self):
        from rag.retriever import _doc_key

        a = Document(page_content="免责声明", metadata={"source": "a.md"})
        b = Document(page_content="免责声明", metadata={"source": "b.md"})
        self.assertNotEqual(_doc_key(a), _doc_key(b))

    def test_score_parsing_strips_think_block(self):
        """思维链里的数字不能被当成分数。"""
        from rag.rerank import _parse_score

        self.assertEqual(_parse_score("<think>评分标准是 0 到 10 分</think>9"), 9.0)

    def test_score_parsing_takes_last_int(self):
        from rag.rerank import _parse_score

        self.assertEqual(_parse_score("这段文档直接回答了问题，评分：8"), 8.0)

    def test_score_parsing_clamps_and_defaults(self):
        from rag.rerank import _parse_score

        self.assertEqual(_parse_score("99"), 10.0)
        self.assertEqual(_parse_score("无法判断"), 5.0)

    def test_config_label_distinguishes_fusion(self):
        """A/B 表格里两行同名配置等于没法比。"""
        from rag.pipeline import RagConfig

        rrf = RagConfig(mode="hybrid", fusion="rrf", top_k=4).label()
        weighted = RagConfig(mode="hybrid", fusion="weighted", top_k=4).label()
        self.assertNotEqual(rrf, weighted)


class TestAbstainDetection(unittest.TestCase):
    """用真实评估跑出来的答案做回归。误判拒答会直接把 abstain_rate 指标算废。"""

    REAL_ABSTAIN = "资料不足，无法回答"
    REAL_ANSWER_ABOUT_ABSTAIN = (
        "在知识库问答中设置拒答机制的原因如下：\n\n"
        "1. **防止错误答案**：对知识库问答而言，答错的代价远高于不答 [1]。\n"
        "2. **避免幻觉**：宁可拒答也不要一本正经地胡说 [1]。\n"
        "3. **确保诚实性**：当检索结果为空时，直接返回「知识库中没有相关信息」[1][4]。"
    )

    def test_detects_real_abstain(self):
        from rag.pipeline import looks_abstained

        self.assertTrue(looks_abstained(self.REAL_ABSTAIN))

    def test_does_not_misfire_on_answer_about_abstaining(self):
        """答案在讨论「拒答」这个话题，本身不是拒答。"""
        from rag.pipeline import looks_abstained

        self.assertFalse(looks_abstained(self.REAL_ANSWER_ABOUT_ABSTAIN))

    def test_cited_short_answer_is_not_abstain(self):
        from rag.pipeline import looks_abstained

        self.assertFalse(looks_abstained("资料中没有相关信息的说法见 [2]"))

    def test_empty_is_not_abstain(self):
        from rag.pipeline import looks_abstained

        self.assertFalse(looks_abstained(""))

    def test_mark_used_citations(self):
        from rag.pipeline import mark_used_citations

        cites = [{"index": 1}, {"index": 2}, {"index": 3}]
        marked = mark_used_citations("结论如此 [1]，另见 [3]。", cites)
        self.assertEqual([c["used"] for c in marked], [True, False, True])


class TestPipelineContext(unittest.TestCase):
    def test_citation_indices_align_with_context(self):
        docs = [
            Document(page_content="内容一", metadata={"file_name": "a.md", "page": 3}),
            Document(page_content="内容二", metadata={"file_name": "b.md", "heading_path": "X > Y"}),
        ]
        context, citations = build_context(docs)
        self.assertIn("[1] 来源: a.md", context)
        self.assertIn("第 3 页", context)
        self.assertIn("章节: X > Y", context)
        self.assertEqual([c["index"] for c in citations], [1, 2])
        self.assertEqual(citations[0]["source"], "a.md")

    def test_empty_docs(self):
        context, citations = build_context([])
        self.assertEqual(context, "")
        self.assertEqual(citations, [])


class TestEvalMetrics(unittest.TestCase):
    def test_hit_and_mrr(self):
        self.assertEqual(hit_at_k([False, True, False]), 1.0)
        self.assertEqual(hit_at_k([False, False]), 0.0)
        self.assertAlmostEqual(mrr([False, True]), 0.5)
        self.assertEqual(mrr([False, False]), 0.0)

    def test_recall(self):
        self.assertAlmostEqual(recall_at_k({"a.md"}, ["a.md", "b.md"]), 0.5)
        self.assertEqual(recall_at_k(set(), []), 0.0)

    def test_ndcg_prefers_earlier_hits(self):
        early = ndcg_at_k([True, False, False], 1)
        late = ndcg_at_k([False, False, True], 1)
        self.assertGreater(early, late)
        self.assertAlmostEqual(early, 1.0)

    def test_ndcg_never_exceeds_one(self):
        """一个来源常对应多个命中 chunk，此时 nDCG 仍必须 <= 1。"""
        self.assertLessEqual(ndcg_at_k([True, True, True], 1), 1.0)
        self.assertAlmostEqual(ndcg_at_k([True, True, True], 1), 1.0)
        self.assertLessEqual(ndcg_at_k([False, True, True], 1), 1.0)

    def test_ndcg_no_hits(self):
        self.assertEqual(ndcg_at_k([False, False], 2), 0.0)

    def test_keyword_coverage(self):
        self.assertAlmostEqual(keyword_coverage("包含 BM25 但没有另一个", ["bm25", "RRF"]), 0.5)
        self.assertIsNone(keyword_coverage("任意", []))


def tearDownModule():
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
