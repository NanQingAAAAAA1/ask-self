"""检索测试脚本。

检索是纯本地操作：问题用 DashScope 嵌入后，在 ChromaDB 中查询 Top-K 相似片段。
（嵌入 API 只发送问题/片段文本，向量索引与原始文件不出本机）

用法：
    python src/retrieve.py "我该不该换工作"
    python src/retrieve.py "焦虑" -k 10
    python src/retrieve.py "焦虑" --source "docs/日记（9月）.md" --source 日记补充（8月）
"""
import argparse
import sys
from typing import Dict, List, Optional

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR, ROOT, TOP_K
from embedding import embed_query


def get_collection():
    if not CHROMA_DIR.exists():
        raise RuntimeError(f"未找到向量库 {CHROMA_DIR}，请先运行 python src/ingest.py")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def _normalize_source(s: str) -> Optional[str]:
    """把「@ 引用」传来的路径统一成入库时 metadata.source 的格式。

    入库时 source = 相对**项目根**的 posix 路径（如 ``docs/日记（9月）.md``）。
    前端目录树给的是相对 **docs/** 的路径（如 ``日记（9月）.md`` 或 ``diary/2026``），
    因此这里统一补上 ``docs/`` 前缀；已经带前缀的原样返回。
    """
    s = (s or "").strip().replace("\\", "/").strip("/")
    if not s:
        return None
    if not s.startswith("docs/"):
        s = "docs/" + s
    return s


def _expand_to_files(sources: List[str]) -> List[str]:
    """把「引用范围」展开成一批**确切的文件 source 列表**。

    前端 @ 树给的既可能是文件（``日记（9月）.md``），也可能是文件夹（``subdir``）。
    文件夹要匹配「该目录下所有文件」，而 ChromaDB 1.5.x 的 where **不支持 ``$like``**
    （只支持 $eq/$ne/$in/$nin/$gt/$gte/$lt/$lte/$contains/$not_contains），
    所以先在本地把文件夹展开成具体文件路径，再用 ``$in`` 精确匹配 —— 语义等价且不依赖
    数据库的前缀匹配能力。

    返回：规范化后的 source 列表（相对项目根，如 ``docs/日记（9月）.md``）。
    """
    root = DOCS_DIR.resolve()
    out: List[str] = []
    seen = set()

    def _add(rel: str) -> None:
        if rel not in seen:
            seen.add(rel)
            out.append(rel)

    for raw in sources or []:
        s = _normalize_source(raw)
        if not s:
            continue
        # docs/xxx → 磁盘上的绝对路径
        try:
            target = (ROOT / s).resolve()
        except Exception:
            continue
        # 安全：只允许 docs/ 之内（防止前端传入越界路径）
        if target != root and root not in target.parents:
            continue
        if target.is_file():
            _add(s)                                   # 直接就是文件
        elif target.is_dir():
            for fp in target.rglob("*"):              # 文件夹 → 展开全部子文件
                if fp.is_file():
                    try:
                        _add(fp.relative_to(ROOT).as_posix())
                    except Exception:
                        continue
    return out


def _build_where(sources: List[str]):
    """把「引用范围」编成 ChromaDB 的 where 条件。

    - sources 为空/None → None（不过滤，全库检索）。
    - 每个元素既可指**文件**（docs/a.md），也可指**文件夹**（subdir）：
      文件夹先由 ``_expand_to_files()`` 展开成具体文件，再统一用 ``$in`` 精确匹配。
    - 展开后无有效文件 → None（等价于未限定，避免把结果过滤成空集）。
    - 只有一个文件 → 单键 ``{"source": {...}}``；多个 → ``$in``。
    """
    files = _expand_to_files(sources)
    if not files:
        return None
    if len(files) == 1:
        return {"source": files[0]}
    return {"source": {"$in": files}}


def retrieve(question: str, top_k: int = TOP_K,
             sources: Optional[List[str]] = None) -> List[Dict]:
    """返回 Top-K 片段：[{text, source, source_type, date, chunk_index, distance, score}]。

    - 跳过 frontmatter 中 private:true 的片段（不参与检索，也不会发给任何 API）。
    - ``sources`` 给定时**只在指定文件/文件夹范围内检索**（@ 引用）；
      为空则全库检索（保持原行为）。
    """
    collection = get_collection()
    qvec = embed_query(question)

    # 用户显式限定了范围时：先解析成确切文件集合。
    # 若范围解析不出任何文件（如路径写错、选中的文件夹是空的），**必须返回空结果**
    # 而不是退化成全库检索 —— 否则「严格限定」会静默失效，把库里其它文档也答出来。
    if sources:
        files = _expand_to_files(sources)
        if not files:
            return []
        where = {"source": files[0]} if len(files) == 1 else {"source": {"$in": files}}
    else:
        where = None

    # 多取一些，过滤 private 后再截断到 top_k
    # 注：带 where 过滤时总数会变少，n_results 仍需 <= 集合总数
    n = min(top_k * 2, max(collection.count(), 1))
    res = collection.query(
        query_embeddings=[qvec],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict] = []
    for text, meta, dist in zip(docs, metas, dists):
        if meta.get("private"):
            continue
        # cosine 距离 → 相似度分数（0-10，便于与重排分数对照展示）
        similarity = max(0.0, 1.0 - float(dist))
        out.append({
            "text": text,
            "source": meta.get("source", ""),
            "source_type": meta.get("source_type", ""),
            "date": meta.get("date", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": float(dist),
            "score": round(similarity * 10, 2),
        })

    # 先按距离排序，再截断到 top_k。
    # 注：原实现是「凑够 top_k 就 break」，那样会漏掉排序在后的高相关片段
    #     （ChromaDB 虽已按距离升序返回，但过滤 private 后需重新对齐）。
    out.sort(key=lambda x: x["distance"])
    return out[:top_k]


def main():
    parser = argparse.ArgumentParser(description="ChromaDB 检索测试")
    parser.add_argument("question", help="检索问题")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help=f"返回条数（默认 {TOP_K}）")
    parser.add_argument("-s", "--source", action="append", default=None,
                        help="限定检索范围（文件路径或文件夹，可重复指定）")
    args = parser.parse_args()

    try:
        results = retrieve(args.question, args.top_k, sources=args.source)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n查询：{args.question}")
    if args.source:
        print(f"限定范围：{', '.join(args.source)}")
    print(f"命中 {len(results)} 条：\n")
    for i, r in enumerate(results, 1):
        preview = r["text"][:90].replace("\n", " ")
        print(f"[{i}] {r['source']}  #{r['chunk_index']}  "
              f"相似度 {r['score']}  [{r['source_type']}] {r['date']}")
        print(f"    {preview}…\n")


if __name__ == "__main__":
    main()
