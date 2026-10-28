"""一键录入 docs/ 下所有 Markdown 文件。

流程（对应计划 4.4）：
  1. 遍历 docs/ 下所有 .md
  2. 解析 YAML frontmatter（如有）
  3. 文件名前缀推断 source_type、文件名中提取 date
  4. 句子边界累积切块（RecursiveCharacterTextSplitter），过滤碎片块
  5. 调用 DashScope 批量嵌入
  6. 写入 ChromaDB PersistentClient

用法：
    python src/ingest.py                # 整体重建（清空集合再录入）
    python src/ingest.py --keep         # 保留已有数据，仅追加
    python src/ingest.py --incremental  # 增量：已录入过的片段直接复用旧向量，只嵌入新增/变化的
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_SOURCE_TYPE,
    DOCS_DIR,
    MIN_CHUNK_LEN,
    PREFIX_TYPE,
    ROOT,
)
from embedding import embed_documents

# ---------- 切块分隔符（优先级从高到低，对应计划 4.2） ----------
SEPARATORS = [
    "\n\n",                                        # 段落
    "\n",                                          # 换行
    "。", "！", "？", "；",                        # 中文句末标点
    ".", "!", "?", ";",                            # 英文句末标点
    "，", ",",                                     # 逗号
    " ",                                           # 空格
    "",                                            # 字符（兜底）
]

# 容错：允许分隔行带多余空白或被反斜杠转义（\---），并兼容 CRLF 换行
FRONTMATTER_RE = re.compile(
    r"^\s*\\?---[ \t]*\r?\n(.*?)\r?\n[ \t]*\\?---[ \t]*\r?\n",
    re.DOTALL,
)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_frontmatter(text: str) -> (Dict, str):
    """解析 YAML frontmatter，返回 (元数据字典, 正文)。无 frontmatter 时返回 ({}, 原文)。"""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[m.end():]


def infer_source_type(filename: str) -> str:
    """从文件名前缀推断 source_type，不符合约定则返回 docs。"""
    for prefix, stype in PREFIX_TYPE.items():
        if filename.startswith(prefix):
            return stype
    return DEFAULT_SOURCE_TYPE


def infer_date(filename: str) -> Optional[str]:
    """从文件名中提取 YYYY-MM-DD 日期。"""
    m = DATE_RE.search(filename)
    return m.group(1) if m else None


def build_splitter() -> RecursiveCharacterTextSplitter:
    """句子边界累积切分器：句号保留在块末尾，语义完整。"""
    return RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator="end",
        length_function=len,
    )


def load_docs() -> List[Dict]:
    """遍历 docs/，产出带元数据的待嵌入片段列表。"""
    splitter = build_splitter()
    records: List[Dict] = []

    if not DOCS_DIR.exists():
        print(f"[warn] docs 目录不存在：{DOCS_DIR}")
        return records

    files = sorted(DOCS_DIR.rglob("*.md"))
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_frontmatter(raw)

        rel = fp.relative_to(ROOT).as_posix()          # 如 docs/diary-2024-01-15.md
        # frontmatter 覆盖文件名推断
        source_type = meta.get("type") or infer_source_type(fp.name)
        date = meta.get("date") or infer_date(fp.name)
        if date is not None:
            date = str(date)
        private = bool(meta.get("private", False))

        chunks = splitter.split_text(body)
        # 碎片过滤：太短的块（标题行、空行残留）丢弃
        chunks = [c.strip() for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]
        if not chunks:
            print(f"[skip] {rel} 无有效片段（均 < {MIN_CHUNK_LEN} 字）")
            continue

        for idx, chunk in enumerate(chunks):
            records.append({
                "text": chunk,
                "source": rel,
                "source_type": source_type,
                "date": date or "",
                "chunk_index": idx,
                "char_count": len(chunk),
                "private": private,
                "chash": chunk_hash(chunk),      # 内容指纹，增量录入用于去重
            })
        print(f"[ok] {rel} → {len(chunks)} 片段")

    print(f"\n共 {len(files)} 个文件，产出 {len(records)} 个片段")
    return records


def chunk_hash(text: str) -> str:
    """片段内容的 SHA1 指纹（用于增量录入时判断"这段是否已录过"）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def get_collection(reset: bool = False):
    """获取（或重建）ChromaDB 集合。"""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[reset] 已清空集合 {COLLECTION_NAME}")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest(reset: bool = True) -> int:
    """录入主流程，返回写入片段数。"""
    records = load_docs()
    if not records:
        print("没有可录入的片段。")
        return 0

    collection = get_collection(reset=reset)

    texts = [r["text"] for r in records]
    print(f"\n正在调用 DashScope 嵌入模型嵌入 {len(texts)} 个片段…")
    vectors = embed_documents(texts)

    ids = [f"{r['source']}#{r['chunk_index']}" for r in records]
    metadatas = [{
        "source": r["source"],
        "source_type": r["source_type"],
        "date": r["date"],
        "chunk_index": r["chunk_index"],
        "char_count": r["char_count"],
        "private": r["private"],
        "chash": r["chash"],          # 内容指纹，供后续 --incremental 去重
    } for r in records]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    print(f"[done] 已写入 {len(ids)} 个片段到 {CHROMA_DIR}")
    return len(ids)


def _load_existing_hashes() -> Dict[str, Tuple[str, List[float], Dict]]:
    """读取现有集合，返回 {内容指纹: (旧id, 旧向量, 旧metadata)}。

    若集合不存在/为空，返回空字典。
    """
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        return {}

    try:
        got = col.get(include=["embeddings", "metadatas"])
    except Exception:
        return {}

    ids = got.get("ids") or []
    embs = got.get("embeddings")
    metas = got.get("metadatas") or []
    if embs is None:
        return {}

    out: Dict[str, Tuple[str, List[float], Dict]] = {}
    for i, _id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        ch = meta.get("chash")
        if not ch:
            continue
        vec = embs[i].tolist() if hasattr(embs[i], "tolist") else list(embs[i])
        out[ch] = (_id, vec, meta)
    return out


def ingest_incremental() -> Dict[str, int]:
    """增量录入：**已录入过的片段复用旧向量，只对新增/变化的片段调用嵌入 API**。

    去重依据是片段内容的 SHA1 指纹（chash）：
      - 指纹在旧库里 → 直接复用旧向量（不花 token）；
      - 指纹是新出现的 → 批量嵌入；
      - 旧库里有、新扫描结果里没有的 → 视为已删除，不再写入（自动清理）。

    返回：{total, reused, embedded, skipped_files}
    """
    records = load_docs()

    old = _load_existing_hashes()
    print(f"[incremental] 旧库已有 {len(old)} 个片段可复用比对")

    reuse_ids, reuse_vecs, reuse_metas, reuse_docs = [], [], [], []
    need: List[Dict] = []
    seen: set = set()

    for r in records:
        ch = r["chash"]
        if ch in seen:                       # 本次扫描内部重复，直接跳过
            continue
        seen.add(ch)
        if ch in old:                        # 命中旧库 → 复用向量，零 API 成本
            reuse_ids.append(old[ch][0])
            reuse_vecs.append(old[ch][1])
            reuse_metas.append({k: v for k, v in r.items() if k != "text"})
            reuse_docs.append(r["text"])
        else:
            need.append(r)

    reused = len(reuse_ids)
    print(f"[incremental] 复用 {reused} 个旧片段（跳过嵌入），需新嵌入 {len(need)} 个片段")

    new_ids, new_vecs, new_docs, new_metas = [], [], [], []
    if need:
        texts = [r["text"] for r in need]
        print(f"正在调用 DashScope 嵌入 {len(texts)} 个新片段…")
        vectors = embed_documents(texts)
        for r, vec in zip(need, vectors):
            new_ids.append(f"{r['source']}#{r['chunk_index']}")
            new_vecs.append(vec)
            new_docs.append(r["text"])
            new_metas.append({k: v for k, v in r.items() if k != "text"})

    # 整体重建集合：先清空，再写入「复用 + 新嵌入」，保证删除的文件被清理
    collection = get_collection(reset=True)
    all_ids = reuse_ids + new_ids
    all_vecs = reuse_vecs + new_vecs
    all_docs = reuse_docs + new_docs
    all_metas = reuse_metas + new_metas
    if all_ids:
        collection.add(ids=all_ids, embeddings=all_vecs,
                       documents=all_docs, metadatas=all_metas)

    print(f"[done] 增量完成：共 {len(all_ids)} 片段"
          f"（复用 {reused} / 新嵌入 {len(new_ids)}）")
    return {
        "total": len(all_ids),
        "reused": reused,
        "embedded": len(new_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="录入 docs/ 到 ChromaDB")
    parser.add_argument("--keep", action="store_true", help="保留已有数据，仅追加")
    parser.add_argument("--incremental", action="store_true",
                        help="增量：已录入过的片段复用旧向量，只嵌入新增/变化的（省 token）")
    args = parser.parse_args()

    try:
        if args.incremental:
            stats = ingest_incremental()
            print(f"[summary] 共 {stats['total']} 片段；"
                  f"复用 {stats['reused']}；新嵌入 {stats['embedded']}")
        else:
            ingest(reset=not args.keep)
    except Exception as e:
        print(f"[error] 录入失败：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
