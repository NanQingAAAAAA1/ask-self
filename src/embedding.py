"""阿里云 DashScope 向量模型封装（OpenAI 兼容接口）。

按 `config.DASHSCOPE_EMBED_MODELS` 的优先级依次尝试，
某个模型报错（额度耗尽 / 403 / 限流 / 超时等）时自动降级到下一个。

对外保留两个接口，ingest（写库）与 retrieve（查询）共用：
    - embed_documents(texts)：批量嵌入文档片段
    - embed_query(text)：嵌入单条查询

⚠️ 维度对齐（重要）：
    不同嵌入模型的向量维度可能不同（如 1024 / 768），而 ChromaDB collection 的
    维度在首次写入时即固定。**切换模型后必须删除 chroma_db/ 或重新建库**
    （ingest.py 默认 reset=True 会重建 collection），否则会因维度不一致报错。
    为避免「入库用 A 模型、检索用 B 模型」导致维度不匹配，本模块会把实际使用的
    模型名写入 chroma_db/embedding_model.txt，检索时优先复用同一模型。

用法（自测）：
    python src/embedding.py "测试文本"
"""
import sys
from typing import List

from openai import OpenAI

from config import (
    CHROMA_DIR,
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_EMBED_MODELS,
)


class EmbeddingError(RuntimeError):
    """所有候选嵌入模型均调用失败。"""


def _log(msg: str) -> None:
    print(f"[embed] {msg}", file=sys.stderr)


_MARKER = CHROMA_DIR / "embedding_model.txt"   # 记录实际使用的嵌入模型
BATCH_SIZE = 10                                 # 单次请求文本条数（DashScope 较保守）


def _client() -> OpenAI:
    if not DASHSCOPE_API_KEY:
        raise EmbeddingError("缺少 DASHSCOPE_API_KEY，请在 .env 中配置")
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL, timeout=60)


def _embed(texts: List[str], model: str) -> List[List[float]]:
    """调用单个模型嵌入一批文本，返回与输入等长的向量列表。"""
    resp = _client().embeddings.create(model=model, input=texts)
    items = sorted(resp.data, key=lambda d: d.index)   # 保证顺序与输入一致
    return [it.embedding for it in items]


def _models() -> List[str]:
    return [m for m in DASHSCOPE_EMBED_MODELS if m]


def _save_marker(model: str) -> None:
    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _MARKER.write_text(model, encoding="utf-8")
    except Exception:
        pass


def _load_marker() -> str:
    try:
        return _MARKER.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _first_working(batch: List[str]):
    """按优先级逐个尝试模型，返回 (model, vectors)；全部失败抛 EmbeddingError。"""
    models = _models()
    errors = []
    for i, m in enumerate(models):
        try:
            vecs = _embed(batch, m)
            if i > 0:
                _log(f"已降级：{models[0]} 不可用，改用 {m}")
            return m, vecs
        except Exception as e:          # 额度耗尽 / 403 / 限流 / 超时 …
            errors.append(f"{m}: {e}")
            _log(f"模型 {m} 调用失败：{e}")
    raise EmbeddingError("所有 DashScope 嵌入模型均调用失败：\n  " + "\n  ".join(errors))


def embed_documents(texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """批量嵌入文档片段（入库用）。

    第一批按优先级探测出可用模型后，后续批次**沿用同一模型**——
    避免中途降级导致同一 collection 内向量维度不一致。
    """
    if not texts:
        return []

    model = None
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        if model is None:
            model, vecs = _first_working(batch)
        else:
            vecs = _embed(batch, model)
        out.extend(vecs)

    if model:
        _save_marker(model)
        _log(f"入库嵌入完成：模型={model}，共 {len(out)} 条，维度={len(out[0]) if out else 0}")
    return out


def embed_query(text: str) -> List[float]:
    """嵌入单条查询（检索用）。

    优先使用入库时记录的模型（保证维度与 collection 一致）；若无记录或该模型
    调用失败，则按优先级依次回退。
    """
    ordered: List[str] = []
    marker = _load_marker()
    if marker:
        ordered.append(marker)
    ordered += [m for m in _models() if m not in ordered]

    errors = []
    for m in ordered:
        try:
            vec = _embed([text], m)[0]
            _log(f"查询嵌入：模型={m}，维度={len(vec)}")
            return vec
        except Exception as e:
            errors.append(f"{m}: {e}")
            _log(f"模型 {m} 调用失败：{e}")
    raise EmbeddingError("所有 DashScope 嵌入模型均调用失败：\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else "测试文本"
    vec = embed_query(txt)
    print(f"嵌入成功，向量维度：{len(vec)}")
