"""本地 API 服务：把 run_agent(...) 包成一个 HTTP 接口，给前端页面（frontend/index.html）用。

用法：
    python src/api.py                 # 默认 127.0.0.1:8000
    python src/api.py --port 8800     # 换端口

接口：
    GET  /api/health     → {"ok": true, "service": "askself", ...}
    POST /api/chat       → 透传 run_agent 的返回字段（answer / citations / ...）
    GET  /api/config     → 返回当前已配置的 Key 状态 + 本机可填写的字段（不回传明文）
    POST /api/config     → 把前端填写的 Key 写入 .env 并热更新到运行环境
    POST /api/ingest     → 重新构建本地向量库（调用 ingest.reset），返回片段数
    GET  /api/kb         → 知识库统计（文件数 / 片段数 / 清单）
    GET  /api/kb/tree    → docs/ 目录树（供前端文件浏览器）
    GET  /api/kb/file    → 读取 docs/ 下某个文本文件
    POST /api/kb/file    → 新建 / 覆盖写入 docs/ 下文本文件
    POST /api/kb/mkdir   → 在 docs/ 下新建文件夹
    POST /api/kb/delete  → 删除 docs/ 下文件或文件夹
    GET  /api/principles → 读取个人原则（memory/principles.md，不进向量库）
    POST /api/principles → 保存个人原则（下次提问立即生效）

设计原则：
- run_agent 的签名与返回结构零改动，本文件只做 HTTP 包装（参数白名单 + 异常兜底）。
- 绑定 127.0.0.1，默认不出本机；不使用 CORS 通配符以外的任何外发请求。
- 所有文件管理接口严格限制在 docs/ 内（_safe_docs_path 防目录穿越），且只放行文本后缀。
"""
from __future__ import annotations

import os
import sys
import traceback
import urllib.parse
from typing import Any, Dict, List, Optional

# 允许 `python src/api.py` 直接运行（保证与同目录模块的扁平导入兼容）。
# 打包成 exe 后模块已在 sys.path 顶层，无需（也不应）再插入解压目录。
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
)
from pydantic import BaseModel, Field

from agent import run_agent
from config import (
    ANALYSIS_DIR,
    BUNDLE_DIR,
    CHAT_HISTORY_DIR,
    CHROMA_DIR,
    DOCS_DIR,
    EXPORT_DIR,
    JEV_THRESHOLD,
    REPORTS_DIR,
    ROOT,
    TOP_K,
)

HOST = os.getenv("API_HOST", "127.0.0.1")
PORT = int(os.getenv("API_PORT", "8000"))

# 前端页面属于「只读资源」，随程序打包（打包后从解压目录读）
FRONTEND_DIR = str(BUNDLE_DIR / "frontend")

app = FastAPI(title="问己 AskSelf · 本地 API", docs_url="/api/docs", openapi_url="/api/openapi.json")

# 仅本机使用：允许 file://（Origin 为 null）与本地页面调试
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """与 run_agent 的参数一一对应（字段名与前端 JSON 保持一致）。"""

    question: str = Field(..., min_length=1, description="用户问题")
    history: Optional[List[Dict[str, Any]]] = None
    top_k: int = Field(TOP_K, ge=1, le=100)
    jev_threshold: float = Field(JEV_THRESHOLD, ge=0, le=10)
    use_jev: bool = True
    use_web_search: bool = True
    sources: Optional[List[str]] = Field(
        None, description="@ 引用的检索范围（文件/文件夹路径列表）；空=全库检索"
    )
    conversation_id: Optional[str] = Field(
        None, description="前端本地会话 id，用于聊天记录持久化后按会话导出/清空"
    )


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """前端启动时探活，用来提示「请先运行 python src/api.py」。"""
    from config import DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, TAVILY_API_KEY

    return {
        "ok": True,
        "service": "askself",
        "keys": {
            "dashscope": bool(DASHSCOPE_API_KEY),
            "deepseek": bool(DEEPSEEK_API_KEY),
            "tavily": bool(TAVILY_API_KEY),
        },
    }


@app.get("/api/v2/heartbeat")
def heartbeat() -> Dict[str, bool]:
    """兼容占位心跳接口（非本项目前端使用）。

    现象：控制台持续出现 `GET /api/v2/heartbeat 404 Not Found`。
    排查：项目代码与 git 历史中均无 heartbeat / api/v2 —— 是外部客户端
    （遗留页面 / 第三方工具）把本服务当成带该惯例端点的服务在轮询。
    处理：按用户要求加此空接口返回 200，消除日志噪音（不改动真实业务）。
    """
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest) -> Any:
    """同步包装 run_agent。

    FastAPI 对 `def` 端点会自动放入线程池执行，不会阻塞事件循环；
    run_agent 本身是同步的 LangGraph 调用，这里不加 async。
    """
    question = (req.question or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "question 不能为空"})

    try:
        result = run_agent(
            question=question,
            history=req.history or [],
            top_k=req.top_k,
            jev_threshold=req.jev_threshold,
            use_jev=req.use_jev,
            use_web_search=req.use_web_search,
            sources=req.sources or None,
        )
        # 成功后把「这一轮」追加到本地 JSONL（不重复存 history）；失败不落地。
        try:
            import chat_store

            chat_store.append_turn(
                question=question,
                answer=result.get("answer", ""),
                citations=result.get("citations", []),
                sources=req.sources or [],
                need_retrieval=bool(result.get("need_retrieval")),
                evidence_count=int(result.get("evidence_count", 0) or 0),
                web_search_triggered=bool(result.get("web_search_triggered")),
                local_count=int(result.get("local_count", 0) or 0),
                web_count=int(result.get("web_count", 0) or 0),
                conversation_id=req.conversation_id or None,
            )
        except Exception as store_exc:  # noqa: BLE001 — 持久化失败不影响问答本身
            print(f"[warn] 聊天记录落盘失败：{store_exc}", file=sys.stderr)
        return JSONResponse(content=result)
    except Exception as exc:  # noqa: BLE001 — 兜底，把错误交给前端卡片展示
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(exc).__name__}: {exc}"},
        )


# ------------------------------------------------------------------
# 配置读写：/api/config（真实读写 .env + 热更新运行环境）
# ------------------------------------------------------------------

# 允许前端读写的 Key 白名单（防止任意 .env 注入）
CONFIG_KEYS = ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY", "TAVILY_API_KEY")

# 掩码里的星号串：既用于展示，也用于识别「用户把掩码原文填回来了」的误操作
_MASK_STARS = "*" * 6


def _mask_secret(value: str) -> str:
    """把已配置的 Key 变成可安全展示的掩码文本（保留前缀与末 4 位）。

    只用于前端占位提示，**绝不回传明文**；Key 太短时整体打码。
    例：``sk-abcdef...9f3a`` → ``sk-******9f3a``，``tvly-xxx`` → ``tvly-******xxxx``。
    """
    s = (value or "").strip()
    if not s:
        return ""
    if len(s) <= 10:                 # 太短，露任何一位都可能泄露 → 整体打码
        return "*" * len(s)
    dash = s.find("-")
    head = s[: dash + 1] if 0 < dash <= 5 else s[:3]   # 保留 "sk-" / "tvly-" 这类公开前缀
    return f"{head}{_MASK_STARS}{s[-4:]}"


def _push_key_to_backend(name: str, value: str) -> None:
    """把单个 Key 同步到 os.environ + config 属性 + 已加载子模块的全局变量。

    config.py 在 import 时把 Key 读成模块级常量，而 embedding / web_search / agent
    又用 `from config import XXX_API_KEY` 在 import 时拷贝了值，所以三处都要回写，
    否则新 Key 对已跑起来的代码不生效。
    """
    import config as _cfg

    os.environ[name] = value
    setattr(_cfg, name, value)
    for mod_name in ("agent", "embedding", "web_search"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, name):
            setattr(mod, name, value)


def _write_env(updates: Dict[str, str]) -> None:
    """把 Key 写回数据目录下的 .env（保留原有其它行与注释，只更新/追加这些键）。

    路径取自 config.ROOT：开发时是项目根，打包后是 exe 同级目录，
    保证打包版也能正常保存 Key。
    """
    env_path = str(ROOT / ".env")
    lines: List[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    # 未出现的键追加到文件末尾
    for key, val in remaining.items():
        out.append(f"{key}={val}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


class ConfigRequest(BaseModel):
    """前端提交的 Key（只允许白名单字段）。

    约定：字段缺省（None）或空字符串（用户留空）都表示「不修改」，
    避免用户只改一个 Key 时误清空其余 Key。
    若要清空某个 Key，请显式传特殊值 ``"__CLEAR__"``。
    """

    DASHSCOPE_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    """返回三个 Key 的配置状态与**掩码文本**（不回传明文），供前端显示。

    - ``configured``：布尔，是否已配置（前端状态点用）。
    - ``masked``：形如 ``sk-******9f3a`` 的掩码串（未配置则空串），
      前端拿它当输入框 placeholder，让用户一眼看出「这把钥匙已经配过了」，
      而不会误以为从未配置。掩码不以任何形式参与提交。
    """
    import config as _cfg

    raw = {k: (getattr(_cfg, k, "") or "") for k in CONFIG_KEYS}
    return {
        "ok": True,
        "configured": {k: bool(v) for k, v in raw.items()},
        "masked": {k: _mask_secret(v) for k, v in raw.items()},
    }


@app.post("/api/config")
def set_config(req: ConfigRequest) -> Any:
    """把前端填写的 Key 写入 .env 并热更新运行环境（无需重启）。

    安全约定：``None`` 或空字符串均视为「不修改」，防止留空误删已有 Key；
    需要清空时显式传 ``"__CLEAR__"``。
    """
    updates: Dict[str, str] = {}
    for key in CONFIG_KEYS:
        val = getattr(req, key)
        if val is None:                # 前端未提交该字段 → 不修改
            continue
        val = val.strip()
        if val == "":                  # 留空 → 不修改（与输入框提示一致）
            continue
        if val == "__CLEAR__":
            updates[key] = ""          # 显式清空
        elif _MASK_STARS in val:       # 用户把前端掩码原文填回来了 → 视为不修改，防止覆盖真 Key
            continue
        else:
            updates[key] = val

    if not updates:
        return JSONResponse(status_code=400, content={"error": "没有需要更新的字段"})

    try:
        _write_env(updates)
        for key, val in updates.items():
            _push_key_to_backend(key, val)
        import config as _cfg
        raw = {k: (getattr(_cfg, k, "") or "") for k in CONFIG_KEYS}
        return JSONResponse(status_code=200, content={
            "ok": True,
            "updated": list(updates.keys()),
            "configured": {k: bool(v) for k, v in raw.items()},
            "masked": {k: _mask_secret(v) for k, v in raw.items()},
        })
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


class IngestRequest(BaseModel):
    mode: str = Field("incremental", description="incremental=增量（复用旧向量，省 token）/ rebuild=整体重建")


@app.post("/api/ingest")
def ingest_now(req: Optional[IngestRequest] = None) -> Any:
    """重新构建本地向量库。

    - mode=incremental（默认）：已录入过的片段复用旧向量，只嵌入新增/变化的，省 token；
    - mode=rebuild：清空集合整体重建（换嵌入模型后必须用这个）。
    """
    mode = (req.mode if req else "incremental") or "incremental"
    try:
        if mode == "rebuild":
            from ingest import ingest as run_ingest
            n = run_ingest(reset=True)
            return {"ok": True, "mode": "rebuild", "chunks": int(n),
                    "reused": 0, "embedded": int(n)}
        from ingest import ingest_incremental
        stats = ingest_incremental()
        return {"ok": True, "mode": "incremental", "chunks": stats["total"],
                "reused": stats["reused"], "embedded": stats["embedded"]}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/kb")
def kb_stats() -> Dict[str, Any]:
    """知识库概况：docs/ 下 md 文件数 + ChromaDB 已入库片段数 + 文件清单。"""
    import config as _cfg

    files: List[str] = []
    try:
        if _cfg.DOCS_DIR.exists():
            for fp in sorted(_cfg.DOCS_DIR.rglob("*.md")):
                try:
                    files.append(fp.relative_to(_cfg.ROOT).as_posix())
                except Exception:
                    files.append(fp.name)
    except Exception:
        files = []

    chunks = 0
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(_cfg.CHROMA_DIR))
        chunks = client.get_collection(_cfg.COLLECTION_NAME).count()
    except Exception:
        chunks = 0

    return {"ok": True, "files": len(files), "chunks": int(chunks), "list": files}


# ------------------------------------------------------------------
# 知识库文件管理（严格限制在 docs/ 内，防目录穿越）
# ------------------------------------------------------------------

# 允许读写的文件后缀（只放行 Markdown，避免误建可执行/其他文件）
KB_ALLOWED_EXT = (".md", ".markdown", ".txt")


def _docs_root():
    """返回 docs/ 的绝对真实路径（不存在则创建）。"""
    import config as _cfg

    _cfg.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return _cfg.DOCS_DIR.resolve()


def _safe_docs_path(rel: str, *, allow_root: bool = False):
    """把前端传来的相对路径解析为 docs/ 内的绝对路径。

    - 拒绝绝对路径、`..`、以及任何逃逸出 docs/ 的路径（防目录穿越）。
    - ``allow_root=True`` 时允许 ``rel`` 为空（指向 docs/ 本身）。
    解析失败抛 ValueError。
    """
    root = _docs_root()
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    if rel == "":
        if allow_root:
            return root
        raise ValueError("路径不能为空")
    if rel.startswith("/") or ".." in rel.split("/"):
        raise ValueError("非法路径")
    target = (root / rel).resolve()
    # 必须仍在 docs/ 之内
    if target != root and root not in target.parents:
        raise ValueError("路径超出 docs/ 范围")
    return target


def _rel_to_docs(path) -> str:
    """绝对路径 → 相对 docs/ 的 posix 路径。"""
    return path.relative_to(_docs_root()).as_posix()


def _build_tree(root) -> List[Dict[str, Any]]:
    """递归构建 docs/ 目录树（文件夹在前、文件在后，各自按名排序）。"""
    nodes: List[Dict[str, Any]] = []
    try:
        entries = sorted(
            root.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except Exception:
        return nodes
    for p in entries:
        if p.name.startswith("."):
            continue
        rel = _rel_to_docs(p)
        if p.is_dir():
            nodes.append({
                "type": "dir",
                "name": p.name,
                "path": rel,
                "children": _build_tree(p),
            })
        elif p.suffix.lower() in KB_ALLOWED_EXT:
            nodes.append({
                "type": "file",
                "name": p.name,
                "path": rel,
                "size": p.stat().st_size,
            })
    return nodes


@app.get("/api/kb/tree")
def kb_tree() -> Any:
    """返回 docs/ 的目录树，供前端文件浏览器渲染。"""
    try:
        root = _docs_root()
        return {"ok": True, "root": "docs", "tree": _build_tree(root)}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/kb/file")
def kb_read_file(path: str) -> Any:
    """读取 docs/ 下某个文本文件的内容。"""
    try:
        fp = _safe_docs_path(path)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if not fp.exists() or not fp.is_file():
        return JSONResponse(status_code=404, content={"error": "文件不存在"})
    if fp.suffix.lower() not in KB_ALLOWED_EXT:
        return JSONResponse(status_code=400, content={"error": "只支持读取 Markdown / 文本文件"})
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})
    return {"ok": True, "path": _rel_to_docs(fp), "content": content}


class FileWriteRequest(BaseModel):
    path: str = Field(..., description="相对 docs/ 的文件路径，如 diary/2026-10-02.md")
    content: str = Field("", description="文件正文")


@app.post("/api/kb/file")
def kb_write_file(req: FileWriteRequest) -> Any:
    """新建或覆盖写入 docs/ 下的文本文件（自动创建父目录）。"""
    if not req.path.strip().lower().endswith(KB_ALLOWED_EXT):
        return JSONResponse(status_code=400, content={"error": "只支持 .md / .markdown / .txt 文件"})
    try:
        fp = _safe_docs_path(req.path)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if fp.is_dir():
        return JSONResponse(status_code=400, content={"error": "同名文件夹已存在"})
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(req.content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})
    return {"ok": True, "path": _rel_to_docs(fp), "size": fp.stat().st_size}


class DeleteRequest(BaseModel):
    path: str = Field(..., description="相对 docs/ 的文件或文件夹路径")


@app.post("/api/kb/delete")
def kb_delete(req: DeleteRequest) -> Any:
    """删除 docs/ 下的文件或（空）文件夹。"""
    try:
        target = _safe_docs_path(req.path)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if target == _docs_root():
        return JSONResponse(status_code=400, content={"error": "不能删除 docs 根目录"})
    if not target.exists():
        return JSONResponse(status_code=404, content={"error": "目标不存在"})
    try:
        import shutil

        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})
    return {"ok": True, "deleted": req.path}


class MkdirRequest(BaseModel):
    path: str = Field(..., description="相对 docs/ 的新文件夹路径")


@app.post("/api/kb/mkdir")
def kb_mkdir(req: MkdirRequest) -> Any:
    """在 docs/ 下新建文件夹（支持多级）。"""
    try:
        target = _safe_docs_path(req.path)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if target == _docs_root():
        return JSONResponse(status_code=400, content={"error": "文件夹名不能为空"})
    if target.exists():
        return JSONResponse(status_code=400, content={"error": "同名文件或文件夹已存在"})
    try:
        target.mkdir(parents=True, exist_ok=False)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})
    return {"ok": True, "path": _rel_to_docs(target)}


# ------------------------------------------------------------------
# 个人原则：/api/principles（独立文件，不进向量库）
# ------------------------------------------------------------------


class PrinciplesRequest(BaseModel):
    text: str = Field("", description="个人原则正文（Markdown）")


@app.get("/api/principles")
def get_principles() -> Any:
    """读取个人原则（不存在则自动落一份模板）。"""
    try:
        from principles import ensure_exists, load_principles

        p = ensure_exists()
        return {
            "ok": True,
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "text": load_principles(),
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/principles")
def set_principles(req: PrinciplesRequest) -> Any:
    """保存个人原则。写回即生效：下次 run_agent 生成时立即带上（无需重启/重建索引）。"""
    try:
        from principles import save_principles

        p = save_principles(req.text or "")
        return {
            "ok": True,
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "length": len(req.text or ""),
        }
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


# ------------------------------------------------------------------
# 日记分析：/api/analysis/*（单篇日记卡 + 周期总结，缓存走 analysis_cache/）
# ------------------------------------------------------------------

class AnalysisGenerateRequest(BaseModel):
    include_private: bool = False
    limit: int = Field(8, ge=1, le=50, description="本次最多生成多少篇（前端循环调用显示进度）")
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    force: bool = False


class AnalysisSummaryRequest(BaseModel):
    granularity: str = Field("month", description="day/week/month/quarter/year/custom")
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    include_private: bool = False
    undated_only: bool = False
    save: bool = True


class AnalysisCancelRequest(BaseModel):
    task: str = Field("all", description="要终止的任务：cards / summary / all")


@app.get("/api/analysis/status")
def analysis_status(include_private: bool = False) -> Any:
    """日记扫描 + 缓存状态：总条目 / 已缓存 / 待生成 / 排除私密。"""
    try:
        import diary_analysis

        scan = diary_analysis.scan_diaries(include_private=include_private)
        return {"ok": True,
                "total": scan["total"], "cached": scan["cached"],
                "pending": scan["pending"], "undated": scan["undated"],
                "private_excluded": scan["private_excluded"],
                "entries": scan["entries"]}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/analysis/generate")
def analysis_generate(req: AnalysisGenerateRequest) -> Any:
    """为缺失的日记条目生成卡片（每次最多 limit 篇，前端循环直到 done）。"""
    try:
        import diary_analysis

        return diary_analysis.generate_cards(
            include_private=req.include_private,
            limit=req.limit,
            date_from=req.date_from,
            date_to=req.date_to,
            force=req.force,
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/analysis/cancel")
def analysis_cancel(req: AnalysisCancelRequest) -> Any:
    """终止正在进行的卡片生成或周期总结（在批次/分块之间生效）。"""
    task = req.task if req.task in ("cards", "summary", "all") else "all"
    try:
        import diary_analysis

        diary_analysis.request_cancel(task)
        return {"ok": True, "task": task}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.get("/api/analysis/cards")
def analysis_cards(date_from: Optional[str] = None, date_to: Optional[str] = None,
                   include_private: bool = False, undated_only: bool = False) -> Any:
    """区间内已生成的日记卡 + 情绪曲线 + 主题/情绪聚合；undated_only 只看「其它」。"""
    try:
        import diary_analysis

        return diary_analysis.query_cards(
            date_from=date_from, date_to=date_to,
            include_private=include_private, undated_only=undated_only)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/analysis/summary")
def analysis_summary(req: AnalysisSummaryRequest) -> Any:
    """周期总结（两级生成，仅在用户点「生成」时调用模型）。"""
    try:
        import diary_analysis

        g = req.granularity if req.granularity in (
            "day", "week", "month", "quarter", "year", "custom") else "month"
        result = diary_analysis.build_period_summary(
            granularity=g, date_from=req.date_from, date_to=req.date_to,
            include_private=req.include_private, undated_only=req.undated_only,
            save=req.save)
        if not result.get("ok"):
            # 被用户终止：返回 499 语义，但保留 200 之外的可识别 cancelled 标记
            return JSONResponse(
                status_code=499 if result.get("cancelled") else 409, content=result)
        return result
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


# ------------------------------------------------------------------
# 导出：/api/export/diary*（日记/材料）与 /api/export/chat（聊天记录）
# ------------------------------------------------------------------

class DiaryExportRequest(BaseModel):
    fmt: str = Field("zip", description="zip / md / merged / json / csv")
    type_filter: str = Field("all", description="all / diary / course / work / study / docs")
    folders: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    include_private: bool = False
    date_from: Optional[str] = None
    date_to: Optional[str] = None


def _export_kwargs(req: "DiaryExportRequest") -> Dict[str, Any]:
    return {
        "type_filter": req.type_filter,
        "folders": req.folders or None,
        "tags": req.tags or None,
        "include_private": req.include_private,
        "date_from": req.date_from or None,
        "date_to": req.date_to or None,
    }


@app.post("/api/export/diary/preview")
def export_diary_preview(req: DiaryExportRequest) -> Any:
    """导出前预览：将导出多少篇 / 多少 KB / 排除私密多少篇。"""
    try:
        import export as export_mod

        return export_mod.preview_diary(**_export_kwargs(req))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


@app.post("/api/export/diary")
def export_diary(req: DiaryExportRequest, background: BackgroundTasks) -> Any:
    """导出日记/材料。ZIP 为 exports/ 下临时文件，下发后自动删除；其余直接返回字节。"""
    try:
        import export as export_mod

        result = export_mod.export_diary(fmt=req.fmt, **_export_kwargs(req))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})

    filename = result["filename"]
    media_type = result["media_type"]
    if "path" in result:
        path = result["path"]
        background.add_task(lambda p: p.unlink(missing_ok=True) if p.exists() else None, path)
        # 不传 filename（FileResponse 会用 latin-1 生成默认头，中文文件名会炸），
        # 只保留 RFC 5987 的 filename*=UTF-8'' 编码头。
        return FileResponse(
            str(path), media_type=media_type,
            headers={"Content-Disposition":
                     f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}"},
        )
    headers = {
        "Content-Disposition":
            f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}"
    }
    return Response(content=result["content"], media_type=media_type, headers=headers)


class ChatExportRequest(BaseModel):
    fmt: str = Field("markdown", description="markdown / json / csv")
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    conversation_id: Optional[str] = None


@app.post("/api/export/chat")
def export_chat(req: ChatExportRequest) -> Any:
    """导出聊天记录（只有 Markdown 对话稿）。"""
    import urllib.parse
    import chat_store
    import export as export_mod

    try:
        data = chat_store.read_turns(
            date_from=req.date_from or None,
            date_to=req.date_to or None,
            conversation_id=req.conversation_id or None,
        )
        turns = export_mod.filter_private_citations(data["turns"])
        if not turns:
            return JSONResponse(status_code=404, content={"error": "所选范围内没有聊天记录"})
        stamp = chat_store.dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        # 聊天记录只导出 Markdown 对话稿（2026-10-23：JSON / CSV 两种格式已按要求移除）。
        # req.fmt 仍保留在请求模型里以避免破坏接口契约，但无论传什么值都走 Markdown。
        content = chat_store.render_markdown(turns).encode("utf-8")
        filename, media = f"聊天记录_{stamp}.md", "text/markdown; charset=utf-8"
        headers = {"Content-Disposition":
                   f"attachment; filename*=UTF-8''{urllib.parse.quote(filename)}"}
        return Response(content=content, media_type=media, headers=headers)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


# ------------------------------------------------------------------
# 聊天历史：/api/chat/history（读取 / 物理清空）
# ------------------------------------------------------------------

@app.get("/api/chat/history")
def chat_history(date_from: Optional[str] = None, date_to: Optional[str] = None,
                 conversation_id: Optional[str] = None,
                 limit: int = 100, offset: int = 0) -> Any:
    """分页读取聊天记录；不传 limit=0 表示不分页。"""
    try:
        import chat_store

        lim = None if limit == 0 else max(1, min(limit, 1000))
        data = chat_store.read_turns(
            date_from=date_from or None, date_to=date_to or None,
            conversation_id=conversation_id or None,
            limit=lim, offset=max(0, offset))
        data["months"] = chat_store.list_months()
        return {"ok": True, **data}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


class ChatHistoryClearRequest(BaseModel):
    scope: str = Field("all", description="all / conversation / range")
    conversation_id: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    confirmed: bool = False


@app.post("/api/chat/history/clear")
def chat_history_clear(req: ChatHistoryClearRequest) -> Any:
    """物理删除聊天记录（整文件 / 按会话重写 / 按区间重写）。必须 confirmed=true。"""
    if not req.confirmed:
        return JSONResponse(status_code=400, content={"error": "请二次确认删除（confirmed=true）"})
    try:
        import chat_store

        result = chat_store.clear_turns(
            scope=req.scope, conversation_id=req.conversation_id or None,
            date_from=req.date_from or None, date_to=req.date_to or None)
        result["months"] = chat_store.list_months()
        return result
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"{type(exc).__name__}: {exc}"})


# ------------------------------------------------------------------
# 静态托管原型页（同一端口，省得再起一个服务器）
# ------------------------------------------------------------------


@app.get("/")
def index() -> Any:
    """托管前端主页面 frontend/index.html。

    注意：此处**手动读取文件并返回 HTMLResponse**，而不是 FileResponse ——
    Starlette 1.x 的 FileResponse 在 httptools 下会因 Content-Length 与实际
    字节数不一致抛 `RuntimeError: Response content longer than Content-Length`
    （新版对长度强校验）。自行读取可保证长度精确一致。
    同时加 no-store，避免浏览器缓存旧页面。
    """
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            html = fh.read()
        return HTMLResponse(
            content=html,
            headers={"Cache-Control": "no-store, must-revalidate"},
        )
    return JSONResponse(status_code=404, content={"error": "未找到 frontend/index.html"})


@app.get("/favicon.ico")
def favicon() -> Any:
    return Response(status_code=204)


def _port_available(host: str, port: int) -> bool:
    """探测端口是否可用（用于打包版：8000 被占时自动顺延，而不是启动失败）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, start: int, tries: int = 20) -> int:
    """从 start 起找一个可用端口。"""
    for p in range(start, start + tries):
        if _port_available(host, p):
            return p
    return start          # 都不可用就交给 uvicorn 报错，不静默改端口


def _safe_console() -> None:
    """Windows 中文控制台默认编码是 GBK，遇到 emoji 等非 GBK 字符会抛
    UnicodeEncodeError 把程序整个搞崩（打包版尤其致命：双击直接闪退）。

    这里把 stdout/stderr 的错误处理改成 replace —— 编码不了的字符变成「?」，
    但程序继续跑。注意刻意**不改 encoding**：控制台是 GBK 时硬改成 UTF-8
    反而会让中文变乱码。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def _bootstrap_data() -> None:
    """首次运行时准备好数据目录与 .env 雏形（打包版尤其需要）。

    数据目录都在 config.ROOT 下：开发时 = 项目根，打包后 = exe 同级目录。
    """
    for d in (DOCS_DIR, CHROMA_DIR, ROOT / "memory",
              ANALYSIS_DIR, REPORTS_DIR, EXPORT_DIR, CHAT_HISTORY_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    env_path = ROOT / ".env"
    if not env_path.exists():
        sample = BUNDLE_DIR / ".env.example"
        try:
            env_path.write_text(
                sample.read_text(encoding="utf-8") if sample.exists() else "",
                encoding="utf-8",
            )
            # 不带 emoji：GBK 控制台会遇到编码问题（见 _safe_console）
            print(f"[api] 已生成 {env_path}（首次运行），请到网页「API 配置」页填入 Key")
        except OSError:
            pass


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="问己 AskSelf 本地 API 服务")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--browser", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    _safe_console()
    _bootstrap_data()

    if args.reload:
        print(f"[api] 问己 AskSelf 服务启动：http://{args.host}:{args.port}/")
        uvicorn.run("api:app", host=args.host, port=args.port, reload=True)
        return

    # 端口被占时自动顺延（打包版双击即用，不能因为 8000 被占就起不来）
    port = _pick_port(args.host, args.port)
    if port != args.port:
        print(f"[api] 端口 {args.port} 被占用，已自动改用 {port}")

    print(f"[api] 问己 AskSelf 服务启动：http://{args.host}:{port}/")
    print(f"[api] 数据目录：{ROOT}")
    print(f"[api] 前端页面：http://{args.host}:{port}/   （frontend/index.html）")
    print(f"[api] 接口：POST /api/chat  GET /api/health  GET|POST /api/config  POST /api/ingest")

    # 打包版双击运行时自动拉起浏览器；命令行手动运行时保持安静（除非加了 --browser）
    open_browser = (not args.no_browser) and (getattr(sys, "frozen", False) or args.browser)
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.2, lambda: webbrowser.open(f"http://{args.host}:{port}/")).start()

    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
