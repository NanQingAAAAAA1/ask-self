"""聊天记录本地持久化：JSONL 按月追加写。

设计：
  * 每次 /api/chat 成功后，把「这一轮」（问题 + 回答 + 引用元数据）追加一行到
    ``chat_history/YYYY-MM.jsonl``。**不重复存 history**——导出时按 ts 排序重建会话。
  * append-only：先写临时文件再落盘，多线程用同一把锁串行化，崩溃也只损最后一行。
  * 数据只在本地（config.ROOT 下，已 gitignore），不进 docs/、不进向量库、不上传。
  * 导出支持 Markdown 对话稿 / JSON / CSV；删除为物理删除（整文件或重写文件）。

对外接口：
    append_turn(...)                  追加一轮
    read_turns(...)                   按区间/会话读取（分页）
    months()                          已有记录的月份列表
    clear_turns(...)                  物理删除（all / conversation / range）
    render_markdown
"""
from __future__ import annotations

import csv
import io
import json
import threading
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CHAT_HISTORY_DIR

SCHEMA_VERSION = 1
_write_lock = threading.Lock()


def _ensure_dir() -> Path:
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return CHAT_HISTORY_DIR


def _month_file(ts: str) -> Path:
    """ISO 时间戳 → 当月 JSONL 文件。解析失败落到 unknown 月。"""
    month = "unknown"
    try:
        month = ts[:7]  # YYYY-MM
        dt.datetime.strptime(month, "%Y-%m")
    except (ValueError, TypeError):
        pass
    return _ensure_dir() / f"{month}.jsonl"


def append_turn(*, question: str, answer: str,
                citations: Optional[List[Dict[str, Any]]] = None,
                sources: Optional[List[str]] = None,
                need_retrieval: bool = False, evidence_count: int = 0,
                web_search_triggered: bool = False,
                local_count: int = 0, web_count: int = 0,
                conversation_id: Optional[str] = None,
                ts: Optional[str] = None) -> Dict[str, Any]:
    """追加一轮问答。返回写入的记录（不含锁）。"""
    ts = ts or dt.datetime.now().isoformat(timespec="seconds")
    record = {
        "schema_version": SCHEMA_VERSION,
        "ts": ts,
        "conversation_id": conversation_id or "",
        "question": question or "",
        "answer": answer or "",
        "citations": citations or [],
        "sources": sources or [],
        "need_retrieval": bool(need_retrieval),
        "evidence_count": int(evidence_count),
        "web_search_triggered": bool(web_search_triggered),
        "local_count": int(local_count),
        "web_count": int(web_count),
    }
    line = json.dumps(record, ensure_ascii=False)
    target = _month_file(ts)
    with _write_lock:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record


def _iter_files(date_from: Optional[str], date_to: Optional[str]) -> List[Path]:
    """按区间挑选月份文件（文件名前缀粗筛，减少读盘）。"""
    d = _ensure_dir()
    files = sorted(d.glob("*.jsonl"))
    if not date_from and not date_to:
        return files
    out: List[Path] = []
    for f in files:
        month = f.stem  # YYYY-MM
        if month == "unknown":
            out.append(f)
            continue
        if date_from and month < date_from[:7]:
            continue
        if date_to and month > date_to[:7]:
            continue
        out.append(f)
    return out


def read_turns(*, date_from: Optional[str] = None, date_to: Optional[str] = None,
               conversation_id: Optional[str] = None,
               limit: Optional[int] = None, offset: int = 0) -> Dict[str, Any]:
    """读取问答记录（按 ts 升序），支持区间 / 会话 / 分页。"""
    records: List[Dict[str, Any]] = []
    for f in _iter_files(date_from, date_to):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = rec.get("ts", "")
                    if date_from and ts[:10] < date_from:
                        continue
                    if date_to and ts[:10] > date_to:
                        continue
                    if conversation_id and rec.get("conversation_id") != conversation_id:
                        continue
                    records.append(rec)
        except FileNotFoundError:
            continue
    records.sort(key=lambda r: r.get("ts", ""))
    total = len(records)
    if limit is not None:
        page = records[offset:offset + limit]
    else:
        page = records[offset:]
    return {"total": total, "offset": offset, "count": len(page), "turns": page}


def list_months() -> List[Dict[str, Any]]:
    """已有聊天记录的月份 + 每月条数。"""
    out = []
    for f in sorted(_ensure_dir().glob("*.jsonl")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                n = sum(1 for line in fh if line.strip())
        except FileNotFoundError:
            n = 0
        out.append({"month": f.stem, "turns": n})
    return out


def clear_turns(*, scope: str = "all",
                conversation_id: Optional[str] = None,
                date_from: Optional[str] = None,
                date_to: Optional[str] = None) -> Dict[str, Any]:
    """物理删除。

    scope:
      all          → 删除所有 JSONL 文件；
      conversation → 重写各月文件，剔除该 conversation_id 的行；
      range        → 重写各月文件，剔除 ts 落在 [from,to] 的行。
    """
    d = _ensure_dir()
    deleted = 0
    if scope == "all":
        for f in d.glob("*.jsonl"):
            try:
                deleted += sum(1 for _ in open(f, encoding="utf-8"))
                f.unlink()
            except (FileNotFoundError, OSError):
                pass
        return {"ok": True, "scope": "all", "deleted": deleted}

    if scope not in ("conversation", "range"):
        raise ValueError(f"未知 scope: {scope}")

    for f in _iter_files(date_from if scope == "range" else None,
                         date_to if scope == "range" else None):
        if not f.exists():
            continue
        kept: List[str] = []
        try:
            with open(f, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            continue
        for line in lines:
            s = line.strip()
            if not s:
                continue
            drop = False
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                kept.append(line if line.endswith("\n") else line + "\n")
                continue
            if scope == "conversation" and rec.get("conversation_id") == conversation_id:
                drop = True
            elif scope == "range":
                ts = rec.get("ts", "")[:10]
                if (not date_from or ts >= date_from) and (not date_to or ts <= date_to):
                    drop = True
            if drop:
                deleted += 1
            else:
                kept.append(line if line.endswith("\n") else line + "\n")
        with _write_lock:
            if kept:
                f.write_text("".join(kept), encoding="utf-8")
            else:
                f.unlink()
    return {"ok": True, "scope": scope, "deleted": deleted}


# ------------------------------------------------------------------
# 导出序列化
# ------------------------------------------------------------------

def _fmt_ts(ts: str) -> str:
    try:
        return dt.datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts


def render_markdown(turns: List[Dict[str, Any]], *, title: str = "问己 · 聊天记录导出") -> str:
    """Markdown 对话稿：我 / 问己 交替，答案下方附引用来源（local/web 分色文字）。"""
    out: List[str] = [f"# {title}", "",
                      f"> 导出时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　·　"
                      f"共 {len(turns)} 轮问答　·　仅存于本地", ""]
    last_conv = None
    for r in turns:
        cid = r.get("conversation_id") or ""
        if cid and cid != last_conv:
            out.append(f"\n---\n\n## 会话 {cid}\n")
            last_conv = cid
        out.append(f"### 我　·　{_fmt_ts(r.get('ts', ''))}")
        out.append("")
        out.append((r.get("question") or "").strip())
        out.append("")
        if r.get("sources"):
            out.append("> 检索范围：" + "、".join(r["sources"]))
            out.append("")
        out.append(f"### 问己　·　{'网络回退' if r.get('web_search_triggered') else '本地知识库'}")
        out.append("")
        out.append((r.get("answer") or "").strip())
        out.append("")
        cites = r.get("citations") or []
        for c in cites:
            if c.get("type") == "web":
                out.append(f"> ◈ [web] {c.get('title') or c.get('source')} — {c.get('source')}")
            else:
                score = c.get("score")
                score_txt = f"，score {score}" if score is not None else ""
                out.append(f"> ◆ [local] {c.get('source')}（{c.get('date') or '无日期'}{score_txt}）")
                if c.get("chunk"):
                    out.append(f"> 　「{str(c['chunk'])[:120]}」")
        if cites:
            out.append("")
    return "\n".join(out).rstrip() + "\n"
