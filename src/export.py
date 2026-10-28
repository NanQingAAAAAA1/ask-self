"""日记 / 知识库材料导出：按条件挑文件 + 序列化为 Markdown / ZIP / JSON / CSV。

边界（重要）：
  * 导出产物落 ``exports/``（config.ROOT 下，已 gitignore），**绝不落 docs/**——
    否则下次 ingest.py --incremental 会把导出 Markdown 当新材料嵌进向量库。
  * ``private: true`` 默认排除；显式 include_private 时，导出物保留 private 标记。
  * 路径复用 api._safe_docs_path 的越界校验思路：只处理 docs/ 内、.md/.markdown/.txt。
  * 日记是按月合集存放的，因此时间区间过滤在「按天条目」粒度上进行：
    原样 MD / ZIP 保留整篇文件（只要区间内有任意一天命中），
    合并 Markdown / JSON / CSV 只导出区间内命中的那些天。

对外接口：
    preview_diary(...)   统计将导出多少篇 / 多少字 / 排除私密多少
    export_diary(...)    生成导出文件（ZIP 走临时文件，其余返回内存字节）
    private_source_set() 私密文件集合（给聊天记录导出兜底过滤引用）
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
import tempfile
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DOCS_DIR, EXPORT_DIR, ROOT
from ingest import infer_date, infer_source_type, parse_frontmatter
from diary_analysis import is_diary_file, split_entries

KB_ALLOWED_EXT = (".md", ".markdown", ".txt")


# ------------------------------------------------------------------
# 扫描与筛选
# ------------------------------------------------------------------

def _rel(fp: Path) -> str:
    return fp.relative_to(DOCS_DIR).as_posix()


def _meta_tags(meta: Dict[str, Any]) -> List[str]:
    raw = meta.get("tags") or []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def _file_date(meta: Dict[str, Any], fp: Path) -> Optional[str]:
    d = meta.get("date") or infer_date(fp.name)
    return str(d) if d else None


def _in_range(date: Optional[str], d_from: Optional[str], d_to: Optional[str]) -> bool:
    if not d_from and not d_to:
        return True
    if not date:
        return False
    if d_from and date < d_from:
        return False
    if d_to and date > d_to:
        return False
    return True


def _folder_match(rel: str, folders: Optional[List[str]]) -> bool:
    """folders 为相对 docs/ 的文件夹/文件路径列表；空 = 不限。"""
    if not folders:
        return True
    for f in folders:
        f = (f or "").strip().replace("\\", "/").strip("/")
        if not f:
            return True
        if rel == f or rel.startswith(f.rstrip("/") + "/"):
            return True
    return False


def collect_files(*, type_filter: str = "diary",
                  folders: Optional[List[str]] = None,
                  tags: Optional[List[str]] = None,
                  include_private: bool = False,
                  date_from: Optional[str] = None,
                  date_to: Optional[str] = None) -> Dict[str, Any]:
    """按条件收集文件。返回 {items, excluded_private, total_scanned}。

    item: {rel, disk_path, type, date, private, tags, raw, body,
           is_diary, entries(全部按天条目), entries_in_range}
    """
    items: List[Dict[str, Any]] = []
    excluded_private = 0
    total_scanned = 0
    tags = [t.strip() for t in (tags or []) if t.strip()]

    if not DOCS_DIR.exists():
        return {"items": [], "excluded_private": 0, "total_scanned": 0}

    for fp in sorted(DOCS_DIR.rglob("*")):
        if not fp.is_file() or fp.suffix.lower() not in KB_ALLOWED_EXT:
            continue
        total_scanned += 1
        rel = _rel(fp)
        if not _folder_match(rel, folders):
            continue
        raw = fp.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(raw)
        private = bool(meta.get("private", False))
        if private and not include_private:
            excluded_private += 1
            continue

        stype = str(meta.get("type") or infer_source_type(fp.name)).strip()
        diary = is_diary_file(fp, meta)
        if type_filter == "diary" and not diary:
            continue
        if type_filter not in ("diary", "all") and stype != type_filter:
            continue

        file_tags = _meta_tags(meta)
        if tags and not any(t in file_tags for t in tags):
            continue

        fdate = _file_date(meta, fp)
        entries = split_entries(body) if diary else []
        entries_in_range = [e for e in entries if _in_range(e["date"], date_from, date_to)]

        # 时间区间过滤
        if date_from or date_to:
            if diary:
                if not entries_in_range:
                    continue
            elif not _in_range(fdate, date_from, date_to):
                continue

        items.append({
            "rel": rel,
            "disk_path": fp,
            "type": "diary" if diary else stype,
            "date": fdate,
            "private": private,
            "tags": file_tags,
            "raw": raw,
            "body": body,
            "is_diary": diary,
            "entries": entries,
            "entries_in_range": entries_in_range,
        })

    return {"items": items, "excluded_private": excluded_private,
            "total_scanned": total_scanned}


def preview_diary(**kwargs) -> Dict[str, Any]:
    """导出前预览：篇数 / 字节数 / 排除私密篇数。"""
    res = collect_files(**kwargs)
    items = res["items"]
    bytes_total = 0
    date_from, date_to = kwargs.get("date_from"), kwargs.get("date_to")
    range_active = bool(date_from or date_to)
    for it in items:
        if range_active and it["is_diary"] and it["entries_in_range"]:
            bytes_total += sum(len(e["body"].encode("utf-8"))
                               for e in it["entries_in_range"])
        else:
            bytes_total += len(it["raw"].encode("utf-8"))
    return {
        "ok": True,
        "files": len(items),
        "bytes": bytes_total,
        "size_kb": round(bytes_total / 1024, 1),
        "excluded_private": res["excluded_private"],
        "range_active": range_active,
        "list": [it["rel"] for it in items],
    }


def private_source_set() -> set:
    """扫描 docs/ 下 frontmatter private:true 的文件（相对项目根的 posix 路径集合）。"""
    out = set()
    if not DOCS_DIR.exists():
        return out
    for fp in DOCS_DIR.rglob("*.md"):
        try:
            meta, _ = parse_frontmatter(fp.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if meta.get("private", False):
            out.add(fp.relative_to(ROOT).as_posix())
    return out


# ------------------------------------------------------------------
# 序列化
# ------------------------------------------------------------------

def _entry_text(entry: Dict[str, str]) -> str:
    head = entry.get("heading") or entry.get("date") or "无日期"
    return f"## {head}\n\n{entry['body'].strip()}\n"


def _range_active(kwargs) -> bool:
    return bool(kwargs.get("date_from") or kwargs.get("date_to"))


def _build_merged_markdown(items: List[Dict[str, Any]], range_filtered: bool) -> str:
    """单文件合并 Markdown：按日期排序，一天一个 ## 分节，标注来源文件。"""
    sections: List[Tuple[str, str]] = []
    for it in items:
        if it["is_diary"]:
            entries = it["entries_in_range"] if range_filtered else it["entries"]
            if not entries:
                # 无日期标题的日记文件 → 整体作为一节
                if not range_filtered and it["body"].strip():
                    sections.append((it["date"] or "9999-99-99",
                                     f"## {Path(it['rel']).stem}（{it['rel']}）\n\n{it['body'].strip()}\n"))
                continue
            for e in entries:
                head = e.get("heading") or e.get("date") or "无日期"
                sections.append((
                    e["date"] or "9999-99-99",
                    f"## {head}　·　{it['rel']}\n\n{e['body'].strip()}\n",
                ))
        else:
            sections.append((
                it["date"] or "9999-99-99",
                f"## {Path(it['rel']).stem}　·　{it['rel']}\n\n{it['body'].strip()}\n",
            ))
    sections.sort(key=lambda x: x[0])
    today = dt.date.today().isoformat()
    header = (
        "# 问己 · 个人记录导出（合并版）\n\n"
        f"> 导出时间：{today}　·　共 {len(sections)} 节　·　本文件在 exports/ 下，不在 docs/ 中，不会被录入索引\n\n---\n\n"
    )
    return header + "\n".join(text for _, text in sections)


def _build_json(items: List[Dict[str, Any]], range_filtered: bool) -> str:
    docs = []
    for it in items:
        if it["is_diary"]:
            entries = it["entries_in_range"] if range_filtered else it["entries"]
            if entries:
                for e in entries:
                    docs.append({
                        "source": it["rel"], "type": "diary",
                        "date": e["date"], "heading": e["heading"],
                        "tags": it["tags"], "private": it["private"],
                        "title": e["heading"], "content": e["body"],
                    })
            elif not range_filtered:
                docs.append({
                    "source": it["rel"], "type": "diary", "date": it["date"],
                    "heading": None, "tags": it["tags"], "private": it["private"],
                    "title": Path(it["rel"]).stem, "content": it["body"],
                })
        else:
            docs.append({
                "source": it["rel"], "type": it["type"], "date": it["date"],
                "heading": None, "tags": it["tags"], "private": it["private"],
                "title": Path(it["rel"]).stem, "content": it["body"],
            })
    payload = {
        "schema_version": 1,
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "count": len(docs),
        "documents": docs,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_csv(items: List[Dict[str, Any]], range_filtered: bool) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "type", "tags", "private", "source", "title", "content"])
    rows: List[Tuple[str, List[Any]]] = []
    for it in items:
        if it["is_diary"]:
            entries = it["entries_in_range"] if range_filtered else it["entries"]
            if not entries and not range_filtered:
                entries = [{"date": it["date"], "heading": None, "body": it["body"]}]
            for e in entries:
                rows.append((e["date"] or "9999-99-99", [
                    e["date"] or "", "diary", "|".join(it["tags"]),
                    "1" if it["private"] else "0", it["rel"],
                    e.get("heading") or Path(it["rel"]).stem,
                    e["body"].replace("\r", " ").replace("\n", " ⏎ "),
                ]))
        else:
            rows.append((it["date"] or "9999-99-99", [
                it["date"] or "", it["type"], "|".join(it["tags"]),
                "1" if it["private"] else "0", it["rel"],
                Path(it["rel"]).stem,
                it["body"].replace("\r", " ").replace("\n", " ⏎ "),
            ]))
    rows.sort(key=lambda x: x[0])
    for _, row in rows:
        writer.writerow(row)
    return "﻿" + buf.getvalue()   # BOM：Excel 打开中文不乱码


def _sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip().strip(".") or "export"


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def export_diary(*, fmt: str = "zip", **kwargs) -> Dict[str, Any]:
    """生成导出物。

    fmt: zip / md（单篇原样）/ merged（合并 Markdown）/ json / csv
    返回：{filename, media_type, content: bytes} 或 {path: 临时 ZIP 路径}
    """
    res = collect_files(**kwargs)
    items = res["items"]
    if not items:
        raise ValueError("没有符合条件的文件可导出")
    range_filtered = _range_active(kwargs)
    stamp = _stamp()
    prefix = "日记" if kwargs.get("type_filter") == "diary" else "知识库材料"

    if fmt in ("merged", "md-merged"):
        content = _build_merged_markdown(items, range_filtered).encode("utf-8")
        return {"filename": f"{prefix}合并_{stamp}.md",
                "media_type": "text/markdown; charset=utf-8", "content": content}
    if fmt == "json":
        content = _build_json(items, range_filtered).encode("utf-8")
        return {"filename": f"{prefix}导出_{stamp}.json",
                "media_type": "application/json; charset=utf-8", "content": content}
    if fmt == "csv":
        content = _build_csv(items, range_filtered).encode("utf-8-sig")
        return {"filename": f"{prefix}导出_{stamp}.csv",
                "media_type": "text/csv; charset=utf-8", "content": content}
    if fmt == "md" and len(items) == 1:
        it = items[0]
        return {"filename": Path(it["rel"]).name,
                "media_type": "text/markdown; charset=utf-8",
                "content": it["raw"].encode("utf-8")}

    # 其余情况（多文件 md，或显式 zip）→ ZIP，保留 docs/ 下目录结构
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    zip_name = f"{prefix}导出_{stamp}.zip"
    tmp = EXPORT_DIR / f".{zip_name}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        used = set()
        for it in items:
            arc = it["rel"].replace("/", "\\")
            # 极端情况下重名加后缀
            base = arc
            n = 1
            while arc in used:
                p = Path(base)
                arc = f"{p.stem}_{n}{p.suffix}"
                n += 1
            used.add(arc)
            zf.writestr(arc, it["raw"])
    final = EXPORT_DIR / zip_name
    tmp.replace(final)
    return {"filename": zip_name,
            "media_type": "application/zip",
            "path": final}


# ------------------------------------------------------------------
# 聊天记录导出（序列化在 chat_store，这里做私密引用兜底过滤）
# ------------------------------------------------------------------

def filter_private_citations(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """导出前再兜底一次：剔除任何引用了私密文件的 citation（理论上检索已过滤）。"""
    priv = private_source_set()
    if not priv:
        return turns
    for r in turns:
        cites = r.get("citations") or []
        r["citations"] = [c for c in cites if c.get("source") not in priv]
    return turns
