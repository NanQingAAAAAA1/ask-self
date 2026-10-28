"""日记分析：单篇「日记卡」生成 + 指纹缓存 + 周期聚合总结。

分析范围 = ``docs/`` 下的**所有文件**（日记、聊天记录、网课/学习笔记、收藏文章、
清单等都算「个人记录」），与 ingest 录入的范围一致，只按材料形态决定切分粒度：

  * **按月合集的日记**（frontmatter ``type: diary``、文件名 ``diary-`` 前缀、
    或位于 ``docs/diary/`` 文件夹）按裸日期标题（``10.1``、``9.6``、
    ``2026-08-15``、``8月24日``）切成「一天一条」；
  * 其他文件（文章、笔记、清单、聊天记录等）**整篇作为一张卡**，日期取
    frontmatter ``date`` 或文件名中的日期，没有日期就是「无日期」卡。

其余设计边界（与 README 的隐私口径一致）：
  * 缓存 key = 条目正文 SHA1 + prompt 版本（与 ingest.py 的 chash 同思路），
    正文没变就不重算；缓存在 ``analysis_cache/``（已 gitignore，不进向量库）。
  * ``private: true`` 的文件默认不分析，必须显式 include_private 才纳入。
  * 单篇卡 prompt 短、强制 JSON，失败重试一次后降级为「仅正则/词典抽取」（degraded）。

对外主要接口：
    scan_diaries(include_private=False)      → 扫描全部记录条目 + 缓存状态
    generate_cards(...)                       → 批量补齐缺失的日记卡（分批调用）
    build_period_summary(...)                 → 周期聚合（周/月/季/年/自定义）
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import datetime as dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    ANALYSIS_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DOCS_DIR,
    REPORTS_DIR,
    ROOT,
)
from ingest import infer_date, infer_source_type, parse_frontmatter

# prompt 升级时 bump 版本号 → 旧缓存自动失效、重新生成
CARD_PROMPT_VERSION = "v1"
CARD_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1

# 单次周期总结喂给模型的「日记卡摘要」上限（条），超过则先分块总结再合并
SUMMARY_BATCH_SIZE = 20

# 运行中终止（本地单用户，进程内事件即可）：
#   卡片分批生成在每「篇」之间检查；周期总结在每「分块」和最终合并前检查。
_CARDS_CANCEL = threading.Event()
_SUMMARY_CANCEL = threading.Event()


def request_cancel(task: str = "all") -> None:
    """请求终止正在进行的生成。task = cards / summary / all。"""
    if task in ("cards", "all"):
        _CARDS_CANCEL.set()
    if task in ("summary", "all"):
        _SUMMARY_CANCEL.set()


def is_cancelled(task: str) -> bool:
    return (_CARDS_CANCEL if task == "cards" else _SUMMARY_CANCEL).is_set()

# ------------------------------------------------------------------
# 1. 日记发现与按天切分
# ------------------------------------------------------------------

# 裸日期标题：整行只有一个日期（允许前面有 # 标题号）
#   10.1 / 10．1 / 10/1 / 2026-08-15 / 2026年8月15日 / 2026.8.15
_DATE_LINE_RE = re.compile(
    r"^\s*#{0,6}\s*"
    r"(?:"
    r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*(?:[-/.月]\s*(\d{1,2}))?\s*日?"
    r"|"
    r"(\d{1,2})\s*[.．/]\s*(\d{1,2})"
    r")\s*$"
)


def is_diary_file(fp: Path, meta: Optional[Dict[str, Any]] = None) -> bool:
    """判断一个 docs/ 下的文件是否属于日记。

    命中任一即为日记：
      1. frontmatter ``type: diary``；
      2. 文件名 ``diary-`` 前缀（ingest 的命名约定）；
      3. 位于 ``docs/diary/`` 文件夹下（本项目实际的合集日记目录）。
    """
    try:
        if meta is None:
            meta, _ = parse_frontmatter(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        meta = meta or {}
    if str(meta.get("type", "")).strip().lower() == "diary":
        return True
    if infer_source_type(fp.name) == "diary":
        return True
    try:
        parts = fp.relative_to(DOCS_DIR).parts
    except ValueError:
        return False
    return len(parts) >= 2 and parts[0].lower() == "diary"


def _parse_date_line(line: str, today: dt.date) -> Optional[str]:
    """把一行裸日期标题解析成 ``YYYY-MM-DD``；不是日期行返回 None。"""
    m = _DATE_LINE_RE.match(line.rstrip("\n"))
    if not m:
        return None
    try:
        if m.group(2):  # 带年份
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3)) if m.group(3) else 1
            return dt.date(year, month, day).isoformat()
        month = int(m.group(4))
        day = int(m.group(5))
    except (ValueError, TypeError):
        return None
    # 只有 M/D：按「不晚于今天」推断年份（年底记跨年的 1 月日记时归到去年）
    year = today.year
    try:
        candidate = dt.date(year, month, day)
    except ValueError:
        return None
    if candidate > today:
        candidate = dt.date(year - 1, month, day)
    return candidate.isoformat()


def split_entries(text: str, today: Optional[dt.date] = None) -> List[Dict[str, str]]:
    """把一篇日记正文按裸日期标题切成多条 ``{heading, date, body}``。

    没有任何日期标题时，返回单条「无日期」条目（date=None）。
    首个日期标题之前的导语若 >= 30 字，保留为一条无日期条目，否则丢弃。
    """
    today = today or dt.date.today()
    lines = text.splitlines()
    entries: List[Dict[str, Any]] = []
    cur_heading: Optional[str] = None
    cur_date: Optional[str] = None
    buf: List[str] = []

    def _flush() -> None:
        nonlocal buf, cur_heading, cur_date
        body = "\n".join(buf).strip()
        if body:
            entries.append({"heading": cur_heading, "date": cur_date, "body": body})
        buf = []

    for line in lines:
        parsed = _parse_date_line(line, today)
        if parsed is not None:
            _flush()
            cur_heading = line.strip().lstrip("#").strip()
            cur_date = parsed
        else:
            buf.append(line)
    _flush()

    # 导语（无日期标题）：太短多半是标题/空行，丢弃
    if entries and entries[0]["date"] is None and len(entries[0]["body"]) < 30:
        entries.pop(0)

    if not entries:
        return []
    return entries


def _fingerprint(body: str) -> str:
    """条目正文指纹（归一化空白后 SHA1），作为缓存文件名。"""
    norm = re.sub(r"\s+", "", body)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def scan_diaries(include_private: bool = False) -> Dict[str, Any]:
    """扫描 docs/ 下**全部**记录文件，返回条目列表与缓存状态。

    - 合集日记（diary 类型 / diary 文件夹）按裸日期标题切成「一天一条」；
    - 其他文件（聊天记录、笔记、文章、清单等）整篇作为一张卡。

    返回：
      {
        entries: [{source, source_type, title, date, heading, private, tags,
                   char_count, fingerprint, cached, degraded?}],
        total, cached, pending, private_excluded, undated
      }
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    private_excluded = 0

    if DOCS_DIR.exists():
        for fp in sorted(DOCS_DIR.rglob("*")):
            if not fp.is_file() or fp.suffix.lower() not in (".md", ".markdown", ".txt"):
                continue
            try:
                raw = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            meta, body = parse_frontmatter(raw)
            rel = fp.relative_to(ROOT).as_posix()
            private = bool(meta.get("private", False))
            tags_raw = meta.get("tags") or []
            tags = ([str(t).strip() for t in tags_raw]
                    if isinstance(tags_raw, list) else [str(tags_raw).strip()])
            tags = [t for t in tags if t]
            file_date = meta.get("date") or infer_date(fp.name)
            if not file_date:
                # 文件名前缀 M.D（如「10.11与AI的聊天.md」）
                m = re.match(r"^(\d{1,2})[.．](\d{1,2})", fp.stem)
                if m:
                    file_date = _parse_date_line(f"{m.group(1)}.{m.group(2)}", dt.date.today())
            file_date = str(file_date) if file_date else None
            source_type = str(meta.get("type") or infer_source_type(fp.name)).strip()
            diary = is_diary_file(fp, meta)
            source_type = "diary" if diary else source_type

            if private and not include_private:
                # 私密文件按「将产生的条目数」计数，保证前端提示篇数准确
                split = split_entries(body) if diary else None
                private_excluded += len(split) if split else 1
                continue

            # 合集日记按天切；其余文件整篇一张卡
            split = split_entries(body) if diary else []
            if split:
                items = [{"heading": it["heading"], "date": it["date"],
                          "body": it["body"], "title": it["heading"]}
                         for it in split]
            else:
                if not body.strip():
                    continue
                items = [{"heading": None, "date": file_date,
                          "body": body, "title": fp.stem}]
            for item in items:
                fp_hash = _fingerprint(item["body"])
                entries.append({
                    "source": rel,
                    "source_type": source_type,
                    "title": item.get("title") or fp.stem,
                    "date": item["date"] or (file_date if item["heading"] is None else None),
                    "heading": item["heading"],
                    "private": private,
                    "tags": tags,
                    "char_count": len(item["body"]),
                    "fingerprint": fp_hash,
                    "cached": _cache_path(fp_hash).exists(),
                    "degraded": _cache_degraded(fp_hash),
                })

    entries.sort(key=lambda e: (e["date"] or "9999-99-99", e["source"], e["heading"] or ""))
    return {
        "entries": entries,
        "total": len(entries),
        "cached": sum(1 for e in entries if e["cached"]),
        "pending": sum(1 for e in entries if not e["cached"]),
        "private_excluded": private_excluded,
        "undated": sum(1 for e in entries if not e["date"]),
    }


# ------------------------------------------------------------------
# 2. 日记卡缓存
# ------------------------------------------------------------------

def _cache_path(fingerprint: str) -> Path:
    return ANALYSIS_DIR / f"{fingerprint}.json"


def _cache_degraded(fingerprint: str) -> bool:
    p = _cache_path(fingerprint)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("degraded"))
    except Exception:
        return False


def load_card(fingerprint: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(fingerprint)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("prompt_version") != CARD_PROMPT_VERSION:
        return None
    return data


def _save_card(fingerprint: str, entry: Dict[str, Any], card: Dict[str, Any],
               *, degraded: bool) -> Dict[str, Any]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": CARD_SCHEMA_VERSION,
        "prompt_version": CARD_PROMPT_VERSION,
        "fingerprint": fingerprint,
        "source": entry["source"],
        "source_type": entry.get("source_type", ""),
        "title": entry.get("title", ""),
        "date": entry["date"],
        "heading": entry["heading"],
        "private": entry.get("private", False),
        "tags": entry.get("tags", []),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model": DEEPSEEK_MODEL,
        "degraded": degraded,
        "card": card,
    }
    tmp = _cache_path(fingerprint).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_cache_path(fingerprint))
    return record


# ------------------------------------------------------------------
# 3. 正则 / 词典双通道抽取（模型失败时的降级方案，也用于补充模型结果）
# ------------------------------------------------------------------

# 意图：第一人称的打算、决定、承诺
_INTENT_RE = re.compile(
    r"[^。！？\n]*(?:我想|我要|我打算|我准备|我决定|我计划|我以后|我下次|以后要|下次要|准备|打算)"
    r"[^。！？\n]*[。！？]?"
)
# 待办：事务性提醒
_TODO_RE = re.compile(r"[^。！？\n]*(?:记得|别忘了|待办|TODO|todo)[^。！？\n]*[。！？]?")

_EMOTION_LEXICON = {
    "焦虑": ["焦虑", "担心", "慌", "不安", "害怕", "怕", "紧张", "内耗"],
    "低落": ["低落", "难过", "丧", "失落", "迷茫", "绝望", "崩溃", "想哭", "郁闷"],
    "愤怒": ["愤怒", "生气", "讨厌", "烦", "恶心", "恨", "垃圾", "水货"],
    "疲惫": ["累", "疲惫", "困", "倦怠", "没精神", "无力"],
    "平静": ["平静", "放松", "安心", "淡定", "从容"],
    "开心": ["开心", "高兴", "兴奋", "激动", "爽", "喜悦", "快乐", "惊喜", "可爱"],
    "期待": ["期待", "希望", "想要", "憧憬"],
}
_POSITIVE = ["开心", "高兴", "兴奋", "爽", "好", "棒", "优秀", "顺利", "希望", "期待", "喜欢", "可爱", "放松", "从容", "值得"]
_NEGATIVE = ["焦虑", "担心", "害怕", "怕", "慌", "烦", "累", "难过", "低落", "丧", "绝望", "崩溃", "讨厌",
             "恶心", "内耗", "迷茫", "浪费", "压力", "痛苦", "不行", "失败", "水货", "垃圾"]


def _split_sentences(body: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?；;\n])", body)
    return [p.strip() for p in parts if p.strip()]


def regex_extract(body: str) -> Dict[str, Any]:
    """无模型通道：正则抽意图/待办、词典估情绪、挑长句当金句。"""
    sentences = _split_sentences(body)

    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()[:120]

    intentions = []
    seen = set()
    for m in _INTENT_RE.finditer(body):
        t = _clean(m.group(0))
        if len(t) >= 6 and t not in seen:
            seen.add(t)
            intentions.append({"text": t, "quote": t, "status": "open"})
    todos = []
    for m in _TODO_RE.finditer(body):
        t = _clean(m.group(0))
        if len(t) >= 4 and t not in seen:
            seen.add(t)
            todos.append({"text": t, "quote": t})

    emotions: List[str] = []
    for label, words in _EMOTION_LEXICON.items():
        if any(w in body for w in words):
            emotions.append(label)
    pos = sum(body.count(w) for w in _POSITIVE)
    neg = sum(body.count(w) for w in _NEGATIVE)
    sentiment = 0.0 if pos + neg == 0 else round(max(-1.0, min(1.0, (pos - neg) / (pos + neg))), 2)
    energy = 3
    if any(w in body for w in ["累", "疲惫", "困", "倦怠", "崩溃"]):
        energy = 2
    if any(w in body for w in ["兴奋", "激动", "干劲", "爽", "加油", "冲"]):
        energy = 4

    quotes = []
    for s in sorted(sentences, key=len, reverse=True):
        t = _clean(s)
        if 15 <= len(t) <= 120:
            quotes.append(t)
        if len(quotes) >= 2:
            break

    return {
        "summary": "",
        "events": [],
        "people": [],
        "places": [],
        "sentiment": sentiment,
        "energy": energy,
        "emotions": emotions,
        "intentions": intentions[:8],
        "todos": todos[:5],
        "quotes": quotes[:2],
        "principle_conflicts": [],
    }


# ------------------------------------------------------------------
# 4. DeepSeek 单篇卡
# ------------------------------------------------------------------

def _llm_chat(messages: List[Dict[str, str]], *, temperature: float = 0.2,
              timeout: float = 60, json_mode: bool = False) -> str:
    """统一的 DeepSeek 对话入口（分块叙事 / 合并 JSON 都走这里）。"""
    from openai import OpenAI

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=timeout)
    kwargs: Dict[str, Any] = {"model": DEEPSEEK_MODEL, "messages": messages,
                              "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _llm_chat_json(messages: List[Dict[str, str]], *, timeout: float = 60) -> str:
    return _llm_chat(messages, timeout=timeout, json_mode=True)


def _load_json(raw: str) -> Any:
    """容错解析：处理 ```json 包裹、前后夹带文字。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        t = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", (raw or "").strip())
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j > i:
            return json.loads(t[i:j + 1])
        raise


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _normalize_card(data: Dict[str, Any], body: str) -> Dict[str, Any]:
    """把模型输出归一化成固定 schema，并补正则兜底。"""
    fallback = regex_extract(body)

    def _as_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def _as_items(v: Any, text_key: str = "text") -> List[Dict[str, str]]:
        out = []
        if isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    text = str(x.get(text_key) or x.get("quote") or "").strip()
                    quote = str(x.get("quote") or text).strip()
                    if text:
                        item = {"text": text[:160], "quote": quote[:200]}
                        if text_key == "text" and x.get("status"):
                            item["status"] = str(x["status"])
                        out.append(item)
                elif str(x).strip():
                    out.append({"text": str(x).strip()[:160], "quote": str(x).strip()[:200]})
        return out

    card = {
        "summary": str(data.get("summary", "")).strip()[:1000],
        "events": _as_list(data.get("events"))[:10],
        "people": _as_list(data.get("people"))[:10],
        "places": _as_list(data.get("places"))[:10],
        "sentiment": round(_clamp(data.get("sentiment"), -1, 1, fallback["sentiment"]), 2),
        "energy": int(_clamp(data.get("energy"), 1, 5, fallback["energy"])),
        "emotions": _as_list(data.get("emotions"))[:6] or fallback["emotions"],
        "intentions": _as_items(data.get("intentions"))[:8] or fallback["intentions"],
        "todos": _as_items(data.get("todos"))[:6] or fallback["todos"],
        "quotes": _as_list(data.get("quotes"))[:3] or fallback["quotes"],
        "principle_conflicts": [],
    }
    # 意图状态默认 open
    for it in card["intentions"]:
        it.setdefault("status", "open")
        if it["status"] not in ("open", "done", "abandoned"):
            it["status"] = "open"
    raw_conflicts = data.get("principle_conflicts") or []
    if isinstance(raw_conflicts, list):
        for x in raw_conflicts:
            if isinstance(x, dict):
                principle = str(x.get("principle") or x.get("rule") or "").strip()
                quote = str(x.get("quote") or x.get("text") or "").strip()
                if principle or quote:
                    card["principle_conflicts"].append(
                        {"principle": principle[:160], "quote": quote[:200]}
                    )
    card["principle_conflicts"] = card["principle_conflicts"][:5]
    return card


def analyze_entry(entry: Dict[str, Any], body: str,
                  principle_text: str = "") -> Tuple[Dict[str, Any], bool]:
    """为单条日记生成结构化卡；返回 (card, degraded)。"""
    principle_block = ""
    if principle_text:
        principle_block = (
            "\n【用户的个人原则（仅供冲突比对，不是日记内容）】\n"
            f"{principle_text[:2000]}\n"
        )
    system = (
        "你是个人记录分析助手。读取用户的一篇个人记录（可能是某天的日记、聊天记录、"
        "网课/学习笔记、收藏文章或清单），输出严格 JSON，用于本地缓存与趋势回顾。"
        "只依据原文，不要编造；没有的字段给空数组/空字符串。\n"
        "JSON 字段：\n"
        '{"summary":"3-5句概括：当天主要发生了什么、做了什么决定、心情主线",'
        '"events":["关键事件，短语"],"people":["涉及的人，没有则空数组"],'
        '"places":["地点或涉及的项目/公司"],'
        '"sentiment":情绪极性,-1到1的小数(负=低落,正=积极),'
        '"energy":能量/精力,1到5的整数(1很疲惫,5很饱满),'
        '"emotions":["主要情绪标签，如 焦虑/平静/兴奋/低落/愤怒/疲惫"],'
        '"intentions":[{"text":"我想/我要/我打算做的事，短语","quote":"日记里对应的原文","status":"open"}],'
        '"todos":[{"text":"事务性待办，如记得交材料","quote":"原文"}],'
        '"quotes":["1-3句值得留存的原文金句，照抄不改写"],'
        '"principle_conflicts":[{"principle":"与之相悖的原则条目","quote":"日记里相悖表述的原文"}]}\n'
        "注意：intentions 是第一人称的意愿/承诺/决定；todos 是事务性提醒，二者分开。"
        + principle_block
    )
    kind = "某天的日记" if entry.get("heading") else "一篇个人记录"
    user = (f"材料类型：{kind}（{entry.get('source_type') or '未分类'}）\n"
            f"日期：{entry.get('date') or '未知'}\n来源文件：{entry.get('source')}\n\n"
            f"正文：\n{body[:6000]}")

    try:
        raw = _llm_chat_json([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        data = _load_json(raw)
        if not isinstance(data, dict):
            raise ValueError("模型输出不是 JSON 对象")
        return _normalize_card(data, body), False
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 日记卡生成失败，降级为正则抽取：{e}", file=sys.stderr)
        card = regex_extract(body)
        # 再试一次（官方建议失败可重试一次）
        try:
            raw = _llm_chat_json([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            data = _load_json(raw)
            if isinstance(data, dict):
                return _normalize_card(data, body), False
        except Exception as e2:  # noqa: BLE001
            print(f"[warn] 重试仍失败，使用降级结果：{e2}", file=sys.stderr)
        return card, True


def generate_cards(*, include_private: bool = False, limit: int = 8,
                   date_from: Optional[str] = None, date_to: Optional[str] = None,
                   force: bool = False) -> Dict[str, Any]:
    """为缺失（或 force 时全部）的日记条目生成卡片。

    每次调用最多处理 ``limit`` 条（前端循环调用以展示进度，避免单次 HTTP 过长）。
    每篇之间检查 ``_CARDS_CANCEL``，收到终止后立即返回已完成部分（cancelled=True）。
    """
    from principles import principles_for_prompt

    _CARDS_CANCEL.clear()  # 每批请求开始时清掉上一次的终止标记
    scan = scan_diaries(include_private=include_private)
    targets = [
        e for e in scan["entries"]
        if (force or not e["cached"])
        and _in_range(e["date"], date_from, date_to)
    ][:max(1, limit)]

    try:
        principle_text = principles_for_prompt()
    except Exception:
        principle_text = ""

    generated, failed, degraded = 0, 0, 0
    cancelled = False
    for e in targets:
        if _CARDS_CANCEL.is_set():
            cancelled = True
            break
        body = _read_entry_body(e)
        if not body:
            failed += 1
            continue
        try:
            card, is_degraded = analyze_entry(e, body, principle_text)
            _save_card(e["fingerprint"], e, card, degraded=is_degraded)
            generated += 1
            if is_degraded:
                degraded += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 分析 {e['source']} {e['heading']} 失败：{exc}", file=sys.stderr)
            failed += 1

    after = scan_diaries(include_private=include_private)
    return {
        "ok": True,
        "processed": generated,
        "failed": failed,
        "degraded": degraded,
        "total": after["total"],
        "cached": after["cached"],
        "pending": after["pending"],
        "private_excluded": after["private_excluded"],
        "cancelled": cancelled,
        "done": after["pending"] == 0 and not cancelled,
    }


def _read_entry_body(entry: Dict[str, Any]) -> str:
    """按 source+heading 重新读出条目正文（缓存只存指纹，正文不落地到缓存）。"""
    fp = ROOT / entry["source"]
    if not fp.exists():
        return ""
    meta, body = parse_frontmatter(fp.read_text(encoding="utf-8", errors="ignore"))
    if entry.get("heading"):
        for item in split_entries(body):
            if item["heading"] == entry["heading"] and item["date"] == entry["date"]:
                return item["body"]
        # 标题匹配退化：按日期匹配
        for item in split_entries(body):
            if item["date"] == entry["date"]:
                return item["body"]
    return body if body.strip() else ""


# ------------------------------------------------------------------
# 5. 查询与周期聚合
# ------------------------------------------------------------------

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


def _load_entries_with_cards(scan_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把扫描结果与缓存卡片拼成完整记录（仅含已缓存的）。"""
    out = []
    for e in scan_entries:
        rec = load_card(e["fingerprint"])
        if rec:
            out.append(rec)
    out.sort(key=lambda r: (r.get("date") or "9999-99-99", r.get("source") or ""))
    return out


def query_cards(*, date_from: Optional[str] = None, date_to: Optional[str] = None,
                include_private: bool = False,
                undated_only: bool = False) -> Dict[str, Any]:
    """返回区间内的日记卡 + 情绪曲线 + 主题/情绪聚合。

    undated_only=True 时忽略日期区间，只返回**无日期**的卡片（前端「其它」分组）。
    """
    scan = scan_diaries(include_private=include_private)
    records = _load_entries_with_cards(scan["entries"])
    if undated_only:
        records = [r for r in records if not r.get("date")]
    else:
        records = [r for r in records if _in_range(r.get("date"), date_from, date_to)]

    series = []
    emotion_counter: Dict[str, int] = {}
    theme_counter: Dict[str, int] = {}
    people_counter: Dict[str, int] = {}
    intentions, conflicts, todos_open = [], [], []
    for r in records:
        c = r["card"]
        series.append({
            "date": r.get("date"),
            "source": r.get("source"),
            "heading": r.get("heading"),
            "sentiment": c.get("sentiment", 0),
            "energy": c.get("energy", 3),
            "emotions": c.get("emotions", []),
            "degraded": r.get("degraded", False),
        })
        for em in c.get("emotions", []):
            emotion_counter[em] = emotion_counter.get(em, 0) + 1
        for ev in c.get("events", []):
            theme_counter[ev] = theme_counter.get(ev, 0) + 1
        for p in c.get("people", []):
            people_counter[p] = people_counter.get(p, 0) + 1
        for it in c.get("intentions", []):
            intentions.append({**it, "date": r.get("date"), "source": r.get("source")})
        for td in c.get("todos", []):
            if td.get("status", "open") == "open":
                todos_open.append({**td, "date": r.get("date"), "source": r.get("source")})
        for cf in c.get("principle_conflicts", []):
            conflicts.append({**cf, "date": r.get("date"), "source": r.get("source")})

    def _topn(counter: Dict[str, int], n: int) -> List[Dict[str, Any]]:
        return [{"name": k, "count": v} for k, v in
                sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]

    sentiments = [s["sentiment"] for s in series if s["sentiment"] is not None]
    energies = [s["energy"] for s in series]
    return {
        "ok": True,
        "range": {"from": date_from, "to": date_to},
        "cards": records,
        "series": series,
        "stats": {
            "entries": len(records),
            "avg_sentiment": round(sum(sentiments) / len(sentiments), 2) if sentiments else None,
            "avg_energy": round(sum(energies) / len(energies), 2) if energies else None,
            "undated_in_range": sum(1 for r in records if not r.get("date")),
        },
        "emotions_top": _topn(emotion_counter, 8),
        "events_top": _topn(theme_counter, 10),
        "people_top": _topn(people_counter, 8),
        "intentions": intentions,
        "todos_open": todos_open,
        "principle_conflicts": conflicts,
        "private_excluded": scan["private_excluded"],
        "pending": scan["pending"],
    }


def _digest(rec: Dict[str, Any]) -> str:
    """把一张日记卡压成喂给周期总结的短摘要（不喂全文）。"""
    c = rec["card"]
    parts = [f"[{rec.get('date') or '无日期'}]"]
    if c.get("summary"):
        parts.append(c["summary"])
    if c.get("events"):
        parts.append("事件：" + "；".join(c["events"][:4]))
    parts.append(f"情绪极性 {c.get('sentiment')}、能量 {c.get('energy')}、标签 {('、'.join(c.get('emotions', [])) or '无')}")
    if c.get("intentions"):
        parts.append("意图：" + "；".join(i.get("text", "") for i in c["intentions"][:4]))
    if c.get("quotes"):
        parts.append("金句：" + " / ".join(f"“{q}”" for q in c["quotes"][:2]))
    return " ".join(parts)


def build_period_summary(*, granularity: str = "month",
                         date_from: Optional[str] = None,
                         date_to: Optional[str] = None,
                         include_private: bool = False,
                         undated_only: bool = False,
                         save: bool = True,
                         should_cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """两级生成：先分块总结日记卡摘要，再合并成周期报告。

    每个结论要求带日期 + 原文引用（前端可点溯源）。
    should_cancel() 返回 True 时在分块之间 / 最终合并前中止，返回 cancelled=True。
    """
    _SUMMARY_CANCEL.clear()  # 进入一次生成即清掉旧标记
    _cancelled = should_cancel or (lambda: _SUMMARY_CANCEL.is_set())

    data = query_cards(date_from=date_from, date_to=date_to,
                       include_private=include_private, undated_only=undated_only)
    records = data["cards"]
    if not records:
        return {
            "ok": False,
            "error": "区间内没有已生成的日记卡，请先在分析页生成单篇日记卡。",
            "pending": data["pending"],
        }
    undated_count = sum(1 for r in records if not r.get("date"))

    digests = [_digest(r) for r in records]
    batches = [digests[i:i + SUMMARY_BATCH_SIZE]
               for i in range(0, len(digests), SUMMARY_BATCH_SIZE)]

    partials: List[str] = []
    for i, batch in enumerate(batches, 1):
        if _cancelled():
            return {
                "ok": False,
                "cancelled": True,
                "stage": "chunk",
                "partials_done": len(partials),
                "total_batches": len(batches),
                "entries": len(records),
            }
        sys_p = (
            "你在帮用户做日记周期回顾。下面是该周期内一批日记的结构化卡片摘要（每天一条）。"
            "请归纳这批日记：主要在忙什么、情绪主线、做成了什么、卡在哪里、反复出现的意图。"
            "用中文条目输出，每个关键判断后用【日期】标注依据。不要编造摘要之外的内容。"
        )
        user_p = "\n\n".join(batch)
        try:
            # 分块总结不走 json_object（输出是叙事条目）
            content = _llm_chat(
                [{"role": "system", "content": sys_p},
                 {"role": "user", "content": user_p}],
                temperature=0.3, timeout=90)
            partials.append(f"（批次 {i}/{len(batches)}）\n" + content)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 周期分块总结 {i} 失败：{e}", file=sys.stderr)
            partials.append(f"（批次 {i} 总结失败）")

    if undated_only:
        label = "其它回顾"
        range_txt = "无日期的其它记录（笔记 / 文章 / 清单 / 聊天记录）"
    else:
        label = {"day": "每日", "week": "周报", "month": "月报", "quarter": "季度回顾",
                 "year": "年度回顾", "custom": "区间回顾"}.get(granularity, "周期回顾")
        range_txt = f"{date_from or '最早'} ~ {date_to or '最新'}"

    if _cancelled():
        return {"ok": False, "cancelled": True, "stage": "merge",
                "partials_done": len(partials), "total_batches": len(batches),
                "entries": len(records)}

    principle_block = ""
    try:
        from principles import principles_for_prompt
        ptext = principles_for_prompt()
        if ptext:
            principle_block = (
                "\n【用户的个人原则】若发现日记中的决定/表述与原则冲突，在「冲突与提醒」里指出，"
                "引用原文与日期；原则本身不是日记内容。\n" + ptext[:1500]
            )
    except Exception:
        pass

    sys_m = (
        "你是个人日记回顾助手。用户会给你同一周期内分批归纳的日记摘要，"
        "请合并成一份诚实、具体、可溯源的中文周期回顾。结构：\n"
        "1. 这段时间主要在忙什么（叙事，2-4 段）\n"
        "2. 情绪主线与情绪低点集中在什么主题/人物/事件上\n"
        "3. 做成了什么 / 卡在哪里\n"
        "4. 反复出现的意图与决策追踪（同一意图反复出现要点名，并给出第一次/最近一次日期）\n"
        "5. 高频主题 Top N（若能判断相比上一周期的升降也写出）\n"
        "6. 前后矛盾的表述（只并列两条原文+日期，不做评判；没有则省略）\n"
        "7. 冲突与提醒（与个人原则相悖之处；没有则省略）\n"
        "8. 下一周期的具体建议（3-5 条）\n"
        "所有判断必须带【日期】依据，不编造。\n"
        "同时输出严格 JSON：{'markdown': 完整 Markdown 报告, "
        "'themes': [{'name': 主题名, 'count': 出现次数}], "
        "'triggers': ['情绪低点的触发因素'], "
        "'decisions': [{'intent': '反复出现的意图', 'first_date': '', 'last_date': '', "
        "'status': 'still_open|landed|dropped'}], "
        "'contradictions': [{'topic': '', 'quote_a': '原文', 'date_a': '', 'quote_b': '原文', 'date_b': ''}], "
        "'citations': [{'date': 'YYYY-MM-DD', 'source': 文件名, 'quote': '支撑报告关键结论的原文'}]}。"
        + principle_block
    )
    scope_note = (
        f"本次汇总的是**无日期的其它记录**（共 {len(records)} 篇，按文件成卡），"
        "没有时间线，请按主题/材料归纳，不要编造日期。\n"
        if undated_only else
        f"共 {len(records)} 张卡片（其中无日期的其它记录 {undated_count} 篇，按 [无日期] 标注，"
        "不要替它们猜测日期）。\n"
    )
    user_m = (f"周期类型：{label}\n区间：{range_txt}\n{scope_note}\n" + "\n\n".join(partials))

    structured: Dict[str, Any] = {}
    try:
        raw = _llm_chat_json([
            {"role": "system", "content": sys_m},
            {"role": "user", "content": user_m},
        ], timeout=120)
        structured = _load_json(raw) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 周期总结生成失败：{e}", file=sys.stderr)
        structured = {"markdown": "\n\n".join(partials)}

    markdown = str(structured.get("markdown") or "\n\n".join(partials)).strip()
    report = {
        "ok": True,
        "label": label,
        "range": {"from": date_from, "to": date_to},
        "granularity": granularity,
        "undated_only": undated_only,
        "entries": len(records),
        "undated": undated_count,
        "markdown": markdown,
        "themes": structured.get("themes") or data["events_top"],
        "triggers": structured.get("triggers") or [],
        "decisions": structured.get("decisions") or [],
        "contradictions": structured.get("contradictions") or [],
        "citations": structured.get("citations") or [],
        "series": data["series"],
        "stats": data["stats"],
        "emotions_top": data["emotions_top"],
        "people_top": data["people_top"],
        "private_excluded": data["private_excluded"],
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }

    if save:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = "其它无日期" if undated_only else f"{date_from or 'early'}_{date_to or 'now'}"
            fname = f"日记{label}_{stamp}.md"
            fname = re.sub(r"[\\/:*?\"<>|]", "_", fname)
            cover = f"覆盖 {len(records)} 篇记录" + (
                f"（含无日期 {undated_count} 篇）" if undated_count and not undated_only else "")
            header = (
                f"# 问己 · {label}（{range_txt}）\n\n"
                f"> 生成时间：{report['generated_at']}　·　{cover}"
                f"（{'已排除私密记录' if not include_private else '包含私密记录'}）\n\n---\n\n"
            )
            (REPORTS_DIR / fname).write_text(header + markdown + "\n", encoding="utf-8")
            report["saved_path"] = (REPORTS_DIR / fname).relative_to(ROOT).as_posix()
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 报告落盘失败：{e}", file=sys.stderr)

    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse

    parser = argparse.ArgumentParser(description="日记分析自测")
    parser.add_argument("--scan", action="store_true", help="只扫描并打印条目/缓存状态")
    parser.add_argument("--generate", action="store_true", help="为缺失条目生成卡片（限 --limit 条）")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--include-private", action="store_true")
    args = parser.parse_args()

    s = scan_diaries(include_private=args.include_private)
    print(f"日记条目 {s['total']}：已缓存 {s['cached']} / 待生成 {s['pending']}"
          f" / 无日期 {s['undated']} / 排除私密 {s['private_excluded']}")
    for e in s["entries"]:
        flag = "✓" if e["cached"] else "·"
        print(f"  [{flag}] {e['date'] or '无日期':<10} {e['heading'] or '-':<8} "
              f"{e['char_count']:>5} 字  {e['source']}")
    if args.generate:
        r = generate_cards(include_private=args.include_private, limit=args.limit)
        print(r)
