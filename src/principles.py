"""个人原则（硬约束）读写。

与 docs/ 证据是两条完全不同的通道：

    docs/ 证据：意图判断 → 相似度检索 → 打分 → 命中才用，可被引用；
    个人原则： 每次生成回答都**无条件注入** system prompt 顶部，不得违背。

因此原则**不进向量库**，单独存成 ``memory/principles.md``（见 config.PRINCIPLES_PATH），
避免混进「证据列表」被当作日记片段引用。

对外接口：
    load_principles()      读取正文（带 mtime 缓存，改文件即自动失效）
    save_principles(text)  覆盖写入
    principles_for_prompt() 取出可直接拼进 system prompt 的段落；无实质内容时返回 ""
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from config import PRINCIPLES_PATH

# 只有标题 / 引用 / 空条目时，视为「没写实质内容」，不注入（避免模板本身被喂给模型）
_BULLET_ONLY = re.compile(r"^[-*+]\s*$|^\d+[.)]\s*$")

_TEMPLATE = """# 个人原则（硬约束）

> 这里写「无论何时都必须遵守」的底线。
> 它**不参与检索**，而是每次提问都被无条件注入，回答不得与之相悖；
> 与 AI 的建议冲突时，以这里为准。
> 只写约束、不写日记，越简短越有力。

## 一、总纲
- 

## 二、学业 / 成长
- 

## 三、金钱 / 消费
- 

## 四、时间 / 精力
- 
"""

# mtime 缓存：文件没变就不重复读盘
_cache: dict = {"mtime": None, "text": ""}


def ensure_exists() -> Path:
    """确保原则文件存在（不存在则落一份模板），返回其路径。"""
    p = Path(PRINCIPLES_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(_TEMPLATE, encoding="utf-8")
        _cache["mtime"] = None      # 新文件，缓存作废
    return p


def load_principles(*, use_cache: bool = True) -> str:
    """读取原则正文。文件不存在时自动创建模板并返回模板内容。"""
    p = ensure_exists()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return ""
    if use_cache and _cache["mtime"] == mtime:
        return _cache["text"]
    text = p.read_text(encoding="utf-8", errors="replace")
    _cache["mtime"] = mtime
    _cache["text"] = text
    return text


def save_principles(text: str) -> Path:
    """覆盖写入原则正文（下次提问立即生效，无需重启/重建索引）。"""
    p = ensure_exists()
    p.write_text(text or "", encoding="utf-8")
    _cache["mtime"] = None          # 主动失效，避免读到旧内容
    return p


def has_content(text: str) -> bool:
    """判断正文里是否有「实质内容」（排除标题、引用、空条目、纯空白）。"""
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(">"):
            continue
        if _BULLET_ONLY.match(s):
            continue
        return True
    return False


def _strip_boilerplate(text: str) -> str:
    """净化注入内容，只保留「有实质内容的标题 + 条目」。

    剔除三类噪音（它们对人有用、对模型是纯 token 浪费）：
      1. 以 ``>`` 开头的引用行 —— 写给「人」看的用法说明；
      2. 只有符号没有内容的空条目（如 ``- ``）；
      3. 因此变成空壳的小节标题（下面一条有效内容都没有）。
    标题与正文原样保留。
    """
    lines = [ln for ln in (text or "").splitlines() if not ln.lstrip().startswith(">")]
    out: List[str] = []
    pending_heading: List[str] = []          # 先挂起标题，等确认有内容再落下
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            pending_heading.append(ln)
            continue
        if _BULLET_ONLY.match(s):            # 空条目
            continue
        if pending_heading:
            if out:
                out.append("")               # 标题前留空行，读起来清楚
            out.extend(pending_heading)
            pending_heading = []
        out.append(ln)
    return "\n".join(out).strip()


def principles_for_prompt() -> str:
    """返回可直接注入 system prompt 的原则段落；无实质内容时返回空串。"""
    text = load_principles()
    if not has_content(text):
        return ""
    return _strip_boilerplate(text)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台中文不乱码
    path = ensure_exists()
    body = load_principles()
    print(f"原则文件：{path}")
    print(f"有实质内容：{has_content(body)}  字数：{len(body)}")
    print("--- 内容 ---")
    print(body)
