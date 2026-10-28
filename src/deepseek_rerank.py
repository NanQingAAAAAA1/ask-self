"""DeepSeek 证据重排。

批处理优化：把 Top-K 片段组装成**一个** Prompt，一次 API 调用完成全部打分，
返回 JSON 数组，避免逐条调用带来的高延迟与高成本。

强制配置：response_format={"type": "json_object"}、temperature=0。

**阈值策略（方案2）**：入参 ``threshold`` 是「绝对阈值」（默认 7）。
固定 7 分对个人笔记太严——零散一句笔记常被打 5~6 分，全库没一条够 7 就被整段丢掉，
表现为「明明 docs 里有却说没有」。因此引入**相对阈值**：
    有效阈值 = min(threshold, max(FLOOR, 最高分 - GAP))
即「有一条够高就按它往下让 GAP 分收人，但不低于 FLOOR」。
``config.JEV_THRESHOLD_FLOOR <= 0`` 时关闭该逻辑，退化为纯绝对阈值。

对外接口 `rerank(question, docs, threshold, max_items)`，agent.py 直接调用。

用法（自测）：
    python src/deepseek_rerank.py "我该不该换工作"
"""
import json
import re
import sys
from typing import Dict, List, Tuple

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    JEV_THRESHOLD,
    JEV_THRESHOLD_FLOOR,
    JEV_THRESHOLD_GAP,
    MAX_RERANK_ITEMS,
)

# ---------- 打分 Prompt（严格按需求提供的模板） ----------
_PROMPT_HEADER = (
    "你是证据相关性评估专家。对以下每个片段，判断其与用户问题的相关性。\n"
    "用户问题：{question}\n"
    "{chunks}"
)
_PROMPT_TAIL = (
    "\n请严格按以下 JSON 数组格式输出，每个片段一个对象：\n"
    "[\n"
    '{"index": 1, "score": <1-10的整数>, "reason": "<简短理由>", "is_relevant": <true/false>},\n'
    '{"index": 2, "score": <1-10的整数>, "reason": "<简短理由>", "is_relevant": <true/false>},\n'
    "...\n"
    "]"
)


def _client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=60)


def _build_prompt(question: str, items: List[Dict]) -> str:
    """按模板拼装批量打分 Prompt。"""
    chunks = "\n".join(f"片段{i}：{d['text']}" for i, d in enumerate(items, 1))
    head = _PROMPT_HEADER.replace("{question}", question).replace("{chunks}", chunks)
    return head + _PROMPT_TAIL


def _load_json(raw: str):
    """容错解析：处理 ```json 包裹、前后夹带文字的情况。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        t = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", (raw or "").strip())
        for op, cl in (("[", "]"), ("{", "}")):
            i, j = t.find(op), t.rfind(cl)
            if i != -1 and j > i:
                try:
                    return json.loads(t[i:j + 1])
                except json.JSONDecodeError:
                    continue
        raise


# 优先在这些键下找打分数组（content 是 json_object 模式下的实际键名）
_PREFERRED_KEYS = ("results", "result", "data", "items", "list",
                   "scores", "content", "chunks", "output", "value")


def _find_array(node):
    """在任意嵌套结构里，递归找出第一个「元素为对象的数组」。

    模型被 json_object 模式约束时，会把本该是数组的结果层层包起来，实测出现过：
      {"type":"json_object","content":[ {...}, {...} ]}            ← content 是数组
      {"type":"json_object","content":"[\\n {...}, {...} ]"}        ← content 是字符串
      {"results":[...]}  /  裸数组  /  ```json [...] ```
    所以要能穿透 dict、字符串化的 JSON、以及更深的嵌套。
    """
    if isinstance(node, list):
        if any(isinstance(x, dict) for x in node):
            return node
        return None
    if isinstance(node, dict):
        for key in _PREFERRED_KEYS:
            if key in node:
                found = _find_array(node[key])
                if found:
                    return found
        for v in node.values():
            found = _find_array(v)
            if found:
                return found
        return None
    if isinstance(node, str):
        t = node.strip()
        if t[:1] in ("[", "{"):
            try:
                return _find_array(json.loads(t))
            except json.JSONDecodeError:
                return None
    return None


def _parse_scores(raw: str) -> Dict[int, Dict]:
    """解析模型输出，尽力兼容各种包装形态。

    该容错是必需的：一旦解析漏项，所有片段会被打成 0 分丢弃，导致明明检索到了证据
    却全部作废并误触发网络回退（表现为"完全没用我的 docs"）。
    最外层解析失败会向上抛异常，由 rerank 走「降级为不过滤」的保护。
    """
    data = _load_json(raw)

    arr = _find_array(data)
    if arr is None and isinstance(data, dict):
        # 最后兜底：形如 {"1": {...}, "2": {...}} 的映射
        arr = [v for v in data.values() if isinstance(v, dict)]

    scores: Dict[int, Dict] = {}
    for pos, obj in enumerate(arr or [], 1):
        if not isinstance(obj, dict):
            continue
        try:
            idx = int(obj.get("index", pos))
        except (TypeError, ValueError):
            idx = pos
        scores[idx] = obj
    return scores


def rerank(
    question: str,
    docs: List[Dict],
    threshold: float = JEV_THRESHOLD,
    max_items: int = MAX_RERANK_ITEMS,
) -> List[Dict]:
    """对 docs 批量打分，保留 score >= threshold 的片段（原地写入 score/reason/is_relevant）。

    返回按分数降序排列的高置信证据列表。
    """
    if not docs:
        return []

    items = docs[:max_items]
    prompt = _build_prompt(question, items)

    scores: Dict[int, Dict] = {}
    parse_failed = False
    try:
        resp = _client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},   # 强制 JSON
            temperature=0,                              # 强制确定性
        )
        raw = resp.choices[0].message.content or "{}"
        scores = _parse_scores(raw)
        if not scores:
            parse_failed = True
            print(f"[warn] 重排返回无法解析，原文前 200 字：{raw[:200]}", file=sys.stderr)
    except Exception as e:
        parse_failed = True
        print(f"[warn] DeepSeek 重排失败：{e}", file=sys.stderr)

    # 降级保护：打分完全不可用时，宁可不过滤，也不能把检索到的证据全丢掉
    # （否则会连锁触发「改写 → 网络回退」，表现为"完全没用我的 docs"）
    if parse_failed and items:
        print(f"[warn] 重排不可用，降级为保留全部 {len(items)} 条（未过滤）", file=sys.stderr)
        kept = [{**d, "score": 0.0, "reason": "重排不可用，未过滤", "is_relevant": True}
                for d in items]
        return kept

    scored: List[Dict] = []
    for i, doc in enumerate(items, 1):
        obj = scores.get(i) or {}
        try:
            sc = float(obj.get("score", 0))
        except (TypeError, ValueError):
            sc = 0.0
        scored.append({
            **doc,
            "score": round(max(0.0, min(10.0, sc)), 1),
            "reason": obj.get("reason", ""),
            "is_relevant": bool(obj.get("is_relevant", False)),
        })

    # 【方案2】相对阈值：固定阈值 7 对「零散一句笔记」太严（模型诚实打 5~6 分就被全丢）。
    # 改为「绝对阈值」与「相对阈值」取较松的那个：
    #     eff = min(threshold, max(FLOOR, 最高分 - GAP))
    # 含义：只要有一条够高（如 8 分），就按 8-2=6 收人，而不是死守 7；
    #       全库最高才 5 分时也不无限放宽，最低收到 FLOOR（默认 6），防止噪声混入。
    # FLOOR<=0 时关闭相对阈值，退化为原来的绝对阈值。
    eff_threshold = threshold
    if scored and JEV_THRESHOLD_FLOOR > 0:
        top = max(d["score"] for d in scored)
        rel = max(JEV_THRESHOLD_FLOOR, top - JEV_THRESHOLD_GAP)
        eff_threshold = min(threshold, rel)

    kept = [d for d in scored if d["score"] >= eff_threshold]
    kept.sort(key=lambda x: x["score"], reverse=True)
    print(f"[rerank] 批量打分 {len(scored)} 条，保留 {len(kept)} 条"
          f"（有效阈值 {eff_threshold}，绝对 {threshold}）", file=sys.stderr)
    return kept


def main():
    sys.path.insert(0, ".")
    from retrieve import retrieve

    question = sys.argv[1] if len(sys.argv) > 1 else "我该不该换工作"
    docs = retrieve(question, top_k=8)
    print(f"\n候选 {len(docs)} 条，一次调用批量打分…\n")
    kept = rerank(question, docs)
    print(f"\n保留 {len(kept)} 条高置信证据：\n")
    for i, d in enumerate(kept, 1):
        print(f"[{i}] {d['score']} 分 · {d['source']}  （{d.get('reason','')}）")
        print(f"    {d['text'][:100]}…\n")


if __name__ == "__main__":
    main()
