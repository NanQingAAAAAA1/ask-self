"""LangGraph 工作流：意图判断 → 检索 → 证据重排(DeepSeek) → 改写重试 → 网络回退 → 生成。

对外只暴露一个函数 run_agent(...)，UI 层只调用它（对应计划 7.2）。

另有一条与检索**并行**的通道：`memory/principles.md`（个人原则·硬约束）
在生成阶段被无条件注入 system prompt，不走检索、不进向量库，详见 generate_node。
"""
import json
import sys
from typing import Dict, List, Optional, TypedDict

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    JEV_THRESHOLD,
    TOP_K,
)

# ------------------------------------------------------------------
# 1. 状态定义（对应计划 5.1）
# ------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    question: str
    history: List[Dict]
    top_k: int
    jev_threshold: float
    use_jev: bool
    use_web_search: bool
    sources: Optional[List[str]]     # @ 引用限定的检索范围（文件/文件夹路径），空=全库

    need_retrieval: bool
    retrieved_docs: List[Dict]
    selected_docs: List[Dict]
    web_results: List[Dict]
    answer: str
    citations: List[Dict]
    evidence_count: int
    retry_count: int
    web_search_needed: bool


def _llm_client():
    """DeepSeek V3 客户端（意图分类 / 改写 / 生成共用）。"""
    from openai import OpenAI
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL, timeout=60)


def _chat(messages, json_mode: bool = False) -> str:
    """统一的 DeepSeek 调用入口。"""
    client = _llm_client()
    kwargs = {"model": DEEPSEEK_MODEL, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ------------------------------------------------------------------
# 2. 节点实现（对应计划 5.2）
# ------------------------------------------------------------------


def classify_intent(state: AgentState) -> AgentState:
    """判断是否需要检索知识库，返回 need_retrieval 布尔值和理由。

    判定倾向：**默认检索**。只有明确与个人记录无关的知识性提问才跳过，
    避免"我现在很无聊该做什么"这类带个人色彩的问题被误判为闲聊而漏检。
    """
    question = state["question"]
    prompt = (
        "判断用户问题是否需要检索「个人知识库」（本人的日记、随想、计划清单、网课笔记、工作与学习记录）。\n"
        "\n"
        "【默认需要检索】拿不准时一律判 true。\n"
        "\n"
        "【硬规则·命中即 true】只要问题里出现**第一人称的意愿或行为**——"
        "如「我想… / 我要… / 我准备… / 我打算… / 我该不该… / 我犹豫…」——"
        "**无论它后面接的话题看起来多通用**（买东西、去哪里、学什么、用什么工具…），"
        "一律判 true。因为这表示用户在按**自己**的处境求建议，答案应当先翻他自己的记录。\n"
        "反例（务必判 true）：「我想在闲鱼上买东西，有什么需要注意的地方吗」——"
        "不要因为「闲鱼买家注意事项」是通用话题就判 false；用户是在问**他自己**该怎么做。\n"
        "\n"
        "【以下情况需要检索 = true】即使问题表面像通用提问：\n"
        "- 涉及说话人本人的情绪、状态、喜好、习惯、打算、困惑、纠结、选择；\n"
        "- 寻求建议 / 办法 / 推荐，而合适的答案取决于此人已有的记录、偏好或经历；\n"
        "- 问及本人过去做过什么、记过什么、学过什么、计划过什么。\n"
        "\n"
        "【以下情况不需要检索 = false】仅限明确与个人记录无关的：\n"
        "- 纯数学计算、代码语法、翻译、名词定义等知识性提问（且**不含第一人称意愿/行为**）；\n"
        "- 纯打招呼寒暄，或要求撰写与个人经历无关的通用文本。\n"
        "\n"
        '只输出 JSON：{"need_retrieval": true/false, "reason": "简述"}'
    )
    try:
        raw = _chat(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": question}],
            json_mode=True,
        )
        data = json.loads(raw)
        need = bool(data.get("need_retrieval", True))
    except Exception as e:
        print(f"[warn] 意图判断失败，默认需要检索：{e}", file=sys.stderr)
        need = True
    return {"need_retrieval": need}


def retrieve_node(state: AgentState) -> AgentState:
    """调用 ChromaDB 检索 Top-K（纯本地）。带 @ 引用时只在该范围内检索。"""
    from retrieve import retrieve
    docs = retrieve(
        state["question"],
        top_k=state.get("top_k", TOP_K),
        sources=state.get("sources") or None,
    )
    return {"retrieved_docs": docs}


def jev_rerank_node(state: AgentState) -> AgentState:
    """证据重排：DeepSeek 批量打分，保留 score >= 阈值。use_jev=False 时全部保留。"""
    docs = state.get("retrieved_docs", [])
    if not state.get("use_jev", True):
        return {"selected_docs": docs, "evidence_count": len(docs)}

    from deepseek_rerank import rerank
    kept = rerank(
        state["question"], docs, threshold=state.get("jev_threshold", JEV_THRESHOLD)
    )
    return {"selected_docs": kept, "evidence_count": len(kept)}


def rewrite_query(state: AgentState) -> AgentState:
    """证据不足时改写查询并重新检索（最多 2 轮）。"""
    question = state["question"]
    round_no = state.get("retry_count", 0) + 1
    prompt = (
        f"用户问题在个人知识库中检索到的有效证据不足。请改写这个问题，"
        f"使用更多同义词和更宽泛的表述，以便检索到相关记录。"
        f"只输出改写后的问题本身，不要解释。"
    )
    try:
        new_q = _chat(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": question}]
        ).strip()
    except Exception as e:
        print(f"[warn] 查询改写失败，沿用原问题：{e}", file=sys.stderr)
        new_q = question

    from retrieve import retrieve
    docs = retrieve(new_q, top_k=state.get("top_k", TOP_K),
                    sources=state.get("sources") or None)

    return {
        "question": new_q,           # 用改写后的问题继续检索
        "retrieved_docs": docs,
        "retry_count": round_no,
    }


def web_search_node(state: AgentState) -> AgentState:
    """Tavily 网络搜索回退（仅发送用户问题）。"""
    from web_search import web_search
    results = web_search(state["question"])
    return {"web_results": results, "web_search_needed": True}


def generate_node(state: AgentState) -> AgentState:
    """基于精选证据（本地 + 网络）生成回答，输出 JSON。"""
    question = state["question"]
    local = state.get("selected_docs", [])
    web = state.get("web_results", []) or []

    # 组装证据上下文
    evidence_blocks = []
    citations: List[Dict] = []
    for i, d in enumerate(local, 1):
        evidence_blocks.append(f"[本地{i}] 来源：{d['source']}（{d.get('date','无日期')}）\n{d['text']}")
        citations.append({
            "source": d["source"],
            "type": "local",
            "source_type": d.get("source_type", ""),
            "date": d.get("date", ""),
            "chunk": d["text"][:300],
            "score": d.get("score", 0),
        })
    for i, w in enumerate(web, 1):
        n = len(local) + i
        evidence_blocks.append(f"[网络{n}] 来源：{w['url']}\n标题：{w['title']}\n摘要：{w['snippet']}")
        citations.append({
            "source": w["url"],
            "type": "web",
            "source_type": "web",
            "date": "",
            "title": w.get("title", ""),
            "chunk": w.get("snippet", "")[:300],
        })

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "（没有可用证据）"

    # 只有 1 条本地证据且无网络补充时，明确告诉模型「这是唯一依据」，
    # 让它基于这一条正常作答、并如实交代依据单薄，而不是笼统回「没有足够记录」。
    sole_note = ""
    if len(local) == 1 and not web:
        sole_note = (
            "注意：本次只有 1 条本地记录与问题相关，它是唯一依据。"
            "请基于它认真作答（用户要的是帮他判断，不是复述原文），"
            "并在回答里自然地说明「你的记录里只提到这一点」，"
            "不要因为只有一条就直接回绝。\n"
        )

    # 【个人原则·硬约束】独立于检索：不判断、不召回，每次生成都无条件注入。
    # 与「证据」语义不同——证据供参考、可被引用；原则是底线、不可违背。
    try:
        from principles import principles_for_prompt
        principle_text = principles_for_prompt()
    except Exception as e:  # noqa: BLE001 — 原则读取失败不应阻断回答
        print(f"[warn] 读取个人原则失败，本次跳过注入：{e}", file=sys.stderr)
        principle_text = ""

    principle_block = ""
    if principle_text:
        principle_block = (
            "【用户已声明的个人原则 · 必须遵守】\n"
            "以下是用户亲自写下的底线。**任何回答都不得与之相悖**；\n"
            "若证据结论或你的建议与之冲突，以原则为准，并明确指出冲突之处。\n"
            "这些是约束、不是参考资料——不要把它们当作证据来引用。\n"
            "————\n"
            f"{principle_text}\n"
            "————\n"
        )

    # 本地 0 条、只有网络结果时：强制模型先声明「你的记录里没有」。
    # 否则用户会把网络结论误当成「AI 看了我的记录得出的」，这是最危险的误导。
    web_only_note = ""
    if not local and web:
        web_only_note = (
            "注意：本次**本地知识库（用户的 docs/）没有检索到任何相关记录**，"
            "可用证据**全部来自网络**。回答开头必须先明确说明"
            "「你的记录里没有相关内容」，再基于网络信息作答；"
            "严禁让用户误以为这些结论来自他自己的记录。\n"
        )

    system = (
        "你是「问己」，一个基于用户个人知识库的决策助手。\n"
        + principle_block
        + web_only_note
        + sole_note
        + "回答必须严格基于提供的证据，不得编造。若证据不足，如实说明并结合网络信息谨慎回答。\n"
        "本地证据与网络证据要区分对待：本地证据代表用户本人的真实记录。\n"
        "输出必须是 JSON，字段：\n"
        '  answer（字符串，回答正文）、\n'
        '  citations（数组，每条含 source、type[local|web]、score[仅本地]）、\n'
        '  confidence（high|medium|low）\n'
    )
    user = f"问题：{question}\n\n可用证据：\n{evidence_text}"

    try:
        raw = _chat([{"role": "system", "content": system},
                     {"role": "user", "content": user}], json_mode=True)
        data = json.loads(raw)
        answer = data.get("answer", "")
        conf = data.get("confidence", "medium")
    except Exception as e:
        print(f"[warn] 生成失败：{e}", file=sys.stderr)
        answer = "（生成回答时出错，请稍后重试）"
        conf = "low"

    return {
        "answer": answer,
        "citations": citations,
        "evidence_count": len(local) + len(web),
        "confidence": conf,
    }


# ------------------------------------------------------------------
# 3. 条件边（对应计划 5.3 的回退逻辑）
# ------------------------------------------------------------------


def route_after_intent(state: AgentState) -> str:
    """需要检索 → retrieve；否则直接生成。

    例外：用户用 @ 指定了范围（sources 非空）时**强制检索**——
    既然他特意圈定了文件，就说明他希望答案基于这些文件，不能再判成闲聊跳过。
    """
    if state.get("sources"):
        return "retrieve"
    return "retrieve" if state.get("need_retrieval", True) else "generate"


def route_after_rerank(state: AgentState) -> str:
    """证据 >= 1 条（已过重排阈值）→ 生成；否则改写重试（最多 2 轮）→ 网络回退。

    ⚠️ 2026-10-03 调整：由「>= 2 条」放宽为「>= 1 条」。
    原因：重排阈值 7 是**绝对分**，而个人笔记往往是零散的一句话，
    模型对「开放性建议型」问题会诚实地打 6 分左右，
    于是全库常常只有 1 条够 7 分（甚至 0 条）——
    老规则下这 1 条会被丢掉、连锁触发「改写 → 网络回退 → 没有足够记录」，
    表现为「明明 docs 里有相关内容却说没有」。
    只要过了阈值就说明模型认为它与问题相关，1 条也应当交给生成模型判断，
    由生成阶段在提示里注明「唯一依据」，而不是流程层直接否掉。

    两种情况下不做网络回退（直接进入生成）：
      1) 网络回退开关关闭（use_web_search=False）；
      2) 用户在输入框用 @ 限定了检索范围（sources 非空）——
         此时属于「严格限定」，答案必须来自所选文件，不能拿网络内容充数。
    """
    if state.get("evidence_count", 0) >= 1:
        return "generate"
    if state.get("retry_count", 0) < 2:
        return "rewrite_query"
    if state.get("sources"):                      # @ 限定了范围 → 不回退网络
        return "generate"
    return "web_search" if state.get("use_web_search", True) else "generate"


def route_after_rewrite(state: AgentState) -> str:
    """改写后重新打分判断：够了就生成，否则继续（重试轮数用尽则网络回退）。"""
    return "jev_rerank"


def route_after_web(state: AgentState) -> str:
    return "generate"


# ------------------------------------------------------------------
# 4. 组装图
# ------------------------------------------------------------------

_compiled = None


def build_graph():
    """构建并编译 LangGraph 工作流（单例，避免重复编译）。"""
    global _compiled
    if _compiled is not None:
        return _compiled

    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve", retrieve_node)
    g.add_node("jev_rerank", jev_rerank_node)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges("classify_intent", route_after_intent,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "jev_rerank")
    g.add_conditional_edges("jev_rerank", route_after_rerank,
                            {"generate": "generate",
                             "rewrite_query": "rewrite_query",
                             "web_search": "web_search"})
    g.add_conditional_edges("rewrite_query", route_after_rewrite,
                            {"jev_rerank": "jev_rerank"})
    g.add_conditional_edges("web_search", route_after_web, {"generate": "generate"})
    g.add_edge("generate", END)

    _compiled = g.compile()
    return _compiled


# ------------------------------------------------------------------
# 5. 对外唯一入口（对应计划 7.2 的函数签名与返回格式）
# ------------------------------------------------------------------


def run_agent(
    question: str,
    history: Optional[List[Dict]] = None,
    top_k: int = TOP_K,
    jev_threshold: float = JEV_THRESHOLD,
    use_jev: bool = True,
    use_web_search: bool = True,
    sources: Optional[List[str]] = None,
) -> Dict:
    """UI 层唯一调用入口。

    参数：
        sources: @ 引用的检索范围（文件或文件夹路径列表，如
                 ["docs/日记（9月）.md", "日记补充（8月）"]）。
                 None/空 = 全库检索（保持原行为）。
                 给定时**只在所选范围内检索**，且不再触发网络回退。

    返回：{answer, citations, need_retrieval, evidence_count, web_search_triggered,
          scoped_sources, local_count, web_count}
          （新增字段均为**追加**，旧调用不受影响）
    """
    app = build_graph()
    init: AgentState = {
        "question": question,
        "history": history or [],
        "top_k": top_k,
        "jev_threshold": jev_threshold,
        "use_jev": use_jev,
        "use_web_search": use_web_search,
        "sources": [s for s in (sources or []) if s],   # 过滤空串
        "retry_count": 0,
        "web_search_needed": False,
        "web_results": [],
        "selected_docs": [],
    }
    final = app.invoke(init)
    citations = final.get("citations", [])
    # 本地 / 网络各自条数：前端据此标注「本次参考范围」与来源构成
    local_count = sum(1 for c in citations if c.get("type") == "local")
    web_count = sum(1 for c in citations if c.get("type") == "web")
    return {
        "answer": final.get("answer", ""),
        "citations": citations,
        "need_retrieval": final.get("need_retrieval", False),
        "evidence_count": final.get("evidence_count", 0),
        "web_search_triggered": bool(final.get("web_search_needed", False)),
        "scoped_sources": init["sources"],              # 回传本次限定范围，便于前端展示
        "local_count": local_count,                     # 本地证据条数
        "web_count": web_count,                         # 网络证据条数
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "我最近在焦虑什么？"
    result = run_agent(q)
    print(f"\n问题：{q}\n")
    print(f"需检索：{result['need_retrieval']}  证据数：{result['evidence_count']}  "
          f"网络回退：{result['web_search_triggered']}\n")
    print("回答：\n", result["answer"])
    print("\n引用来源：")
    for i, c in enumerate(result["citations"], 1):
        tag = "网络" if c["type"] == "web" else "本地"
        print(f"  [{i}] ({tag}) {c['source']}  {c.get('score','')}")
