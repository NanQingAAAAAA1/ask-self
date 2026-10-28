"""Tavily 网络搜索回退模块。

仅在本地知识库证据不足时触发。**只把用户问题本身**作为查询词发给 Tavily，
不包含任何本地日记内容。返回结构化摘要，直接可用作 RAG 证据。
用法（自测）：
    python src/web_search.py "最近有什么新的 AI 论文"
"""
import sys
from typing import Dict, List

from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, WEB_TIMEOUT


def web_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> List[Dict]:
    """调用 Tavily 搜索，返回 [{title, url, domain, snippet, type:'web'}]。

    失败或超时返回空列表（调用方据此降级为「仅本地证据生成」）。
    """
    if not TAVILY_API_KEY:
        print("[warn] 缺少 TAVILY_API_KEY，跳过网络搜索", file=sys.stderr)
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        print("[warn] 未安装 tavily-python，跳过网络搜索", file=sys.stderr)
        return []

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(
            query=query,                 # 仅发送问题本身
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            timeout=WEB_TIMEOUT,
        )
    except Exception as e:
        print(f"[warn] Tavily 搜索失败（降级为仅本地证据）：{e}", file=sys.stderr)
        return []

    results: List[Dict] = []
    for item in resp.get("results", []) or []:
        url = item.get("url", "")
        results.append({
            "title": item.get("title", ""),
            "url": url,
            "domain": _domain_of(url),
            "snippet": item.get("content", "") or item.get("snippet", ""),
            "type": "web",
        })
    return results


def _domain_of(url: str) -> str:
    """从 URL 提取域名，用于 UI 展示。"""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc or url
    except Exception:
        return url


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "最近有什么新的 AI 论文"
    hits = web_search(query)
    print(f"\n查询：{query}")
    print(f"返回 {len(hits)} 条网络结果：\n")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['title']}")
        print(f"    {h['domain']}  {h['url']}")
        print(f"    {h['snippet'][:120]}…\n")


if __name__ == "__main__":
    main()
