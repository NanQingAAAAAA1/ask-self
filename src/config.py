"""统一配置与共享客户端。

所有模块（ingest / retrieve / deepseek_rerank / web_search / agent）都从这里读取
路径、模型名与 API Key，避免各处重复读取 .env。

路径分两类（打包成 exe 时尤其重要）：
  * BUNDLE_DIR —— **只读资源**，随程序一起分发（frontend/ 前端页面）。
                  打包后指向 PyInstaller 的解压目录（临时、只读）。
  * ROOT       —— **可写数据**（docs/ chroma_db/ memory/ .env）。
                  打包后指向 **exe 所在目录**，这样数据不会随临时目录被清掉，
                  用户也能直接管理自己的资料。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SRC_DIR = Path(__file__).resolve().parent

# 是否运行在 PyInstaller 打包出的可执行文件里
FROZEN = bool(getattr(sys, "frozen", False))

# 只读资源根：打包后 = 解压临时目录（sys._MEIPASS）；开发时 = 项目根
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", _SRC_DIR.parent))

# 可写数据根：打包后 = exe 同级目录；开发时 = 项目根
ROOT = Path(sys.executable).resolve().parent if FROZEN else _SRC_DIR.parent

load_dotenv(ROOT / ".env")

# ---------- 路径 ----------
DOCS_DIR = ROOT / "docs"                 # 所有 Markdown 材料统一放这里
CHROMA_DIR = ROOT / "chroma_db"          # ChromaDB 持久化目录
COLLECTION_NAME = "askself"              # 向量库集合名

# ---------- 个人原则（硬约束）----------
# 独立于 docs/ 的单独文件：**不进向量库、不参与检索**，
# 而是每次生成回答时无条件注入 system prompt 顶部，作为不得违背的底线。
# 与「证据」语义不同——证据按需召回、可被引用；原则是常驻约束、不可违背。
PRINCIPLES_PATH = ROOT / "memory" / "principles.md"

# ---------- 日记分析 / 导出 / 聊天记录（全部本地可写，不进向量库）----------
ANALYSIS_DIR = ROOT / "analysis_cache"   # 单篇日记卡缓存（按「内容指纹+prompt版本」增量）
REPORTS_DIR = ROOT / "reports"           # 周期总结 Markdown（切勿放进 docs/，否则会被增量录入）
EXPORT_DIR = ROOT / "exports"            # 日记/聊天导出文件（ZIP 临时文件等）
CHAT_HISTORY_DIR = ROOT / "chat_history"  # 聊天记录 JSONL（按月分文件，append-only）

# ---------- 切块参数 ----------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_LEN = 50                       # 碎片过滤：小于该字符数的块丢弃

# ---------- 检索参数 ----------
TOP_K = 20
JEV_THRESHOLD = 7                        # 证据重排「绝对」阈值（DeepSeek 打分）
JEV_THRESHOLD_FLOOR = 6                  # 【相对阈值下限】有效阈值 = min(JEV_THRESHOLD,
                                         #   max(JEV_THRESHOLD_FLOOR, 最高分 - JEV_THRESHOLD_GAP))
                                         #   设 0 关闭相对阈值 → 退化为纯绝对阈值。
                                         #   2026-10-03 实测（8 题对照）：相对阈值让「重排后命中」
                                         #   7/8→8/8、零证据 1/8→0/8，且对打分抖动更稳（详见记忆）。
JEV_THRESHOLD_GAP = 2                    # 【相对阈值】最高分往下让多少分
MAX_RERANK_ITEMS = 20                    # 单次证据重排最多打分的片段数（控成本）

# ---------- 模型 / API ----------
# 嵌入：阿里云 DashScope（OpenAI 兼容接口）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL",
                               "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 嵌入模型优先级：按额度到期时间排序（先过期的排前面），逐个尝试，失败自动降级到下一个。
# 以下模型代码已在阿里云百炼控制台确认（纯文本向量模型）。
#
# ⚠️ 维度对齐提醒：两个模型的向量维度可能不同（如 1024 / 768），而 ChromaDB
#    collection 的维度在首次写入时即固定。若降级使用了维度不同的模型，必须
#    删除 chroma_db/ 或重新建库（ingest.py 默认 reset=True 会重建 collection），
#    否则写入/查询会因维度不一致报错。
DASHSCOPE_EMBED_MODELS = [
    os.getenv("DASHSCOPE_EMBED_MODEL_1", "qwen3.7-text-embedding"),        # 第一优先：优先消耗快过期额度
    os.getenv("DASHSCOPE_EMBED_MODEL_2", "qwen3.7-text-embedding-flash"),  # 第二优先：兜底
]
# qwen3.7-text-rerank 属于 rerank 类模型（输入 query + documents 返回相关性分数，
# 不产出向量），不是嵌入模型，故不列入。若确认其可作嵌入用，加入上面的列表即可。
DASHSCOPE_EMBED_MODEL_SKIPPED = os.getenv("DASHSCOPE_EMBED_MODEL_3", "qwen3.7-text-rerank")

# 生成 / 意图判断 / 查询改写 / 证据重排：DeepSeek（OpenAI 兼容）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
WEB_TIMEOUT = float(os.getenv("WEB_TIMEOUT", "5"))   # 网络搜索超时（秒）

# ---------- 文件名前缀 → source_type ----------
PREFIX_TYPE = {
    "diary-": "diary",
    "course-": "course",
    "work-": "work",
    "study-": "study",
}
DEFAULT_SOURCE_TYPE = "docs"
