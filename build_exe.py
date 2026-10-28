"""一键打包：问己 AskSelf → Windows exe。

为什么用脚本而不是写死命令：本项目有几个**必须处理**的打包细节，
散在命令行里容易漏。脚本把它们固化下来：

  1. 资源 vs 数据分离：frontend/ .env.example 打进程序（只读资源），
     docs/ chroma_db/ memory/ .env 运行时在 **exe 同级目录**创建（可写数据）。
     —— 这一步靠 `src/config.py` 的 BUNDLE_DIR / ROOT 实现，打包前请勿改动。
  2. uvicorn 的 loop/protocol/lifespan 是动态导入，PyInstaller 静态扫描找不到，
     必须显式 --hidden-import，否则启动即崩。
  3. chromadb 带数据文件，需整包收集（--collect-all）。
  4. **排除 onnxruntime**：本项目写入/查询**总是显式传向量**，ChromaDB 默认的
     ONNX 嵌入永远不被调用（已实测 import chromadb 不会加载它），
     排除后体积少上百 MB。
  5. 扁平导入：源码是 `from agent import ...` 这种同目录导入，
     需要把 src/ 加进搜索路径（--paths src）。

用法（在本项目根目录，用项目自带的 .venv）：

    .venv\\Scripts\\python.exe build_exe.py              # 文件夹版（推荐，启动快）
    .venv\\Scripts\\python.exe build_exe.py --onefile     # 单文件版（便于拷贝分发）
    .venv\\Scripts\\python.exe build_exe.py --console     # 保留控制台窗口（看日志）

产物：
    文件夹版 → dist/AskSelf/AskSelf.exe（连同整个 AskSelf 文件夹一起分发）
    单文件版 → dist/AskSelf.exe
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

# 只读资源：随程序打包
DATAS = [
    ("frontend", "frontend"),      # 前端单文件页面（api.py 实际托管的就是它）
    (".env.example", "."),         # 首次运行据此生成 .env
]

# 动态导入、静态扫不到的模块
HIDDEN = [
    "uvicorn.logging",
    "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    "anyio", "sniffio", "h11", "httpx", "httpcore",
]

# 需要整包收集（含数据文件）的依赖
COLLECT_ALL = ["chromadb"]

# 需要收集全部子模块的依赖（动态导入多）
COLLECT_SUBMODULES = ["langgraph", "langchain_text_splitters", "langchain_core"]

# 明确排除：省体积（onnxruntime 的排除理由见模块 docstring 第 4 条）
EXCLUDES = [
    "onnxruntime",
    "tkinter",
    "matplotlib",
    "pandas",
    "IPython",
    "pytest",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="打包问己 AskSelf 为 Windows exe")
    ap.add_argument("--onefile", action="store_true",
                    help="打成单个 exe（启动时解压到临时目录，略慢）")
    ap.add_argument("--console", action="store_true", default=True,
                    help="保留控制台窗口以便看日志（默认开启）")
    ap.add_argument("--noconsole", action="store_true",
                    help="隐藏控制台窗口（日志不可见，不推荐）")
    ap.add_argument("--clean", action="store_true", default=True, help="构建前清理缓存")
    args = ap.parse_args()

    if shutil.which("pyinstaller") is None:
        # 优先用 .venv 里的 pyinstaller 模块
        try:
            import PyInstaller  # noqa: F401
        except ImportError:
            print("[build] 未检测到 PyInstaller，正在安装到项目 .venv …")
            rc = subprocess.call([PY, "-m", "pip", "install", "pyinstaller"])
            if rc != 0:
                print("[build] PyInstaller 安装失败，请检查网络后重试。")
                return rc

    cmd = [PY, "-m", "PyInstaller", "--noconfirm", "--clean",
           "--name", "AskSelf",
           "--paths", "src",
           "--onefile" if args.onefile else "--onedir"]
    cmd += ["--console"] if not args.noconsole else ["--noconsole"]

    for src, dst in DATAS:
        cmd += ["--add-data", f"{src}{os.pathsep}{dst}"]
    for m in HIDDEN:
        cmd += ["--hidden-import", m]
    for m in COLLECT_ALL:
        cmd += ["--collect-all", m]
    for m in COLLECT_SUBMODULES:
        cmd += ["--collect-submodules", m]
    for m in EXCLUDES:
        cmd += ["--exclude-module", m]

    cmd.append(str(ROOT / "src" / "api.py"))     # 入口

    print("[build] 执行：\n  " + " ".join(cmd) + "\n")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print("[build] 打包失败。")
        return rc

    exe_rel = Path("AskSelf.exe") if args.onefile else Path("AskSelf") / "AskSelf.exe"
    outp = ROOT / "dist" / exe_rel
    print("\n[build] 完成 →", outp)
    print("[build] 提示：docs/、chroma_db/、memory/、.env 会在**首次运行**时"
          "自动在 exe 同级目录创建。")
    if not args.onefile:
        print("[build] 分发方式：整个 dist/AskSelf 文件夹一起拷贝给对方。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
