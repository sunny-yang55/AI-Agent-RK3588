"""
File Operation Tools

提供:
- 文件读取
- 文件搜索
"""

from pathlib import Path

from tools.common.tool import tool

PROJECT_ROOT = Path("E:/AI-Agent")


@tool(description="读取指定文件内容", parameters={"path": "文件路径"})
def read_file(path: str):

    file = Path(path)

    if not file.exists():
        return {"success": False, "error": f"File not found: {path}"}

    try:
        content = file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        return {
            "success": True,
            "path": str(file),
            "length": len(content),
            "content": content,
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }


@tool(description="搜索项目代码关键词", parameters={"keyword": "搜索关键词"})
def search_code(keyword):
    """
    Search keyword in python files.
    """

    results = []

    for file in PROJECT_ROOT.rglob("*.py"):

        if ".git" in file.parts:
            continue

        try:
            content = file.read_text(encoding="utf-8")

            if keyword in content:
                results.append(
                    {
                        "file": str(file),
                    }
                )

        except Exception:
            continue

    return {
        "keyword": keyword,
        "results": results,
        "count": len(results),
    }
