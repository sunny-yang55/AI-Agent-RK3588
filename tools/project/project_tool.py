"""
Project Tool v2Agent调用入口
"""

import time
from pathlib import Path

from tools.common.tool import tool

# 项目分析缓存
_PROJECT_CACHE = {}

from .analyzer import PythonAnalyzer
from .report import MarkdownReporter
from .scanner import ProjectScanner


@tool(description="扫描项目目录结构")
def read_project(project_path="."):

    start = time.time()

    path = str(Path(project_path).resolve())

    if path in _PROJECT_CACHE:
        result = _PROJECT_CACHE[path]["scan"]

    else:
        scanner = ProjectScanner(path)

        result = scanner.scan()

        _PROJECT_CACHE[path] = {"scan": result}

    print("read_project cost:", round(time.time() - start, 3), "s")

    return {
        "root": result["root"],
        "files": [str(x) for x in result["files"]],
        "count": result["count"],
    }


@tool(description="AST分析Python代码")
def analyze_code(project_path="."):

    start = time.time()

    path = str(Path(project_path).resolve())

    if path not in _PROJECT_CACHE:

        read_project(path)

    if "analysis" in _PROJECT_CACHE[path]:

        result = _PROJECT_CACHE[path]["analysis"]

        print("analyze_code(cache) cost:", round(time.time() - start, 3), "s")

        return result

    scan = _PROJECT_CACHE[path]["scan"]

    analyzer = PythonAnalyzer()

    result = analyzer.analyze_project(scan["files"])

    _PROJECT_CACHE[path]["analysis"] = result

    print("analyze_code cost:", round(time.time() - start, 3), "s")

    return result


@tool(description="生成Markdown项目分析报告")
def generate_report(project_path=".", output="project_report.md"):

    start = time.time()

    path = str(Path(project_path).resolve())

    if path not in _PROJECT_CACHE:

        read_project(path)

    scan = _PROJECT_CACHE[path]["scan"]

    scanner = ProjectScanner(path)

    stats = scanner.statistics(scan["files"])

    analysis = analyze_code(path)

    reporter = MarkdownReporter()

    md = reporter.generate(scan, stats, analysis)

    Path(output).write_text(md, encoding="utf-8")

    print("generate_report cost:", round(time.time() - start, 3), "s")

    return {"status": "success", "report": output, "size": len(md)}


# print("Project Tool v2 loaded")
