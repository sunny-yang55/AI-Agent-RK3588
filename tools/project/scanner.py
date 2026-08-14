"""
Project Scanner

负责:
- 项目目录扫描
- 文件统计
- 路径管理
"""

from pathlib import Path

DEFAULT_IGNORE = {
    ".git",
    "__pycache__",
    # Python virtual environment
    "venv",
    "venv310",
    ".venv",
    # Python cache
    ".pytest_cache",
    # package directories
    "site-packages",
    # Node
    "node_modules",
    # project logs
    "logs",
}


class ProjectScanner:

    def __init__(
        self,
        project_path: str,
        ignore=None,
    ):

        self.root = Path(project_path).resolve()

        self.ignore = set(ignore) if ignore else DEFAULT_IGNORE

    def scan(self):

        files = []

        suffixes = {
            ".py",
            ".md",
            ".json",
            ".yaml",
            ".yml",
        }

        for path in self.root.rglob("*"):

            # 文件夹提前过滤
            if any(part in self.ignore for part in path.parts):
                continue

            if not path.is_file():
                continue

            if path.suffix.lower() in suffixes:

                files.append(path)

        return {
            "root": str(self.root),
            "files": files,
            "count": len(files),
        }

    def statistics(self, files):

        result = {}

        for f in files:

            ext = f.suffix

            result[ext] = result.get(ext, 0) + 1

        return result
