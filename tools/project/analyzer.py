"""
Python AST Analyzer
"""

import ast
from pathlib import Path


class PythonAnalyzer:

    def analyze_file(self, file_path: Path):

        result = {
            "file": str(file_path),
            "classes": [],
            "functions": [],
            "imports": [],
        }

        try:

            source = file_path.read_text(encoding="utf-8")

            tree = ast.parse(source)

        except Exception as e:

            result["error"] = str(e)

            return result

        for node in ast.walk(tree):

            if isinstance(node, ast.ClassDef):

                result["classes"].append(node.name)

            elif isinstance(node, ast.FunctionDef):

                result["functions"].append(node.name)

            elif isinstance(node, ast.Import):

                for n in node.names:

                    result["imports"].append(n.name)

            elif isinstance(node, ast.ImportFrom):

                if node.module:

                    result["imports"].append(node.module)

        return result

    def analyze_project(self, files):

        reports = []

        for f in files:

            if f.suffix == ".py":

                reports.append(self.analyze_file(f))

        return reports
