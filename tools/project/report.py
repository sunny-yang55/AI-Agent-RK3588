"""
Markdown Report Generator
"""

from datetime import datetime


class MarkdownReporter:

    def generate(
        self,
        scan,
        statistics,
        analysis,
    ):

        lines = []

        lines.append("# AI-Agent Project Analysis Report\n")

        lines.append(f"""
## Project
路径:
生成时间:
{datetime.now()}

""")

        lines.append("## File Statistics\n")

        for k, v in statistics.items():

            lines.append(f"- {k}: {v}")

        lines.append("\n## Python Analysis\n")

        for item in analysis:

            lines.append(f"""
### {item["file"]}


Classes:

{item["classes"]}


Functions:

{item["functions"]}


Imports:

{item["imports"]}


""")

        return "\n".join(lines)
