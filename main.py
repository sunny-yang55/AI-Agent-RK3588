"""
AI-Agent — Production Entry Point.

A production-ready entry point for initializing and running the Agent package.
Supports Windows (development) and Ubuntu/RK3588 (deployment) environments.
"""

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import argparse
import logging
import os
import platform
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from runtime import RuntimeManager

# ---------------------------------------------------------------------------
# Project path setup — ensure the project root is on sys.path so that the
# agent package can be imported regardless of the working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Agent package imports
# ---------------------------------------------------------------------------
from agent import Agent, MemoryManager, ToolRegistry  # noqa: E402
from tools.common.tool_metadata import ToolMetadata
from tools.llm.adapter import LLMAdapter

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "agent.log"

# Canonical logger for this module
_logger = logging.getLogger("main")

# Format strings: one concise (console) and one verbose (file)
_CONSOLE_FORMAT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_FILE_FORMAT = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)-18s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def configure_logging(*, verbose: bool = False) -> None:
    """Set up console and file logging for the entire application.

    Args:
        verbose: When True, set the root logger and all agent loggers to
                 DEBUG level.  Otherwise INFO is used.
    """
    level = logging.DEBUG if verbose else logging.WARNING

    # ----- Root logger -----
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers that might have been added by libraries or repeated
    # invocations (idempotency).
    root.handlers.clear()

    # ----- Console handler (stderr, for real-time feedback) -----
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(_CONSOLE_FORMAT)
    root.addHandler(console)

    # ----- File handler (persistent, rotating-like via daily logs is a
    #       possible future enhancement; for now a single file is fine) -----
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(_LOG_FILE), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_FILE_FORMAT)
        root.addHandler(file_handler)
    except OSError as exc:
        # Non-fatal: the agent can still run with console logging only.
        root.warning("Could not create file log handler: %s", exc)

    # Silence noisy third-party loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _logger.info(
        "Logging configured (level=%s, file=%s)", logging.getLevelName(level), _LOG_FILE
    )


# ---------------------------------------------------------------------------
# Signal handling — graceful shutdown on SIGINT / SIGTERM
# ---------------------------------------------------------------------------
_agent_instance: Agent | None = None


def _handle_shutdown(signum: int, _frame: Any) -> None:
    """Callback for SIGINT / SIGTERM: stop the agent and exit cleanly."""
    name = signal.Signals(signum).name
    _logger.warning("Received %s — shutting down gracefully...", name)
    if _agent_instance is not None:
        _agent_instance.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)

# ---------------------------------------------------------------------------
# Tool handler implementations
# ---------------------------------------------------------------------------
# The RuleBasedPlanner matches the keyword "analyze" in the goal and
# creates three steps: gather_context → identify_requirements →
# propose_approach.  We register concrete handlers for each action below.
# These handlers use the current project directory as their working context
# and return structured dictionaries that the executor records as results.


def _scan_directory(base: Path, max_depth: int = 2) -> dict[str, Any]:
    """Recursively scan *base* up to *max_depth* and return summary stats.

    Used by several tool handlers to gather filesystem context.
    """
    stats: dict[str, Any] = {
        "root": str(base),
        "max_depth": max_depth,
        "directories": 0,
        "files": 0,
    }
    top_items: list[str] = []

    try:
        for entry in sorted(base.iterdir()):
            name = entry.name
            # Skip hidden / .git directories to keep output clean
            if name.startswith("."):
                continue
            if entry.is_dir():
                top_items.append(f"[DIR]  {name}/")
            else:
                top_items.append(f"[FILE] {name}")
    except PermissionError:
        _logger.warning("Permission denied scanning: %s", base)

    # Walk the tree to count items
    for root_str, dirs, files in os.walk(str(base)):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        depth = root_str[len(str(base)) :].count(os.sep)
        if depth >= max_depth:
            dirs.clear()
        stats["directories"] += len(dirs)
        stats["files"] += len(files)

    stats["top_level"] = top_items[:50]  # cap for readability
    return stats


def _handler_gather_context(**params: Any) -> dict[str, Any]:
    """Gather relevant context and information about the project."""
    _logger.debug("gather_context handler invoked with params=%s", params)

    root = _PROJECT_ROOT
    result: dict[str, Any] = {
        "action": "gather_context",
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": str(root),
        "directory_scan": _scan_directory(root),
    }

    # Check for key project files
    key_files = ["README.md", "main.py", ".gitignore"]
    result["key_files_present"] = {f: (root / f).exists() for f in key_files}

    # Check agent package
    agent_dir = root / "agent"
    if agent_dir.is_dir():
        result["agent_package"] = {
            "path": str(agent_dir),
            "files": sorted(
                f.name for f in agent_dir.iterdir() if f.is_file() and f.suffix == ".py"
            ),
        }
    else:
        result["agent_package"] = "MISSING"

    _logger.info(
        "gather_context: scanned %d files across %d directories",
        result["directory_scan"]["files"],
        result["directory_scan"]["directories"],
    )
    return result


def _handler_identify_requirements(**params: Any) -> dict[str, Any]:
    """Identify key requirements and constraints of the project."""
    _logger.debug("identify_requirements handler invoked with params=%s", params)

    root = _PROJECT_ROOT
    result: dict[str, Any] = {"action": "identify_requirements"}

    # Read the README for high-level requirements
    readme_path = root / "README.md"
    if readme_path.exists():
        try:
            readme_content = readme_path.read_text(encoding="utf-8")
            readme_lines = readme_content.strip().splitlines()
            result["readme_summary"] = {
                "file": str(readme_path),
                "line_count": len(readme_lines),
                "first_10_lines": readme_lines[:10],
            }
        except OSError as exc:
            result["readme_summary"] = {"error": str(exc)}
    else:
        result["readme_summary"] = "README.md not found"

    # Analyse the agent package for its public API surface
    agent_init = root / "agent" / "__init__.py"
    if agent_init.exists():
        try:
            imports = [
                line.strip()
                for line in agent_init.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("from .")
                or line.strip().startswith("import ")
            ]
            result["agent_public_api"] = imports
        except OSError as exc:
            result["agent_public_api"] = {"error": str(exc)}

    # Identify constraints from platform
    result["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }

    _logger.info(
        "identify_requirements: readme_lines=%d, api_exports=%d",
        (
            result.get("readme_summary", {}).get("line_count", 0)
            if isinstance(result.get("readme_summary"), dict)
            else 0
        ),
        len(result.get("agent_public_api", [])),
    )
    return result


def _handler_propose_approach(**params: Any) -> dict[str, Any]:
    """Propose an approach or solution based on gathered context."""
    _logger.debug("propose_approach handler invoked with params=%s", params)

    root = _PROJECT_ROOT
    result: dict[str, Any] = {
        "action": "propose_approach",
        "timestamp": time.time(),
        "approach": [
            {
                "phase": 1,
                "title": "Core Foundation (Current)",
                "description": (
                    "The agent/ package implements a modular agent framework "
                    "with planning, execution, and memory subsystems. "
                    "The architecture supports extensible tool registration "
                    "and multiple execution modes (SEQUENTIAL, PARALLEL, DAG)."
                ),
            },
            {
                "phase": 2,
                "title": "Tool Ecosystem",
                "description": (
                    "Populate the tools/ directory with concrete tool handlers "
                    "for LLM interaction, robot control, speech processing, and "
                    "vision tasks. Integrate with MCP servers under mcp_servers/."
                ),
            },
            {
                "phase": 3,
                "title": "Configuration & Deployment",
                "description": (
                    "Implement configuration loading from config/settings.json, "
                    "config/models.json, and config/mcp.json.  Create deployment "
                    "scripts under deployment/rk3588/ and deployment/windows/."
                ),
            },
            {
                "phase": 4,
                "title": "Multi-Agent & Integration",
                "description": (
                    "Build multi-agent collaboration on top of the current "
                    "single-agent architecture. Integrate with ROS2 for robot "
                    "control and add real-time speech/vision pipelines."
                ),
            },
        ],
        "recommendation": (
            "Complete the tool ecosystem (Phase 2) next, starting with LLM "
            "tool handlers in tools/llm/, as these are needed by all other "
            "components.  Then implement config loading (Phase 3) to support "
            "flexible deployment across Windows and RK3588."
        ),
    }

    # Include a concise summary of the project structure
    result["project_overview"] = {
        "agent_modules": ["agent.py", "planner.py", "executor.py", "memory.py"],
        "empty_dirs": [
            str(d.relative_to(root))
            for d in [root / "tools", root / "mcp_servers", root / "deployment"]
            if d.is_dir()
        ],
        "config_status": (
            "Empty — needs implementation"
            if all(
                (root / "config" / f).stat().st_size == 0
                for f in ["settings.json", "models.json", "mcp.json"]
                if (root / "config" / f).exists()
            )
            else "Partial"
        ),
    }

    _logger.info(
        "propose_approach: generated %d-phase roadmap", len(result["approach"])
    )
    return result


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------
def init_agent(*, logs_dir: Path | None = None) -> Agent:
    """Create, configure, and return a ready-to-use Agent instance.

    Steps performed:
    1. Instantiate MemoryManager with platform-appropriate paths.
    2. Create a default Agent (which auto-wires Planner + Executor).
    3. Register tool handlers for the "analyze" action pattern.
    4. Return the fully initialized agent.

    Args:
        logs_dir: Optional custom directory for persistent memory files.
                  Defaults to <project_root>/logs.

    Returns:
        A configured Agent ready for `run()` or `run_sync()`.
    """
    _logger.info("-" * 60)
    _logger.info("Initializing AI-Agent ...")

    # ---- 1. Memory subsystem ----
    logs = logs_dir or (_PROJECT_ROOT / "logs")
    try:
        logs.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _logger.warning("Cannot create logs directory %s: %s", logs, exc)

    long_term_path = logs / "long_term_memory.json"
    episodic_path = logs / "episodic_memory.json"

    memory = MemoryManager(
        working_capacity=256,
        short_term_capacity=1024,
        long_term_path=str(long_term_path),
        episodic_path=str(episodic_path),
    )
    # DEBUG
    # print(f"[PERF] Memory init: {time.time()-t_memory:.2f}s")
    _logger.info(
        "  MemoryManager ready (long_term=%s, episodic=%s)",
        long_term_path,
        episodic_path,
    )

    # ---- 2. Agent creation ----
    # The Agent() constructor creates default Planner and Executor internally,
    # wiring them together with the memory manager.

    agent = Agent(memory=memory)
    # print(f"[PERF] Agent init: {time.time()-t_agent:.2f}s")

    # ======================================
    # Register LLM Response Tool
    # ======================================

    llm = LLMAdapter()

    def generate_response(goal: str, **kwargs):

        try:
            response = llm.chat(goal)

            return {"response": response}

        except Exception as e:

            return {"response": f"LLM error: {e}"}

    agent.tools.register(
        ToolMetadata(
            name="generate_response",
            description="Generate natural language response using LLM",
            handler=generate_response,
            parameters={"goal": "string"},
        )
    )

    _logger.info("Registered generate_response tool")
    _logger.info("  Agent instance created (type=%s)", type(agent).__name__)

    # ---- 3. Tool registration ----
    # Register the handlers that the RuleBasedPlanner's "analyze" pattern will
    # look up by action name.  Each handler receives `**step.params` (currently
    # empty dicts) and returns a structured dict.

    _logger.info(
        "  Registered %d tools: %s", len(agent.tools), agent.tools.list_tools()
    )
    # ---- Tool Discovery v0.6.8 ----
    try:

        discovered = agent.tools.discover()

        if isinstance(discovered, dict):

            names = list(discovered.keys())

        else:

            names = [getattr(t, "name", str(t)) for t in discovered]

        _logger.info("Discovered tools (%d): %s", len(names), names)

    except Exception as exc:

        _logger.warning("  Tool discovery failed: %s", exc)

    # ---- 4. Environment report ----
    _logger.info("  Platform : %s", platform.platform())
    _logger.info("  Python   : %s", sys.version.split()[0])
    _logger.info("  CWD      : %s", _PROJECT_ROOT)
    _logger.info("Agent initialization complete.")
    _logger.info("-" * 60)

    global _agent_instance  # noqa: PLW0603
    _agent_instance = agent
    return agent


# ---------------------------------------------------------------------------
# Output helpers — structured printing of results
# ---------------------------------------------------------------------------
def _print_header(title: str, *, char: str = "=") -> None:
    """Print a decorated section header."""
    width = 72
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_initialization_status(agent: Agent) -> None:
    """测试
    Print the agent's initialization status to stdout.
    _print_header("INITIALIZATION STATUS")

    ctx = agent.context
    print(f"  Session ID   : {ctx.session_id}")
    print(f"  Agent State  : {ctx.state.name}")
    print(
        f"  Registered Tools ({len(agent.tools)}): "
        f"{', '.join(tool.name for tool in agent.tools.list_tools())}"
    )

    mem = agent.memory
    print(f"  Working Memory entries : {len(mem.working)}")
    print(f"  Short-Term  entries    : {len(mem.short_term)}")
    print(f"  Long-Term   entries    : {len(mem.long_term)}")
    print(f"  Episodic    entries    : {len(mem.episodic)}")

    # Verify critical paths exist
    agent_dir = _PROJECT_ROOT / "agent"
    print(f"  Agent package           : {'OK' if agent_dir.is_dir() else 'MISSING'}")
    print(f"  Platform                : {platform.platform()}")
    """


def print_plan(agent: Agent) -> None:
    """Print the generated plan details."""
    _print_header("GENERATED PLAN")

    plan = agent.context.current_plan
    if plan is None:
        print("  No plan generated.")
        return

    summary = plan.summary()
    print(f"  Plan ID     : {summary['plan_id']}")
    print(f"  Goal        : {summary['goal']}")
    print(f"  Total Steps : {summary['total_steps']}")
    print(f"  Completed   : {summary['completed']}")
    print(f"  Failed      : {summary['failed']}")
    print(f"  Pending     : {summary['pending']}")

    print(f"\n  Steps:")
    for i, step in enumerate(plan.steps, 1):
        status_mark = {
            "PENDING": "   ",
            "COMPLETED": "[OK]",
            "FAILED": "[FAIL]",
            "IN_PROGRESS": "[..]",
            "SKIPPED": "[--]",
        }.get(step.status.name, "[?]")
        print(
            f"  {i}. {status_mark} {step.step_id:20s} | {step.action:30s} | {step.description}"
        )


def print_execution_result(result: dict[str, Any]) -> None:
    """Print the overall execution result returned by agent.run()."""
    _print_header("EXECUTION RESULT")

    success = result.get("success", False)
    status_text = "SUCCESS" if success else "FAILURE"
    print(f"  Status       : {status_text}")
    print(f"  Goal         : {result.get('goal', 'N/A')}")
    print(f"  Duration     : {result.get('duration_s', 0):.3f} seconds")

    plan_summary = result.get("plan_summary", {})
    if plan_summary:
        print(
            f"  Plan Steps   : {plan_summary.get('total_steps', 0)} total, "
            f"{plan_summary.get('completed', 0)} completed, "
            f"{plan_summary.get('failed', 0)} failed"
        )

    print(f"\n  Step History:")
    for entry in result.get("history", []):
        status = entry.get("status", "UNKNOWN")
        icon = {"COMPLETED": "+", "FAILED": "!"}.get(status, "?")
        print(
            f"    [{icon}] {entry.get('step_id', '?'):20s} | {entry.get('action', '?'):30s} | "
            f"{entry.get('duration_ms', 0):7.1f}ms"
        )

        error = entry.get("error")
        if error:
            print(f"          Error: {error[:120]}")


def print_memory_summary(agent: Agent) -> None:
    """Print a comprehensive summary of the agent's memory subsystems."""
    _print_header("MEMORY SUMMARY")

    mem = agent.memory

    # ---- Working Memory (volatile scratchpad) ----
    print(
        f"\n  [Working Memory]  capacity={mem.working.capacity}, entries={len(mem.working)}"
    )
    working_snapshot = mem.working.snapshot()
    if working_snapshot:
        for key, value in working_snapshot.items():
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"    {key}: {val_str}")
    else:
        print("    (empty)")

    # ---- Short-Term Memory (recent history FIFO) ----
    print(
        f"\n  [Short-Term Memory]  capacity={mem.short_term.capacity}, entries={len(mem.short_term)}"
    )
    recent = mem.short_term.recent(n=10)
    if recent:
        for entry in recent:
            val_str = str(entry.value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"    {entry.key}: {val_str}")
    else:
        print("    (empty)")

    # ---- Long-Term Memory (persistent file-backed) ----
    print(f"\n  [Long-Term Memory]  entries={len(mem.long_term)}")
    lt_snapshot = mem.long_term.snapshot()
    if lt_snapshot:
        for key, value in lt_snapshot.items():
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"    {key}: {val_str}")
    else:
        print("    (empty — populate with agent.store_long_term())")

    # ---- Episodic Memory (full episode traces) ----
    print(f"\n  [Episodic Memory]  entries={len(mem.episodic)}")
    episodes = mem.episodic.recent(n=5)
    if episodes:
        for ep in episodes:
            data = ep.get("data", {})
            print(
                f"    id={ep['id']} | goal={data.get('goal', '?')} | "
                f"success={data.get('success', False)} | "
                f"duration={data.get('duration_s', 0):.2f}s"
            )
    else:
        print("    (empty)")

    # ---- Filesystem paths ----
    print(f"\n  [Storage Paths]")
    print(f"    Long-term  : {mem.long_term._storage_path}")
    print(f"    Episodic   : {mem.episodic._storage_path}")


from tools.speech.speech_manager import SpeechManager


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Production entry point for the AI-Agent.

    Orchestrates: logging setup → agent initialization → demo task execution
    → result presentation → clean shutdown.

    Args:
        argv: Command-line argument list (defaults to sys.argv[1:]).

    Returns:
        0 on success, non-zero on failure.
    """
    # ---- 1. Parse command-line arguments ----
    parser = argparse.ArgumentParser(
        description="AI-Agent — Production Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                              # Run the default demo\n"
            '  python main.py --goal "Analyze the project"  # Custom goal\n'
            "  python main.py --verbose                     # Enable DEBUG logging\n"
        ),
    )
    parser.add_argument(
        "--goal",
        "-g",
        type=str,
        default=None,
        help="Goal / task for the agent to execute.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging for detailed diagnostics.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory for log files and persistent memory (default: <project>/logs).",
    )
    args = parser.parse_args(argv)

    # ---- 2. Configure logging ----
    configure_logging(verbose=args.verbose)
    _logger.info("AI-Agent starting (goal=%r)", args.goal)

    # ---- 3. Initialize the Agent ----
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        agent = init_agent(logs_dir=logs_dir)
    except Exception as exc:
        _logger.critical(
            "Agent initialization failed: %s\n%s", exc, traceback.format_exc()
        )
        print(f"\nFATAL: Could not initialize Agent.\n{exc}", file=sys.stderr)
        return 1

    # ---- 4. Print initialization status ----
    print_initialization_status(agent)

    # ---- Tool Discovery v0.6.8 ----

    try:

        discovered = agent.tools.discover()

        _logger.info(
            "Discovered tools (%d): %s",
            len(discovered),
            [tool["name"] for tool in discovered],
        )

    except Exception as exc:

        _logger.warning("Tool discovery failed: %s", exc)

    # ---- 5. Run the demo task ----
    _logger.info("Running task: %s", args.goal)

    result: dict[str, Any]

    try:

        if args.goal:

            result = agent.run_sync(args.goal)

        else:

            runtime = RuntimeManager(agent)
            # print(f"[PERF] Runtime init: {time.time()-t_runtime:.2f}s")
            runtime.start_voice_loop()

            return 0

    except KeyboardInterrupt:

        _logger.warning("Task interrupted by user.")

        agent.stop()

        print("\nInterrupted by user.")

        return 130
    except Exception as exc:
        _logger.critical("Task execution failed: %s\n%s", exc, traceback.format_exc())
        print(f"\nFATAL: Task failed.\n{exc}", file=sys.stderr)
        return 1

    # ---- 6. Print generated plan ----
    print_plan(agent)

    # ---- 7. Print execution result ----
    print_execution_result(result)

    # ---- 8. Print memory summary ----
    print_memory_summary(agent)

    # ---- 9. Determine exit code ----
    success = result.get("success", False)
    exit_code = 0 if success else 1
    _logger.info("AI-Agent finished (exit_code=%d)", exit_code)
    print(f"\nExit code: {exit_code}")
    return exit_code


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
