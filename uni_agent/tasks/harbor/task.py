"""Run local Harbor tasks through the Harbor CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from uni_agent.agents import AgentConfig

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import task_result_from_harbor_trial

logger = logging.getLogger(__name__)


class HarborAgentConfig(AgentConfig):
    """Harbor agent name and model endpoint."""

    name: str = Field(default="", description="Harbor built-in agent name or custom agent import path.")


class HarborTaskConfig(TaskConfig):
    """Configuration for one evaluation-only Harbor CLI trial."""

    name: str = "harbor"
    sandbox: str = Field(default="docker", description="Harbor environment backend.")
    agent: HarborAgentConfig = Field(default_factory=HarborAgentConfig)
    timeout_multiplier: float = Field(default=1.0, gt=0)
    trials_dir: Path = Field(default=Path("/tmp/uni_agent_harbor_trials"))
    run_oracle_solution: bool = Field(
        default=False,
        description="Run Harbor's oracle agent instead of the configured agent.",
    )

    @field_validator("agent", mode="before")
    @classmethod
    def _resolve_agent(cls, value: Any) -> Any:
        return value

    @model_validator(mode="after")
    def _validate_local_trial(self) -> HarborTaskConfig:
        instance_id = self.metadata.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("HarborTask metadata requires a non-empty instance_id")

        raw_task_path = self.metadata.get("task_path")
        if not isinstance(raw_task_path, str) or not raw_task_path:
            raise ValueError("HarborTask metadata requires an absolute task_path")
        task_path = Path(raw_task_path).expanduser()
        if not task_path.is_absolute():
            raise ValueError(f"Harbor task_path must be absolute, got {raw_task_path!r}")
        if not task_path.is_dir() or not (task_path / "task.toml").is_file():
            raise ValueError(f"Harbor task_path is not a task directory: {task_path}")

        self.trials_dir = self.trials_dir.expanduser()
        if not self.trials_dir.is_absolute():
            raise ValueError(f"Harbor trials_dir must be absolute, got {self.trials_dir}")
        if not self.sandbox.strip():
            raise ValueError("Harbor sandbox must be non-empty")
        if not self.run_oracle_solution and not self.agent.name.strip():
            raise ValueError("Harbor agent.name must be non-empty unless run_oracle_solution is enabled")
        return self


def build_harbor_trial_command(config: HarborTaskConfig, *, trial_name: str) -> list[str]:
    """Build the argv for one collision-safe Harbor trial."""
    task_path = str(config.metadata["task_path"])
    agent_name = "oracle" if config.run_oracle_solution else config.agent.name
    command = [
        "harbor",
        "trial",
        "start",
        "--path",
        task_path,
        "--trial-name",
        trial_name,
        "--trials-dir",
        str(config.trials_dir),
        "--agent",
        agent_name,
        "--env",
        config.sandbox,
        "--timeout-multiplier",
        f"{config.timeout_multiplier:g}",
    ]

    if not config.run_oracle_solution and config.agent.model.model_name:
        command.extend(["--model", config.agent.model.model_name])
    return command


def build_harbor_process_env(config: HarborTaskConfig) -> dict[str, str] | None:
    """Map the runtime model endpoint into Harbor's host process environment."""
    if config.run_oracle_solution:
        return None

    process_env: dict[str, str] = {}
    if config.agent.model.base_url:
        process_env["OPENAI_BASE_URL"] = config.agent.model.base_url
    if config.agent.model.api_key:
        process_env["OPENAI_API_KEY"] = config.agent.model.api_key
    return process_env or None


@dataclass(frozen=True)
class HarborCLIResult:
    exit_code: int
    stdout: str
    stderr: str


async def run_harbor_cli(command: list[str], *, env: dict[str, str] | None = None) -> HarborCLIResult:
    """Run Harbor directly on the host and capture its terminal output."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env} if env else None,
    )
    try:
        stdout, stderr = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    return HarborCLIResult(
        exit_code=process.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


@register_task("harbor")
class HarborTask(Task):
    name = "harbor"
    config_model = HarborTaskConfig

    async def run(self) -> TaskResult:
        config: HarborTaskConfig = self.config  # type: ignore[assignment]
        instance_id = str(config.metadata["instance_id"])
        trial_name = f"uni-agent-{uuid4().hex}"
        trial_dir = config.trials_dir / trial_name
        command = build_harbor_trial_command(config, trial_name=trial_name)
        process_env = build_harbor_process_env(config)

        try:
            await asyncio.to_thread(config.trials_dir.mkdir, parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"failed to create Harbor trials directory {config.trials_dir}: {exc}") from exc

        logger.info(
            "starting Harbor trial (instance_id=%s, agent=%s, model=%s, environment=%s, trial=%s)",
            instance_id,
            "oracle" if config.run_oracle_solution else config.agent.name,
            None if config.run_oracle_solution else config.agent.model.model_name,
            config.sandbox,
            trial_name,
        )
        started = time.perf_counter()
        try:
            response = await run_harbor_cli(command, env=process_env)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Harbor CLI executable was not found; install Harbor 0.20 or later and ensure `harbor` is on PATH"
            ) from exc
        elapsed = time.perf_counter() - started

        result_path = trial_dir / "result.json"
        try:
            result_bytes = await asyncio.to_thread(result_path.read_bytes)
        except FileNotFoundError as exc:
            detail = (response.stderr or response.stdout).strip()[-2000:]
            message = f"Harbor trial did not write {result_path}"
            if detail:
                message += f": {detail}"
            raise RuntimeError(message) from exc

        try:
            payload = json.loads(result_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Harbor wrote an invalid trial result at {result_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Harbor trial result at {result_path} is not a JSON object")

        result = task_result_from_harbor_trial(
            payload,
            trial_dir=trial_dir,
            cli_exit_code=response.exit_code,
            stdout=response.stdout,
            stderr=response.stderr,
            elapsed=elapsed,
        )
        info = result.extra_info or {}
        if not info.get("eval_completed"):
            logger.warning(
                "Harbor trial incomplete for %s: %s",
                instance_id,
                (info.get("eval_report") or {}).get("error"),
            )
        logger.info(
            "Harbor trial done: instance_id=%s reward=%.3f resolved=%s elapsed=%.1fs",
            instance_id,
            float(result.reward),
            info.get("resolved"),
            elapsed,
        )
        return result
