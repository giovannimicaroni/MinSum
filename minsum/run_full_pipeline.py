"""Run the full MinSum pipeline (01 -> 05) from a single YAML config.

Each step runs as its own subprocess (python -m minsum.Experiments.run_XX), so
TensorFlow and torch never share a process and a step can use a different
interpreter. The runner wires the hand-off names between steps and validates
the whole config against the step scripts before anything runs.

Examples:
  python -m minsum.run_full_pipeline
  python -m minsum.run_full_pipeline --config my_run.yaml --dry-run
  python -m minsum.run_full_pipeline --steps clustering,postprocessing
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from minsum.utils_full_pipeline import (
    DEFAULT_CONFIG,
    STEP_ORDER,
    ConfigError,
    load_config,
    plan_steps,
    resolve_steps,
    subprocess_env,
    validate_options,
    write_run_record,
)


def main() -> None:
    cli = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cli.add_argument(
        "--steps",
        default=None,
        help=f"Comma-separated steps overriding run.steps. Choices: {','.join(STEP_ORDER)}.",
    )
    cli.add_argument("--dry-run", action="store_true", help="Validate and print commands only.")
    args = cli.parse_args()
    # Keep runner headers in order with step output when stdout is a file (e.g. SLURM logs).
    sys.stdout.reconfigure(line_buffering=True)

    try:
        config = load_config(args.config)
        planned, handoff = plan_steps(config, resolve_steps(config, args.steps))
        env = subprocess_env()
        validate_options(planned, env)
    except ConfigError as exc:
        cli.exit(2, f"Config error ({args.config}):\n{exc}\n")

    print(f"Config: {args.config}")
    for key, value in handoff.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        for index, step in enumerate(planned, start=1):
            print(f"\n[{index}/{len(planned)}] {step.name}\n  {shlex.join(step.command())}")
        return

    record_dir = write_run_record(config, planned, handoff)
    print(f"Run record: {record_dir}")
    pipeline_start = time.time()
    for index, step in enumerate(planned, start=1):
        command = step.command()
        print(f"\n{'=' * 80}\n[{index}/{len(planned)}] {step.name}\n  {shlex.join(command)}\n{'=' * 80}")
        step_start = time.time()
        result = subprocess.run(command, env=env)
        elapsed = time.time() - step_start
        if result.returncode != 0:
            print(
                f"\nStep '{step.name}' failed with exit code {result.returncode} "
                f"after {elapsed:.1f}s:\n  {shlex.join(command)}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
        print(f"[{index}/{len(planned)}] {step.name} finished in {elapsed:.1f}s")
    print(f"\nFull pipeline finished in {time.time() - pipeline_start:.1f}s")


if __name__ == "__main__":
    main()
