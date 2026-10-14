"""Config parsing, validation and step planning for run_full_pipeline."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from minsum.Experiments.run_utils import default_cluster_run_name


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "cfg_full_pipeline.yaml"
STEP_ORDER = (
    "metadata",
    "clustering_embedding",
    "clustering",
    "verification_embedding",
    "postprocessing",
)
OPTION_PATTERN = re.compile(r"(?<![\w-])--[A-Za-z0-9][\w-]*")
# Inputs that must already exist; checked up front instead of failing per image.
INPUT_PATH_OPTIONS = ("dataset_root", "weights_path", "model_root", "shared_val_pairs_path")


@dataclass(frozen=True)
class StepScript:
    module: str
    required: tuple[str, ...] = ()
    # Options parsed as true/false values instead of store_true flags.
    bool_values: tuple[str, ...] = ()


METADATA_SCRIPT = StepScript("run_01_build_metadata", required=("dataset_root",))
CLUSTERING_SCRIPT = StepScript("run_03_fit_clusters")
POSTPROCESSING_SCRIPT = StepScript("run_05_postprocessing")
CLUSTERING_EXTRACTORS = {
    "deepface": StepScript("run_02a_extract_deepface_clustering"),
    "fairface": StepScript("run_02b_extract_fairface_clustering", required=("weights_path",)),
}
VERIFICATION_EXTRACTORS = {
    "arcface": StepScript("run_04a_extract_arcface_verification"),
    "adaface": StepScript("run_04b_extract_adaface_verification", required=("weights_path",)),
    "face_recognition": StepScript(
        "run_04c_extract_face_recognition_verification",
        bool_values=("normalize",),
    ),
    "dface": StepScript(
        "run_04d_extract_dface_verification",
        bool_values=("align", "enforce_detection"),
    ),
    "magface": StepScript("run_04e_extract_magface_verification", required=("weights_path",)),
    "elasticface": StepScript(
        "run_04f_extract_elasticface_verification",
        required=("weights_path",),
    ),
    "transface": StepScript("run_04g_extract_transface_verification", required=("weights_path",)),
}


@dataclass
class PlannedStep:
    name: str
    script: StepScript
    python: str
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> str:
        return f"minsum.Experiments.{self.script.module}"

    def command(self) -> list[str]:
        return [self.python, "-m", self.module, *options_to_argv(self.options, self.script)]


class ConfigError(ValueError):
    pass


def options_to_argv(options: dict[str, Any], script: StepScript) -> list[str]:
    argv: list[str] = []
    for key, value in options.items():
        flag = f"--{key.replace('_', '-')}"
        if value is None:
            continue
        if isinstance(value, bool):
            if key in script.bool_values:
                argv += [flag, str(value).lower()]
            elif value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, ",".join(str(item) for item in value)]
        else:
            argv += [flag, str(value)]
    return argv


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return config


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name) or {}
    if not isinstance(section, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    return dict(section)


def _select_extractor(
    config: dict[str, Any],
    stage: str,
    extractors: dict[str, StepScript],
) -> tuple[str, StepScript, dict[str, Any], str | None]:
    section = config_section(config, stage)
    name = section.pop("extractor", None)
    python = section.pop("python", None)
    if name not in extractors:
        raise ConfigError(
            f"{stage}.extractor must be one of {sorted(extractors)}, got {name!r}"
        )
    unknown = sorted(set(section) - set(extractors))
    if unknown:
        raise ConfigError(
            f"Unknown keys in '{stage}': {unknown}. Extractor options belong under "
            f"'{stage}.{name}'."
        )
    options = section.get(name) or {}
    if not isinstance(options, dict):
        raise ConfigError(f"'{stage}.{name}' must be a mapping")
    return name, extractors[name], dict(options), python


def _require(options: dict[str, Any], keys: tuple[str, ...], where: str) -> list[str]:
    return [f"{where}.{key}" for key in keys if options.get(key) in (None, "")]


def plan_steps(config: dict[str, Any], steps: list[str]) -> tuple[list[PlannedStep], dict[str, str]]:
    """Resolve the config into ordered step commands plus the wired hand-off names."""

    unknown_steps = sorted(set(steps) - set(STEP_ORDER))
    if unknown_steps:
        raise ConfigError(f"Unknown steps {unknown_steps}; choose from {list(STEP_ORDER)}")

    run = config_section(config, "run")
    default_python = run.get("python") or sys.executable
    common = {"project_root": run.get("project_root"), "dataset_name": run.get("dataset_name")}
    missing = _require(run, ("project_root",), "run")

    cluster_extractor, cluster_script, cluster_options, cluster_python = _select_extractor(
        config, "clustering_embedding", CLUSTERING_EXTRACTORS
    )
    verify_extractor, verify_script, verify_options, verify_python = _select_extractor(
        config, "verification_embedding", VERIFICATION_EXTRACTORS
    )
    cluster_embedding = cluster_options.get("embedding_name")
    verification_embedding = verify_options.get("embedding_name")
    if not cluster_embedding:
        missing.append(f"clustering_embedding.{cluster_extractor}.embedding_name")
    if not verification_embedding:
        missing.append(f"verification_embedding.{verify_extractor}.embedding_name")

    clustering = config_section(config, "clustering")
    clustering_python = clustering.pop("python", None)
    use_pca = bool(clustering.get("use_pca", False))
    pca_components = clustering.get("pca_components", 64)
    cluster_run_name = clustering.get("run_name") or default_cluster_run_name(
        clustering.get("n_clusters", 15),
        use_pca,
        pca_components if use_pca else None,
    )
    clustering["run_name"] = cluster_run_name

    metadata = config_section(config, "metadata")
    metadata_python = metadata.pop("python", None)
    postprocessing = config_section(config, "postprocessing")
    postprocessing_python = postprocessing.pop("python", None)

    candidates = {
        "metadata": PlannedStep(
            "metadata",
            METADATA_SCRIPT,
            metadata_python or default_python,
            {**common, **metadata},
        ),
        "clustering_embedding": PlannedStep(
            f"clustering_embedding ({cluster_extractor})",
            cluster_script,
            cluster_python or default_python,
            {**common, **cluster_options},
        ),
        "clustering": PlannedStep(
            "clustering",
            CLUSTERING_SCRIPT,
            clustering_python or default_python,
            {**common, "cluster_embedding": cluster_embedding, **clustering},
        ),
        "verification_embedding": PlannedStep(
            f"verification_embedding ({verify_extractor})",
            verify_script,
            verify_python or default_python,
            {**common, **verify_options},
        ),
        "postprocessing": PlannedStep(
            "postprocessing",
            POSTPROCESSING_SCRIPT,
            postprocessing_python or default_python,
            {
                **common,
                "cluster_embedding": cluster_embedding,
                "cluster_run_name": cluster_run_name,
                "verification_embedding": verification_embedding,
                **postprocessing,
            },
        ),
    }
    selected_names = [name for name in STEP_ORDER if name in steps]
    selected = [candidates[name] for name in selected_names]
    config_paths = {
        "clustering_embedding": f"clustering_embedding.{cluster_extractor}",
        "verification_embedding": f"verification_embedding.{verify_extractor}",
    }
    for name in selected_names:
        step = candidates[name]
        missing += _require(step.options, step.script.required, config_paths.get(name, name))
    if missing:
        raise ConfigError("Missing required config values: " + ", ".join(missing))

    nonexistent = [
        f"{config_paths.get(name, name)}.{key}: {candidates[name].options[key]}"
        for name in selected_names
        for key in INPUT_PATH_OPTIONS
        if candidates[name].options.get(key) not in (None, "")
        and not Path(candidates[name].options[key]).expanduser().exists()
    ]
    if nonexistent:
        raise ConfigError("Input paths do not exist:\n  " + "\n  ".join(nonexistent))

    handoff = {
        "cluster_embedding": cluster_embedding,
        "cluster_run_name": cluster_run_name,
        "verification_embedding": verification_embedding,
    }
    return selected, handoff


def subprocess_env() -> dict[str, str]:
    # Make minsum importable for interpreters where it is not installed.
    env = os.environ.copy()
    repo_root = str(PACKAGE_DIR.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo_root, env.get("PYTHONPATH")]))
    return env


def validate_options(steps: list[PlannedStep], env: dict[str, str]) -> None:
    """Check every generated option against the step script's --help output."""

    problems = []
    for step in steps:
        result = subprocess.run(
            [step.python, "-m", step.module, "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            problems.append(
                f"{step.name}: '{step.python} -m {step.module} --help' failed:\n"
                f"{result.stderr.strip()}"
            )
            continue
        known = set(OPTION_PATTERN.findall(result.stdout))
        argv = options_to_argv(step.options, step.script)
        unknown = [arg for arg in argv if arg.startswith("--") and arg not in known]
        if unknown:
            problems.append(f"{step.name}: unknown options {unknown} for {step.module}")
    if problems:
        raise ConfigError("Config does not match the step scripts:\n  " + "\n  ".join(problems))


def resolve_steps(config: dict[str, Any], steps_arg: str | None) -> list[str]:
    """Steps from a comma-separated override, else run.steps, else every step."""

    if steps_arg:
        return [step.strip() for step in steps_arg.split(",") if step.strip()]
    return list(config_section(config, "run").get("steps") or STEP_ORDER)


def write_run_record(
    config: dict[str, Any],
    steps: list[PlannedStep],
    handoff: dict[str, str],
) -> Path:
    run = config["run"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = (
        Path(run["project_root"]) / "outputs" / "full_runs" / str(run.get("dataset_name")) / stamp
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "run": {**run, "steps": [s.name for s in steps]}, "handoff": handoff}
    with (record_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    commands = "\n".join(shlex.join(step.command()) for step in steps)
    (record_dir / "commands.sh").write_text(f"#!/usr/bin/env bash\nset -e\n{commands}\n")
    return record_dir
