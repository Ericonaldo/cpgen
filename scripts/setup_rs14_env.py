#!/usr/bin/env python3
"""Create and validate the deterministic RS1.4 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


LOCK_PATH = Path("third_party/vendor-lock.json")
DEFAULT_ENVIRONMENT_NAME = "cpgen-rs14-unified"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "cpgen" / "rs14"
_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_VENDOR_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_VENDOR_INSTALL_ORDER = (
    "robosuite",
    "robomimic",
    "mimicgen",
    "cpgen-envs",
    "curobo",
)
_RUNTIME_VERSIONS = {
    "robosuite": ("robosuite", "1.4.1"),
    "mujoco": ("mujoco", "2.3.2"),
    "robomimic": ("robomimic", "0.3.1"),
    "mink": ("mink", "0.0.7"),
    "curobo": ("nvidia-curobo", "0.7.8"),
}
_SQUARE_ENVIRONMENTS = ("Square_D0", "Square_D1", "SquareWide")


def _lock_error(message: str) -> ValueError:
    return ValueError("invalid vendor lock: {}".format(message))


def _locked_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _lock_error("{} must be a non-empty relative path".format(field))
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise _lock_error("{} must stay within the repository".format(field))
    return path


def _validate_lock(lock: Any) -> dict:
    if not isinstance(lock, dict):
        raise _lock_error("top-level value must be an object")
    if lock.get("schema_version") != 1:
        raise _lock_error("schema_version must be 1")
    vendors = lock.get("vendors")
    if not isinstance(vendors, dict) or not vendors:
        raise _lock_error("vendors must be a non-empty object")

    for name, vendor in vendors.items():
        if (
            not isinstance(name, str)
            or not _VENDOR_NAME_PATTERN.fullmatch(name)
            or not isinstance(vendor, dict)
        ):
            raise _lock_error("each vendor must have a safe name and object value")
        _locked_relative_path(vendor.get("path"), field="{}.path".format(name))
        url = vendor.get("url")
        if not isinstance(url, str) or not url:
            raise _lock_error("{}.url must be a non-empty string".format(name))
        commit = vendor.get("commit")
        if not isinstance(commit, str) or not _COMMIT_PATTERN.fullmatch(commit):
            raise _lock_error(
                "{}.commit must be a full 40-character commit SHA".format(name)
            )
        patch = vendor.get("patch")
        if patch is not None:
            if not isinstance(patch, dict):
                raise _lock_error("{}.patch must be an object".format(name))
            _locked_relative_path(
                patch.get("path"), field="{}.patch.path".format(name)
            )
            digest = patch.get("sha256")
            if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
                raise _lock_error(
                    "{}.patch.sha256 must be a 64-character SHA-256".format(name)
                )
    return lock


def discover_repository(start: Path) -> Path:
    """Find the containing checkout by walking upward from *start*."""
    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / LOCK_PATH).is_file() and (candidate / ".git").exists():
            return candidate
    raise ValueError(
        "could not find a repository containing {}".format(LOCK_PATH.as_posix())
    )


def load_vendor_lock(root: Path) -> dict:
    """Load and structurally validate ``third_party/vendor-lock.json``."""
    path = Path(root).resolve() / LOCK_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _lock_error("could not read {}: {}".format(path, exc)) from exc
    return _validate_lock(payload)


def _git(
    repository: Path,
    *args: str,
    extra_environment: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ.copy()
    environment.update({"LC_ALL": "C", "LANG": "C"})
    if extra_environment is not None:
        environment.update(extra_environment)
    command = ["git", "-C", str(repository), *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            text=True,
        )
    except OSError as exc:
        raise ValueError("could not execute git: {}".format(exc)) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(
            "git command failed for {}: {}".format(repository, detail)
        )
    return completed.stdout.strip()


def _repository_path(root: Path, relative: Any, *, field: str) -> Path:
    root = root.resolve()
    path = (root / _locked_relative_path(relative, field=field)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise _lock_error("{} resolves outside the repository".format(field)) from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_repository(root: Path, lock: dict) -> None:
    """Fail unless every locked checkout and patch matches its immutable identity."""
    root = Path(root).resolve()
    validated = _validate_lock(lock)
    for name, vendor in validated["vendors"].items():
        source = _repository_path(root, vendor["path"], field="{}.path".format(name))
        if not source.is_dir():
            raise ValueError("{} vendor checkout is missing: {}".format(name, source))
        if _git(source, "rev-parse", "--is-inside-work-tree") != "true":
            raise ValueError("{} is not a Git checkout: {}".format(name, source))
        actual_commit = _git(source, "rev-parse", "HEAD")
        expected_commit = vendor["commit"].lower()
        if actual_commit.lower() != expected_commit:
            raise ValueError(
                "{} HEAD is {}; expected {}".format(
                    name, actual_commit, expected_commit
                )
            )

        patch = vendor.get("patch")
        if patch is None:
            continue
        patch_path = _repository_path(
            root, patch["path"], field="{}.patch.path".format(name)
        )
        if not patch_path.is_file():
            raise ValueError("{} patch is missing: {}".format(name, patch_path))
        actual_digest = _sha256(patch_path)
        if actual_digest.lower() != patch["sha256"].lower():
            raise ValueError(
                "{} patch sha256 is {}; expected {}".format(
                    name, actual_digest, patch["sha256"].lower()
                )
            )


def _cache_identity(name: str, vendor: dict) -> tuple[str, dict]:
    patch = vendor.get("patch")
    identity = {
        "schema_version": 1,
        "vendor": name,
        "commit": vendor["commit"].lower(),
        "patch_sha256": (
            patch["sha256"].lower() if patch is not None else None
        ),
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity


def _cached_tree_matches(
    path: Path, identity: dict, patch: Path | None
) -> bool:
    with tempfile.TemporaryDirectory(prefix=".rs14-index-") as temporary:
        temporary_path = Path(temporary)
        expected_environment = {
            "GIT_INDEX_FILE": str(temporary_path / "expected-index")
        }
        _git(
            path,
            "read-tree",
            identity["commit"],
            extra_environment=expected_environment,
        )
        if patch is not None:
            _git(
                path,
                "apply",
                "--cached",
                "--check",
                str(patch),
                extra_environment=expected_environment,
            )
            _git(
                path,
                "apply",
                "--cached",
                str(patch),
                extra_environment=expected_environment,
            )
        added_paths = _git(
            path,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=A",
            extra_environment=expected_environment,
        ).splitlines()
        expected_tree = _git(
            path, "write-tree", extra_environment=expected_environment
        )

        actual_environment = {
            "GIT_INDEX_FILE": str(temporary_path / "actual-index")
        }
        _git(
            path,
            "read-tree",
            identity["commit"],
            extra_environment=actual_environment,
        )
        _git(
            path,
            "add",
            "-u",
            "--",
            ".",
            extra_environment=actual_environment,
        )
        if added_paths:
            _git(
                path,
                "add",
                "-f",
                "--",
                *added_paths,
                extra_environment=actual_environment,
            )
        actual_tree = _git(path, "write-tree", extra_environment=actual_environment)
    return actual_tree == expected_tree


def _matches_cache(
    path: Path, identity: dict, patch: Path | None
) -> bool:
    marker = path / ".rs14-source.json"
    if not path.is_dir() or not marker.is_file():
        return False
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
        if actual != identity:
            return False
        if _git(path, "rev-parse", "HEAD").lower() != identity["commit"]:
            return False
        if not _cached_tree_matches(path, identity, patch):
            return False
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    return True


def _reject_nested_cache_root(root: Path, cache_root: Path, lock: dict) -> None:
    for name, vendor in lock["vendors"].items():
        source = _repository_path(root, vendor["path"], field="{}.path".format(name))
        try:
            cache_root.relative_to(source)
        except ValueError:
            continue
        raise ValueError(
            "cache root {} must not be inside vendor checkout {}".format(
                cache_root, source
            )
        )


def _build_vendor_cache(
    *,
    source: Path,
    patch: Path | None,
    destination: Path,
    identity: dict,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".{}-".format(destination.name), dir=str(destination.parent)
        )
    )
    checkout = temporary / "source"
    try:
        _git(source.parent, "clone", "--quiet", "--no-hardlinks", str(source), str(checkout))
        _git(checkout, "checkout", "--quiet", "--detach", identity["commit"])
        if patch is not None:
            _git(checkout, "apply", "--check", str(patch))
            _git(checkout, "apply", str(patch))
        marker = checkout / ".rs14-source.json"
        marker.write_text(
            json.dumps(identity, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            os.replace(str(checkout), str(destination))
        except OSError:
            if not _matches_cache(destination, identity, patch):
                raise
    finally:
        shutil.rmtree(str(temporary), ignore_errors=True)


def prepare_vendor_sources(
    root: Path, cache_root: Path, lock: dict
) -> dict[str, Path]:
    """Materialize every locked vendor in the immutable source cache."""
    root = Path(root).resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    validated = _validate_lock(lock)
    _reject_nested_cache_root(root, cache_root, validated)
    verify_repository(root, lock)

    sources = {}
    for name, vendor in validated["vendors"].items():
        source = _repository_path(root, vendor["path"], field="{}.path".format(name))
        patch = vendor.get("patch")
        fingerprint, identity = _cache_identity(name, vendor)
        destination = cache_root / "{}-{}".format(name, fingerprint[:16])
        patch_path = (
            _repository_path(
                root, patch["path"], field="{}.patch.path".format(name)
            )
            if patch is not None
            else None
        )
        if not _matches_cache(destination, identity, patch_path):
            if destination.exists():
                raise ValueError(
                    "cached vendor source has an invalid identity: {}".format(
                        destination
                    )
                )
            _build_vendor_cache(
                source=source,
                patch=patch_path,
                destination=destination,
                identity=identity,
            )
        sources[name] = destination
    return sources


def _target_python_prefix(name: str) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "--name",
        name,
        "python",
    ]


def build_conda_environment_command(
    *, root: Path, name: str, environment_exists: bool
) -> list[str]:
    """Build the idempotent Conda create or update command."""
    action = "update" if environment_exists else "create"
    return [
        "conda",
        "env",
        action,
        "--name",
        name,
        "--file",
        str(Path(root).resolve() / "env.yml"),
    ]


def build_target_pip_commands(
    *, root: Path, name: str, sources: Mapping[str, Path]
) -> list[list[str]]:
    """Build dependency-suppressed pip phases for the target environment."""
    missing = [vendor for vendor in _VENDOR_INSTALL_ORDER if vendor not in sources]
    if missing:
        raise ValueError(
            "prepared vendor sources are missing: {}".format(", ".join(missing))
        )

    prefix = _target_python_prefix(name) + ["-m", "pip", "install"]
    commands = [prefix + ["--no-deps", "mink==0.0.7"]]
    for vendor in _VENDOR_INSTALL_ORDER:
        options = ["--no-deps"]
        if vendor == "curobo":
            options.append("--no-build-isolation")
        commands.append(prefix + options + [str(Path(sources[vendor]).resolve())])
    commands.append(
        prefix + ["--no-deps", "--editable", str(Path(root).resolve())]
    )
    return commands


def _runtime_verification_source() -> str:
    expected = repr(_RUNTIME_VERSIONS)
    environments = repr(_SQUARE_ENVIRONMENTS)
    return "\n".join(
        [
            "import importlib",
            "from importlib import metadata",
            "expected = {}".format(expected),
            "for module_name, (distribution, wanted) in expected.items():",
            "    module = importlib.import_module(module_name)",
            "    actual = getattr(module, '__version__', None)",
            "    if actual is None:",
            "        actual = metadata.version(distribution)",
            "    if str(actual).lstrip('v') != wanted:",
            "        raise RuntimeError('{} version is {}; expected {}'.format("
            "module_name, actual, wanted))",
            "import mimicgen",
            "import cpgen_envs",
            "from robosuite.environments.base import REGISTERED_ENVS",
            "required = {}".format(environments),
            "missing = [name for name in required if name not in REGISTERED_ENVS]",
            "if missing:",
            "    raise RuntimeError('missing Square environments: {}'.format("
            "', '.join(missing)))",
            "print('RS1.4 runtime verification passed')",
        ]
    )


def build_runtime_verification_command(name: str) -> list[str]:
    """Build the target-environment import, version, and registry probe."""
    return _target_python_prefix(name) + ["-c", _runtime_verification_source()]


def _run_checked(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=capture_output,
            text=True,
        )
    except OSError as exc:
        raise ValueError(
            "could not execute {}: {}".format(command[0], exc)
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = ""
        if capture_output:
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = ": {}".format(detail) if detail else ""
        raise ValueError(
            "command failed ({}): {}{}".format(
                exc.returncode, " ".join(command), suffix
            )
        ) from exc


def conda_environment_exists(name: str) -> bool:
    """Return whether Conda already has the exact named environment."""
    completed = _run_checked(
        ["conda", "env", "list", "--json"], capture_output=True
    )
    try:
        payload = json.loads(completed.stdout)
    except (AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("conda returned invalid environment-list JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("conda returned invalid environment-list JSON")
    environments = payload.get("envs", [])
    if not isinstance(environments, list):
        raise ValueError("conda returned invalid environment-list JSON")
    details = payload.get("envs_details")
    if details is not None:
        if not isinstance(details, dict):
            raise ValueError("conda returned invalid environment-details JSON")
        matches = []
        for prefix, detail in details.items():
            if (
                not isinstance(prefix, str)
                or not isinstance(detail, dict)
                or not isinstance(detail.get("name"), str)
            ):
                raise ValueError(
                    "conda returned invalid environment-details JSON"
                )
            if detail["name"] == name:
                matches.append(prefix)
        if len(matches) > 1:
            raise ValueError(
                "conda environment name is ambiguous: {} ({})".format(
                    name, ", ".join(sorted(matches))
                )
            )
        return len(matches) == 1

    info = _run_checked(["conda", "info", "--json"], capture_output=True)
    try:
        info_payload = json.loads(info.stdout)
        environment_directories = info_payload.get("envs_dirs", [])
    except (AttributeError, json.JSONDecodeError) as exc:
        raise ValueError("conda returned invalid info JSON") from exc
    if not isinstance(info_payload, dict) or not isinstance(
        environment_directories, list
    ):
        raise ValueError("conda returned invalid info JSON")
    if not all(isinstance(path, str) for path in environments):
        raise ValueError("conda returned invalid environment-list JSON")
    if not all(isinstance(path, str) for path in environment_directories):
        raise ValueError("conda returned invalid info JSON")

    known_prefixes = {
        os.path.abspath(os.path.expanduser(path)) for path in environments
    }
    named_prefixes = {
        os.path.abspath(os.path.join(os.path.expanduser(directory), name))
        for directory in environment_directories
    }
    matches = known_prefixes.intersection(named_prefixes)
    if len(matches) > 1:
        raise ValueError(
            "conda environment name is ambiguous: {} ({})".format(
                name, ", ".join(sorted(matches))
            )
        )
    return len(matches) == 1


def initialize_submodules(root: Path) -> None:
    """Initialize the vendor checkouts required by the lock file."""
    _run_checked(
        [
            "git",
            "-C",
            str(Path(root).resolve()),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ]
    )


def check_system_requirements() -> None:
    """Check host tools needed to resolve and build the pinned environment."""
    required = ("conda", "git", "gcc-11", "g++-11", "nvcc")
    missing = [
        executable
        for executable in required
        if shutil.which(executable) is None
    ]
    if missing:
        raise ValueError(
            "missing required system executables: {} "
            "(use --skip-system-checks to bypass this preflight)".format(
                ", ".join(missing)
            )
        )


def verify_environment(name: str) -> None:
    """Run final runtime validation in an existing target environment."""
    _run_checked(build_runtime_verification_command(name))


def bootstrap_environment(
    *, root: Path, name: str, sources: Mapping[str, Path]
) -> None:
    """Create or update the environment, install locked sources, and verify it."""
    exists = conda_environment_exists(name)
    _run_checked(
        build_conda_environment_command(
            root=root,
            name=name,
            environment_exists=exists,
        )
    )
    for command in build_target_pip_commands(
        root=root,
        name=name,
        sources=sources,
    ):
        _run_checked(command)
    verify_environment(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify the deterministic RS1.4 environment."
    )
    parser.add_argument("--name", default=DEFAULT_ENVIRONMENT_NAME)
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT, dest="cache_dir"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-system-checks", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = discover_repository(Path.cwd())
        initialize_submodules(root)
        lock = load_vendor_lock(root)
        verify_repository(root, lock)
        if not args.skip_system_checks:
            check_system_requirements()
        if args.verify_only:
            verify_environment(args.name)
            print(
                "RS1.4 repository and runtime verification passed: {}".format(
                    root
                )
            )
            return 0
        sources = prepare_vendor_sources(root, args.cache_dir, lock)
        bootstrap_environment(root=root, name=args.name, sources=sources)
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    result = {
        "environment_name": args.name,
        "repository": str(root),
        "vendor_sources": {
            name: str(path) for name, path in sorted(sources.items())
        },
    }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
