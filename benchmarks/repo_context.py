from __future__ import annotations

import os
import re
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

_PATCH_PATH = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_ALLOWED_SUFFIXES = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".json"}
_LOW_VALUE_CONTEXT_PARTS = {
    ".github",
    "docs",
    "doc",
    "examples",
    "locale",
    "migrations",
}


def _ordered_context_paths(all_files: list[str], patch: str) -> list[str]:
    changed = [path for path in _PATCH_PATH.findall(patch) if path in all_files]
    changed_set = set(changed)
    parent_dirs = {str(PurePosixPath(path).parent) for path in changed}
    roots = {
        PurePosixPath(path).parts[0]
        for path in changed
        if PurePosixPath(path).parts
    }
    candidates = [
        path
        for path in all_files
        if PurePosixPath(path).suffix.lower() in _ALLOWED_SUFFIXES
        and (not roots or PurePosixPath(path).parts[0] in roots)
        and (
            path in changed_set
            or not any(part in _LOW_VALUE_CONTEXT_PARTS for part in PurePosixPath(path).parts)
        )
    ]
    same_parent = [
        path
        for path in candidates
        if path not in changed_set and str(PurePosixPath(path).parent) in parent_dirs
    ]
    same_root = [
        path
        for path in candidates
        if path not in changed_set and path not in same_parent
    ]
    return list(dict.fromkeys(changed + sorted(same_parent) + sorted(same_root)))


class GitRepoCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        ssh_key = os.getenv("GITHUB_SSH_KEY")
        self.ssh_key = Path(ssh_key).expanduser() if ssh_key else None

    def _git_dir(self, repo: str) -> Path:
        return self.root / (repo.replace("/", "__") + ".git")

    def _fetch_url(self, repo: str) -> str:
        scheme = os.getenv("GITHUB_FETCH_URL_SCHEME", "https").lower()
        if scheme == "ssh":
            return f"git@github.com:{repo}.git"
        if scheme != "https":
            raise ValueError("GITHUB_FETCH_URL_SCHEME must be https or ssh")
        return f"https://github.com/{repo}.git"

    def _archive_path(self, repo: str, commit: str) -> Path:
        return self.root / "snapshots" / repo.replace("/", "__") / f"{commit}.tar.gz"

    def _run(
        self, args: list[str], *, check: bool = True, allow_lazy_fetch: bool = False
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if allow_lazy_fetch:
            env.pop("GIT_NO_LAZY_FETCH", None)
        else:
            env["GIT_NO_LAZY_FETCH"] = "1"
        if self.ssh_key is not None and self.ssh_key.exists():
            env["GIT_SSH_COMMAND"] = (
                f"ssh -i {self.ssh_key} -o IdentitiesOnly=yes "
                "-o StrictHostKeyChecking=accept-new"
            )
        return subprocess.run(
            args, check=check, capture_output=True, text=True, env=env
        )

    def ensure_repo(self, repo: str) -> Path:
        git_dir = self._git_dir(repo)
        url = self._fetch_url(repo)
        if not git_dir.exists():
            git_dir.mkdir(parents=True)
            self._run(["git", "init", "--bare", str(git_dir)])
            self._run(["git", f"--git-dir={git_dir}", "remote", "add", "origin", url])
        else:
            self._run(["git", f"--git-dir={git_dir}", "remote", "set-url", "origin", url])
        return git_dir

    def ensure_commit(self, repo: str, commit: str) -> Path:
        git_dir = self.ensure_repo(repo)
        def has_commit() -> bool:
            return (
                self._run(
                    [
                        "git",
                        f"--git-dir={git_dir}",
                        "cat-file",
                        "-e",
                        f"{commit}^{{commit}}",
                    ],
                    check=False,
                ).returncode
                == 0
            )

        if has_commit():
            return git_dir
        last_error = ""
        for attempt in range(3):
            result = self._run(
                [
                    "git",
                    "-c",
                    "http.version=HTTP/1.1",
                    f"--git-dir={git_dir}",
                    "fetch",
                    "--depth=1",
                    "--filter=blob:none",
                    "--no-tags",
                    "origin",
                    commit,
                ],
                check=False,
            )
            if has_commit():
                return git_dir
            last_error = result.stderr.strip()
            time.sleep(attempt + 1)
        raise RuntimeError(f"failed to fetch {repo}@{commit}: {last_error}")

    def ensure_archive(self, repo: str, commit: str) -> Path:
        archive = self._archive_path(repo, commit)
        if archive.exists() and tarfile.is_tarfile(archive):
            return archive
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".part")
        archive_mirror = os.getenv("GITHUB_ARCHIVE_MIRROR", "").rstrip("/")
        url = (
            f"{archive_mirror}/{repo}/archive/{commit}.tar.gz"
            if archive_mirror
            else f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        )
        env = os.environ.copy()
        download_mode = env.get("GITHUB_DOWNLOAD_MODE", "auto").lower()
        if archive_mirror and "GITHUB_DOWNLOAD_MODE" not in env:
            download_mode = "direct"
        if download_mode not in {"auto", "direct", "proxy"}:
            raise ValueError("GITHUB_DOWNLOAD_MODE must be one of: auto, direct, proxy")
        proxy_names = (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
        )
        direct_env = env.copy()
        for name in proxy_names:
            direct_env.pop(name, None)
        has_proxy = any(env.get(name) for name in proxy_names)
        modes = [download_mode]
        if download_mode == "auto":
            modes = ["direct", "proxy"] if has_proxy else ["direct"]
        result: subprocess.CompletedProcess[str] | None = None
        for mode in modes:
            temporary.unlink(missing_ok=True)
            mode_env = direct_env if mode == "direct" else env
            probing_direct = (
                download_mode == "auto" and mode == "direct" and has_proxy
            )
            speed_guard = (
                ["--speed-limit", "131072", "--speed-time", "15"]
                if probing_direct
                else []
            )
            retry_args = (
                []
                if probing_direct
                else [
                    "--retry",
                    "5",
                    "--retry-all-errors",
                    "--retry-delay",
                    "1",
                ]
            )
            attempts = 1 if probing_direct else 3
            for attempt in range(attempts):
                result = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        *retry_args,
                        "--connect-timeout",
                        "10",
                        *speed_guard,
                        "--output",
                        str(temporary),
                        url,
                    ],
                    check=False,
                    env=mode_env,
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0 and tarfile.is_tarfile(temporary):
                    temporary.replace(archive)
                    return archive
                temporary.unlink(missing_ok=True)
                time.sleep(attempt + 1)
            if mode == "direct" and download_mode == "auto":
                continue
        detail = result.stderr if result is not None else "no download attempted"
        raise RuntimeError(
            f"Failed to download valid snapshot {repo}@{commit}: {detail}"
        )

    def prefetch_archives(
        self, snapshots: list[tuple[str, str]], workers: int = 8
    ) -> list[dict[str, Any]]:
        unique = sorted(set(snapshots))
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.ensure_archive, repo, commit): (repo, commit)
                for repo, commit in unique
            }
            for future in as_completed(futures):
                repo, commit = futures[future]
                path = future.result()
                digest = sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                rows.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
        return sorted(rows, key=lambda row: (row["repo"], row["commit"]))

    def prefetch_commits(
        self, snapshots: list[tuple[str, str]], workers: int = 4
    ) -> list[dict[str, Any]]:
        """Fetch commit trees without downloading a complete archive per task."""
        grouped: dict[str, list[str]] = {}
        for repo, commit in sorted(set(snapshots)):
            grouped.setdefault(repo, []).append(commit)

        def fetch_repo(item: tuple[str, list[str]]) -> list[dict[str, Any]]:
            repo, commits = item
            result = []
            for commit in commits:
                git_dir = self.ensure_commit(repo, commit)
                result.append(
                    {
                        "repo": repo,
                        "commit": commit,
                        "storage": "partial_git",
                        "git_dir": str(git_dir),
                    }
                )
            return result

        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(grouped) or 1)) as pool:
            futures = [pool.submit(fetch_repo, item) for item in grouped.items()]
            for future in as_completed(futures):
                rows.extend(future.result())
        return sorted(rows, key=lambda row: (row["repo"], row["commit"]))

    def list_files(self, repo: str, commit: str) -> list[str]:
        git_dir = self.ensure_commit(repo, commit)
        result = self._run(
            ["git", f"--git-dir={git_dir}", "ls-tree", "-r", "--name-only", commit]
        )
        return [line for line in result.stdout.splitlines() if line]

    def read_file(self, repo: str, commit: str, path: str) -> str | None:
        git_dir = self.ensure_commit(repo, commit)
        result = self._run(
            ["git", f"--git-dir={git_dir}", "show", f"{commit}:{path}"],
            check=False,
            allow_lazy_fetch=True,
        )
        if result.returncode != 0 or "\x00" in result.stdout:
            return None
        if len(result.stdout.encode("utf-8", errors="ignore")) > 256 * 1024:
            return None
        return result.stdout

    def build_context(
        self,
        repo: str,
        commit: str,
        patch: str,
        *,
        target_chars: int,
    ) -> str:
        archive = self._archive_path(repo, commit)
        if not archive.exists() or not tarfile.is_tarfile(archive):
            return self._build_context_from_git(repo, commit, patch, target_chars)

        git_files = set(self.list_files(repo, commit))
        with tarfile.open(archive, "r:gz") as bundle:
            members: dict[str, tarfile.TarInfo] = {}
            for member in bundle.getmembers():
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) < 2 or member.size > 256 * 1024:
                    continue
                relative = str(PurePosixPath(*parts[1:]))
                if relative in git_files:
                    members[relative] = member
            all_files = sorted(members)
            ordered = _ordered_context_paths(all_files, patch)
            chunks: list[str] = []
            size = 0
            for path in ordered:
                source = bundle.extractfile(members[path])
                if source is None:
                    continue
                raw = source.read()
                if b"\x00" in raw:
                    continue
                content = raw.decode("utf-8", errors="replace")
                chunk = f"\n\n### FILE: {path}\n{content}"
                chunks.append(chunk)
                size += len(chunk)
                if size >= target_chars:
                    break
        return "".join(chunks)

    def _build_context_from_git(
        self, repo: str, commit: str, patch: str, target_chars: int
    ) -> str:
        all_files = sorted(self.list_files(repo, commit))
        ordered = _ordered_context_paths(all_files, patch)
        chunks: list[str] = []
        size = 0
        for path in ordered:
            content = self.read_file(repo, commit, path)
            if content is None:
                continue
            chunk = f"\n\n### FILE: {path}\n{content}"
            chunks.append(chunk)
            size += len(chunk)
            if size >= target_chars:
                break
        return "".join(chunks)
