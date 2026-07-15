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


class GitRepoCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        ssh_key = os.getenv("GITHUB_SSH_KEY")
        self.ssh_key = Path(ssh_key).expanduser() if ssh_key else None

    def _git_dir(self, repo: str) -> Path:
        return self.root / (repo.replace("/", "__") + ".git")

    def _archive_path(self, repo: str, commit: str) -> Path:
        return self.root / "snapshots" / repo.replace("/", "__") / f"{commit}.tar.gz"

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["GIT_NO_LAZY_FETCH"] = "1"
        if self.ssh_key is not None and self.ssh_key.exists():
            env["GIT_SSH_COMMAND"] = (
                f"ssh -i {self.ssh_key} -o IdentitiesOnly=yes "
                "-o StrictHostKeyChecking=accept-new"
            )
        return subprocess.run(args, check=check, capture_output=True, text=True, env=env)

    def ensure_repo(self, repo: str) -> Path:
        git_dir = self._git_dir(repo)
        if not git_dir.exists():
            url = (
                f"git@github.com:{repo}.git"
                if self.ssh_key is not None and self.ssh_key.exists()
                else f"https://github.com/{repo}.git"
            )
            git_dir.mkdir(parents=True)
            self._run(
                ["git", "init", "--bare", str(git_dir)]
            )
            self._run(["git", f"--git-dir={git_dir}", "remote", "add", "origin", url])
        return git_dir

    def ensure_commit(self, repo: str, commit: str) -> Path:
        git_dir = self.ensure_repo(repo)
        exists = self._run(
            ["git", f"--git-dir={git_dir}", "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
        )
        if exists.returncode != 0:
            self._run(
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
                ]
            )
        return git_dir

    def ensure_archive(self, repo: str, commit: str) -> Path:
        archive = self._archive_path(repo, commit)
        if archive.exists() and tarfile.is_tarfile(archive):
            return archive
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".part")
        url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        env = os.environ.copy()
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
            env.pop(name, None)
        for attempt in range(3):
            result = subprocess.run(
                [
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "5", "--retry-all-errors", "--retry-delay", "1",
                    "--continue-at", "-", "--output", str(temporary), url,
                ],
                check=False, env=env, text=True, capture_output=True,
            )
            if result.returncode == 0 and tarfile.is_tarfile(temporary):
                temporary.replace(archive)
                return archive
            if result.returncode == 33:
                temporary.unlink(missing_ok=True)
            time.sleep(attempt + 1)
        raise RuntimeError(f"Failed to download valid snapshot {repo}@{commit}: {result.stderr}")

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
                rows.append({
                    "repo": repo, "commit": commit, "path": str(path),
                    "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                })
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
            ["git", f"--git-dir={git_dir}", "show", f"{commit}:{path}"], check=False
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
        git_files = set(self.list_files(repo, commit))
        archive = self.ensure_archive(repo, commit)
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
            changed = [path for path in _PATCH_PATH.findall(patch) if path in members]
            roots = {PurePosixPath(path).parts[0] for path in changed if PurePosixPath(path).parts}
            related = [
                path
                for path in all_files
                if PurePosixPath(path).suffix.lower() in _ALLOWED_SUFFIXES
                and (not roots or PurePosixPath(path).parts[0] in roots)
            ]
            remaining = [
                path
                for path in all_files
                if PurePosixPath(path).suffix.lower() in _ALLOWED_SUFFIXES
                and path not in changed
                and path not in related
            ]
            ordered = list(dict.fromkeys(changed + sorted(related) + sorted(remaining)))
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
