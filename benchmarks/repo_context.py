from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath


_PATCH_PATH = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_ALLOWED_SUFFIXES = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".json"}


class GitRepoCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _git_dir(self, repo: str) -> Path:
        return self.root / (repo.replace("/", "__") + ".git")

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=check, capture_output=True, text=True)

    def ensure_repo(self, repo: str) -> Path:
        git_dir = self._git_dir(repo)
        if not git_dir.exists():
            self._run(
                [
                    "git",
                    "-c",
                    "http.version=HTTP/1.1",
                    "clone",
                    "--bare",
                    "--filter=blob:none",
                    "--no-tags",
                    f"https://github.com/{repo}.git",
                    str(git_dir),
                ]
            )
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
                    "origin",
                    commit,
                ]
            )
        return git_dir

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
        all_files = self.list_files(repo, commit)
        changed = [path for path in _PATCH_PATH.findall(patch) if path in all_files]
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
            content = self.read_file(repo, commit, path)
            if not content:
                continue
            chunk = f"\n\n### FILE: {path}\n{content}"
            chunks.append(chunk)
            size += len(chunk)
            if size >= target_chars:
                break
        return "".join(chunks)

