import io
import tarfile

from benchmarks.repo_context import GitRepoCache


def add_file(bundle: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    bundle.addfile(info, io.BytesIO(content))


def test_archive_context_prefers_patch_paths_and_skips_binary(tmp_path, monkeypatch) -> None:
    cache = GitRepoCache(tmp_path)
    archive = cache._archive_path("owner/repo", "commit")
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as bundle:
        add_file(bundle, "repo-commit/src/changed.py", b"changed = True\n")
        add_file(bundle, "repo-commit/src/other.py", b"other = True\n")
        add_file(bundle, "repo-commit/src/binary.py", b"bad\x00data")
        add_file(bundle, "repo-commit/ignored.bin", b"ignored")

    monkeypatch.setattr(
        cache, "list_files",
        lambda repo, commit: ["src/changed.py", "src/other.py", "src/binary.py", "ignored.bin"],
    )
    patch = "diff --git a/src/changed.py b/src/changed.py\n+++ b/src/changed.py\n"
    context = cache.build_context("owner/repo", "commit", patch, target_chars=10_000)
    assert context.index("src/changed.py") < context.index("src/other.py")
    assert "changed = True" in context
    assert "binary.py" not in context
    assert "ignored.bin" not in context
