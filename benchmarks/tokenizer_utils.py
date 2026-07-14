from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def tokenizer_fingerprint(path: str | Path) -> str:
    root = Path(path)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
    digest = hashlib.sha256()
    found = False
    for name in names:
        candidate = root / name
        if not candidate.exists():
            continue
        found = True
        digest.update(name.encode())
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"No tokenizer artifacts found under {root}")
    return "sha256:" + digest.hexdigest()


def chat_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )
        )
    text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    if add_generation_prompt:
        text += "\nassistant:"
    return list(tokenizer.encode(text, add_special_tokens=True))


def token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", token_id))
    return "sha256:" + digest.hexdigest()


def fit_system_content(
    tokenizer: PreTrainedTokenizerBase,
    content: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    raw_ids = list(tokenizer.encode(content, add_special_tokens=False))
    low, high = 0, len(raw_ids)
    best_text = ""
    best_ids: list[int] = []
    while low <= high:
        middle = (low + high) // 2
        candidate = tokenizer.decode(raw_ids[:middle], skip_special_tokens=True)
        ids = chat_tokens(
            tokenizer,
            [{"role": "system", "content": candidate}],
            add_generation_prompt=False,
        )
        if len(ids) <= target_tokens:
            best_text, best_ids = candidate, ids
            low = middle + 1
        else:
            high = middle - 1
    return best_text, best_ids
