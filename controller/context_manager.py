from __future__ import annotations


def _truncate_by_words(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def deduplicate_lines(text: str) -> str:
    seen: set[str] = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def format_observation(raw: str, max_tokens: int = 800) -> str:
    text = deduplicate_lines(raw)
    text = _truncate_by_words(text, max(1, max_tokens - 4))
    return f"<|obs|> {text} <|obs_end|>"

