from __future__ import annotations

import random
import re


def perturb_numbers(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    return re.sub(
        r"\b(\d+)\b",
        lambda m: str(max(1, int(m.group(1)) + rng.randint(-5, 5))),
        text,
    )


def rename_identifiers(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)
    keywords = {"def", "return", "import", "class", "self"}
    names_set = set(names)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for name in sorted(names_set - keywords, key=lambda s: (-len(s), s)):
        ref = f"ref_{rng.randint(100, 999)}"
        while ref in used or ref in names_set:
            ref = f"ref_{rng.randint(100, 999)}"
        used.add(ref)
        mapping[name] = ref
    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def paraphrase_prompt(text: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    prefixes = ["Please ", "Kindly ", "Can you ", ""]
    suffixes = ["", " Thanks.", " Be specific."]
    return rng.choice(prefixes) + text.strip().rstrip(".") + rng.choice(suffixes)
