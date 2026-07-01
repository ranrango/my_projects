#!/usr/bin/env python3
"""Scan tracked portfolio files for publish-blocking private data."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS = {
    "api-key-shaped secret": re.compile(r"sk-proj-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]{12,}"),
    "hardcoded api_key assignment": re.compile(
        r"api_key\s*=\s*[\"']sk-[A-Za-z0-9_-]{12,}[\"']"
    ),
    "local user path": re.compile(r"(?:file://)?/Users/[A-Za-z0-9._-]+"),
    "mac temp path": re.compile(r"/(?:private/)?var/folders/"),
    "demo rtsp password": re.compile("rtsp://" + "user:password"),
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".sh",
    ".env",
    ".example",
    ".svg",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(line) for line in output.splitlines() if line.strip()]


def is_text_candidate(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name == ".env.example"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    hits: list[tuple[str, str]] = []
    for path in tracked_files():
        if not is_text_candidate(path) or not path.exists():
            continue
        text = read_text(path)
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                hits.append((str(path), label))

    if hits:
        print("Privacy scan failed:")
        for path, label in hits:
            print(f"- {path}: {label}")
        return 1

    print("Privacy scan passed: no targeted hits in tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
