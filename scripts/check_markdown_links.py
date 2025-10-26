#!/usr/bin/env python3
"""
Simple Markdown link checker for this repository.
- Checks that relative links in Markdown files point to existing files/paths.
- Ignores external links (http/https), mailto, and pure anchors (#section).
- Allows links with anchors (e.g., README.md#usage) by validating the file part.

Usage:
    python scripts/check_markdown_links.py
Exits non-zero if broken links are found.
"""

import re
import sys
from pathlib import Path

MD_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files and href prefixes to ignore (legacy papers or ephemeral assets)
IGNORE_FILES = {
    "docs/multiview_yolov8_research_paper.md",
    "docs/revised_retail_detection_paper.md",
    "docs/FINAL_SUBMISSION_PAPER.md",
}
IGNORE_HREF_PREFIXES = (
    "data/results/",  # generated per-experiment assets not tracked in repo
)


def is_external(href: str) -> bool:
    return href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:")


def should_ignore(md_file: Path, href: str) -> bool:
    rel = str(md_file.relative_to(REPO_ROOT))
    if rel in IGNORE_FILES:
        return True
    for prefix in IGNORE_HREF_PREFIXES:
        if href.strip().startswith(prefix):
            return True
    return False


def normalize_target(md_file: Path, href: str) -> Path:
    # strip surrounding whitespace
    href = href.strip()
    # handle anchors inside href
    if "#" in href:
        href = href.split("#", 1)[0]
    if not href:
        return None  # pure anchor
    # absolute path relative to repo root
    if href.startswith("/"):
        return (REPO_ROOT / href.lstrip("/")).resolve()
    # otherwise treat as relative to the markdown file's directory
    return (md_file.parent / href).resolve()


def check_file(md_file: Path) -> list[tuple[str, str]]:
    broken = []
    text = md_file.read_text(encoding="utf-8", errors="ignore")
    # find both standard links and image links
    candidates = []
    candidates += MD_LINK_PATTERN.findall(text)
    candidates += MD_IMAGE_PATTERN.findall(text)
    for href in candidates:
        if not href or is_external(href) or href.startswith("#"):
            continue
        if should_ignore(md_file, href):
            continue
        target = normalize_target(md_file, href)
        if target is None:
            continue
        if not target.exists():
            broken.append((str(md_file.relative_to(REPO_ROOT)), href))
    return broken


def main():
    md_files = sorted(REPO_ROOT.rglob("*.md"))
    broken_all = []
    for md in md_files:
        broken_all.extend(check_file(md))
    if broken_all:
        print("Broken Markdown links found:")
        for src, href in broken_all:
            print(f" - In {src}: '{href}' does not exist")
        sys.exit(1)
    else:
        print("Markdown link check passed: no broken relative links found.")
        sys.exit(0)


if __name__ == "__main__":
    main()