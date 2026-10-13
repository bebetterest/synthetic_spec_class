"""Split Markdown into sections without treating headings inside fences as real."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^[ \t]{0,3}(```+|~~~+)")
_ATTRS = re.compile(r"\s*\{[^{}]*\}\s*$")
_ANCHOR = re.compile(r"\{#([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class Section:
    """One Markdown section.

    ``id`` prefers an explicit ``{#anchor}`` on the heading so resume keys stay
    stable when the heading text is edited. Otherwise it is a slug of the title.
    """

    id: str
    title: str
    level: int
    body: str
    ancestors: tuple[str, ...]


def split_sections(text: str, level: int) -> list[Section]:
    """Return sections whose heading has exactly ``level`` hashes.

    A section body runs until the next heading of the same or higher level,
    so a level-2 section includes the level-3 headings under it. Headings
    inside fenced code blocks are ignored. Empty bodies are skipped.
    """

    if level < 1 or level > 6:
        raise ValueError("heading level must be from 1 to 6")
    headings = _headings_outside_fences(text)
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    seen: dict[str, int] = {}
    lines = text.splitlines()
    for index, (line_index, heading_level, title, anchor) in enumerate(headings):
        while stack and stack[-1][0] >= heading_level:
            stack.pop()
        ancestors = tuple(item[1] for item in stack)
        stack.append((heading_level, title))
        if heading_level != level:
            continue
        end = len(lines)
        for later_index, later_level, _title, _anchor in headings[index + 1 :]:
            if later_level <= heading_level:
                end = later_index
                break
        body = "\n".join(lines[line_index + 1 : end]).strip()
        if not body:
            continue
        sections.append(
            Section(
                id=_unique_id(anchor or _slug(title), seen),
                title=title,
                level=heading_level,
                body=body,
                ancestors=ancestors,
            )
        )
    return sections


def _headings_outside_fences(text: str) -> list[tuple[int, int, str, str | None]]:
    headings: list[tuple[int, int, str, str | None]] = []
    fence: str | None = None
    for line_index, line in enumerate(text.splitlines()):
        marker = _fence_marker(line)
        if marker is not None:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is not None:
            continue
        match = _HEADING.match(line)
        if match is None:
            continue
        raw = match.group(2)
        anchor_match = _ANCHOR.search(raw)
        title = _ATTRS.sub("", raw).strip()
        if title:
            anchor = anchor_match.group(1) if anchor_match else None
            headings.append((line_index, len(match.group(1)), title, anchor))
    return headings


def _fence_marker(line: str) -> str | None:
    match = _FENCE.match(line)
    if match is None:
        return None
    return match.group(1)[:3]


def _unique_id(base: str, seen: dict[str, int]) -> str:
    seen[base] = seen.get(base, 0) + 1
    if seen[base] == 1:
        return base
    return f"{base}-{seen[base]}"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"
