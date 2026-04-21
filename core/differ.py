"""
Structural diff engine with prompt anatomy tagging.
No model dependencies — pure Python.
"""

from __future__ import annotations

import difflib
import re
from core.schemas import (
    AnatomyType, DiffTag, DiffSegment, PromptDiff
)

# ---------------------------------------------------------------------------
# Anatomy keyword patterns
# ---------------------------------------------------------------------------

ANATOMY_PATTERNS: list[tuple[AnatomyType, list[str]]] = [
    (AnatomyType.ROLE, [
        r"\byou are\b", r"\bact as\b", r"\bbehave as\b", r"\byour role\b",
        r"\byou're a\b", r"\bassume the role\b", r"\bpersona\b",
    ]),
    (AnatomyType.CONSTRAINT, [
        r"\bdo not\b", r"\bdon't\b", r"\bavoid\b", r"\bnever\b",
        r"\bmust\b", r"\bshould not\b", r"\bwithout\b", r"\bno more than\b",
        r"\bexclude\b", r"\brefrain\b", r"\blimit(ed)? to\b",
    ]),
    (AnatomyType.FORMAT, [
        r"\brespond in\b", r"\bformat(ted)?\b", r"\bbullet point", r"\bnumbered list\b",
        r"\bjson\b", r"\bmarkdown\b", r"\btable\b", r"\bstructured\b",
        r"\bone sentence\b", r"\bparagraph\b", r"\bword limit\b",
    ]),
    (AnatomyType.EXAMPLE, [
        r"\bfor example\b", r"\bfor instance\b", r"\bsuch as\b", r"\be\.g\.\b",
        r"\blike this\b", r"\bsample\b", r"\bdemonstrate\b", r"\bshown below\b",
    ]),
    (AnatomyType.CONTEXT, [
        r"\bcontext\b", r"\bbackground\b", r"\bgiven that\b", r"\bassuming\b",
        r"\bthe user is\b", r"\bthe audience\b", r"\bsituation\b",
    ]),
    (AnatomyType.INSTRUCTION, [
        r"\bexplain\b", r"\bsummarise\b", r"\bsummarize\b", r"\bwrite\b",
        r"\bgenerate\b", r"\blist\b", r"\bdescribe\b", r"\banalyse\b",
        r"\banalyze\b", r"\bcompare\b", r"\bidentify\b", r"\bcreate\b",
    ]),
]


def classify_anatomy(text: str) -> AnatomyType:
    """Classify a text segment into a prompt anatomy type."""
    text_lower = text.lower()
    for anatomy_type, patterns in ANATOMY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return anatomy_type
    return AnatomyType.UNKNOWN


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _sentence_tokenize(text: str) -> list[str]:
    """Split text into sentence-level tokens for diff."""
    # Split on sentence boundaries while preserving delimiters
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Also split on newlines
    result = []
    for part in parts:
        lines = part.split("\n")
        result.extend(lines)
    return [p for p in result if p.strip()]


def _word_tokenize(text: str) -> list[str]:
    """Split into word-level tokens."""
    return re.findall(r'\S+|\s+', text)


# ---------------------------------------------------------------------------
# Core differ
# ---------------------------------------------------------------------------

def build_diff(prompt_v1: str, prompt_v2: str) -> PromptDiff:
    """
    Build a full structural diff with anatomy tagging.
    Uses sentence-level diff for high-level changes,
    word-level for inline changes within sentences.
    """
    segs_v1 = _sentence_tokenize(prompt_v1)
    segs_v2 = _sentence_tokenize(prompt_v2)

    matcher = difflib.SequenceMatcher(None, segs_v1, segs_v2, autojunk=False)
    opcodes = matcher.get_opcodes()

    result_v1: list[DiffSegment] = []
    result_v2: list[DiffSegment] = []
    anatomy_changes: list[dict] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for sent in segs_v1[i1:i2]:
                result_v1.append(DiffSegment(text=sent, tag=DiffTag.UNCHANGED, anatomy=classify_anatomy(sent)))
            for sent in segs_v2[j1:j2]:
                result_v2.append(DiffSegment(text=sent, tag=DiffTag.UNCHANGED, anatomy=classify_anatomy(sent)))

        elif tag == "delete":
            for sent in segs_v1[i1:i2]:
                anatomy = classify_anatomy(sent)
                result_v1.append(DiffSegment(text=sent, tag=DiffTag.REMOVED, anatomy=anatomy))
                anatomy_changes.append({"type": anatomy.value, "tag": "removed", "text": sent})

        elif tag == "insert":
            for sent in segs_v2[j1:j2]:
                anatomy = classify_anatomy(sent)
                result_v2.append(DiffSegment(text=sent, tag=DiffTag.ADDED, anatomy=anatomy))
                anatomy_changes.append({"type": anatomy.value, "tag": "added", "text": sent})

        elif tag == "replace":
            # Mark as CHANGED on both sides
            for sent in segs_v1[i1:i2]:
                anatomy = classify_anatomy(sent)
                result_v1.append(DiffSegment(text=sent, tag=DiffTag.CHANGED, anatomy=anatomy))
                anatomy_changes.append({"type": anatomy.value, "tag": "changed_from", "text": sent})
            for sent in segs_v2[j1:j2]:
                anatomy = classify_anatomy(sent)
                result_v2.append(DiffSegment(text=sent, tag=DiffTag.CHANGED, anatomy=anatomy))
                anatomy_changes.append({"type": anatomy.value, "tag": "changed_to", "text": sent})

    return PromptDiff(
        segments_v1=result_v1,
        segments_v2=result_v2,
        anatomy_changes=anatomy_changes,
    )


def summarise_anatomy_changes(diff: PromptDiff) -> dict[str, list[str]]:
    """Group anatomy changes by type for display."""
    summary: dict[str, list[str]] = {}
    for change in diff.anatomy_changes:
        t = change["type"]
        label = f"[{change['tag'].upper()}] {change['text'][:80]}{'...' if len(change['text']) > 80 else ''}"
        summary.setdefault(t, []).append(label)
    return summary
