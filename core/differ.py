"""
Structural diff engine with prompt anatomy tagging.
No model dependencies — pure Python.
"""

from __future__ import annotations

import difflib
import re
from core.schemas import AnatomyType, DiffTag, DiffSegment, PromptDiff


# ---------------------------------------------------------------------------
# Anatomy keyword patterns — ordered from most-specific to most-general.
# CONSTRAINT checked before FORMAT to avoid "do not use bullet points"
# being tagged as FORMAT instead of CONSTRAINT.
# ---------------------------------------------------------------------------

ANATOMY_PATTERNS: list[tuple[AnatomyType, list[str]]] = [

    (AnatomyType.ROLE, [
        # Explicit persona assignment
        r"\byou are\b", r"\byou're a\b", r"\byou are a\b", r"\byou are an\b",
        r"\bact as\b", r"\bbehave as\b", r"\bpretend (to be|you are)\b",
        r"\byour role\b", r"\bassume the role\b", r"\btake the role\b",
        r"\bpersona\b", r"\bimpersonate\b", r"\bspeak as\b",
        r"\bas a (senior|junior|expert|professional|specialist|developer|engineer|scientist|doctor|teacher|coach|analyst|consultant|writer|editor|lawyer|manager)\b",
        r"\bacting as\b", r"\bposing as\b",
    ]),

    (AnatomyType.CONSTRAINT, [
        # Prohibitions and limits
        r"\bdo not\b", r"\bdon't\b", r"\bdo n't\b",
        r"\bnever\b", r"\bavoid\b", r"\brefrain\b", r"\bwithhold\b",
        r"\bwithout\b", r"\bexclude\b", r"\bno (more than|less than|longer than)\b",
        r"\bmust not\b", r"\bshould not\b", r"\bcannot\b", r"\bcan't\b",
        r"\bforbid\b", r"\bprohibit\b", r"\brestrict\b",
        r"\bonly (use|include|mention|refer|respond|output|say|write|give)\b",
        r"\blimit(ed)? to\b", r"\bat most\b", r"\bno more than\b",
        r"\bstrictly\b", r"\bmandatory\b", r"\brequired\b",
        r"\bdo not (include|mention|say|write|use|add|repeat|provide|refer)\b",
        r"\bnot (allowed|permitted|acceptable)\b",
    ]),

    (AnatomyType.FORMAT, [
        # Output structure and presentation
        r"\brespond (in|with|using)\b", r"\bformat(ted)?\b", r"\bstructure(d)?\b",
        r"\bas a (list|numbered list|table|paragraph|summary|report)\b",
        r"\bbullet point", r"\bnumbered list\b", r"\bordered list\b",
        r"\buse (bullet|numbered|markdown|json|yaml|xml|html|a table|headers)\b",
        r"\bjson\b", r"\bmarkdown\b", r"\byaml\b", r"\bxml\b",
        r"\btable\b", r"\bheading(s)?\b", r"\bsubheading\b",
        r"\bone sentence\b", r"\bone word\b", r"\bone paragraph\b",
        r"\bword limit\b", r"\bmax(imum)? (words?|sentences?|characters?|lines?|bullets?)\b",
        r"\bkeep it (short|brief|concise|under)\b",
        r"\bout(put|line)\b",
        r"\bin (plain|simple|formal|technical|casual|professional) (english|language|text|prose|tone)\b",
        r"\bin the following format\b", r"\busing the format\b",
        r"\bas a (list|table|paragraph|summary|report|essay|email|code block)\b",
    ]),

    (AnatomyType.EXAMPLE, [
        # Demonstration and illustration
        r"\bfor example\b", r"\bfor instance\b",
        r"\bsuch as\b", r"\blike this\b",
        r"\be\.g\.", r"\bi\.e\.",
        r"\bsample\b", r"\bexample(s)?\b",
        r"\bdemonstrate\b", r"\bshown? below\b", r"\bas follows\b",
        r"\bsee (below|above|the following)\b",
        r"\bhere('s| is) an? (example|sample|instance)\b",
        r"\bthe following (example|sample)\b",
        r"\bInput:\b", r"\bOutput:\b",   # few-shot markers
    ]),

    (AnatomyType.CONTEXT, [
        # Background, setting, audience info
        r"\bcontext\b", r"\bbackground\b", r"\bsetting\b",
        r"\bgiven that\b", r"\bassuming\b", r"\bassume\b",
        r"\bthe user (is|has|wants|needs)\b",
        r"\bthe audience\b", r"\bthe reader\b",
        r"\bsituation\b", r"\bscenario\b", r"\buse case\b",
        r"\bin the context of\b", r"\bin the (field|domain|area) of\b",
        r"\bworking (on|with|in)\b",
        r"\bour (company|product|team|project|system|app|service)\b",
        r"\bthe (following|above|below) (text|content|document|input|data|passage)\b",
    ]),

    (AnatomyType.INSTRUCTION, [
        # Action verbs — what the model is asked to DO
        # Wide net — this is the catch-all for task specification
        r"\bexplain\b", r"\bsummarise\b", r"\bsummarize\b",
        r"\bwrite\b", r"\bgenerate\b", r"\bcreate\b", r"\bproduce\b",
        r"\blist\b", r"\bdescribe\b", r"\boutline\b",
        r"\banalyse\b", r"\banalyze\b", r"\bevaluate\b", r"\bassess\b",
        r"\bcompare\b", r"\bcontrast\b", r"\bidentify\b",
        r"\btranslate\b", r"\bparaphrase\b", r"\brewrite\b", r"\bedit\b",
        r"\bclassify\b", r"\bcategorize\b", r"\blabel\b", r"\brate\b", r"\bscore\b",
        r"\bextract\b", r"\bfind\b", r"\bdetect\b", r"\blocate\b",
        r"\banswer\b", r"\brespond\b", r"\breply\b",
        r"\bcheck\b", r"\bverify\b", r"\bvalidate\b", r"\btest\b",
        r"\bfix\b", r"\bcorrect\b", r"\bimprove\b", r"\boptimize\b", r"\boptimise\b",
        r"\bconvert\b", r"\btransform\b", r"\bformat\b",
        r"\bpredict\b", r"\bestimate\b", r"\bcalculate\b", r"\bcompute\b",
        r"\bbrainstorm\b", r"\bsuggest\b", r"\brecommend\b", r"\bpropose\b",
        r"\bhelp (me|us)\b", r"\btell me\b", r"\bgive me\b", r"\bshow me\b",
        r"\bwhat (is|are|was|were|do|does|should|would|can)\b",
        r"\bhow (to|do|does|can|should|would)\b",
        r"\bwhy (is|are|do|does|did|would)\b",
        r"\bwhen (is|are|do|does|should)\b",
        r"\bwho (is|are|was|were)\b",
        r"\btask:\b", r"\byour (task|job|goal|objective|mission) is\b",
        r"\bplease\b",
        r"\bI (want|need|would like|am looking for|want a)\b", r"\bI.?d (like|want)\b",
    ]),
]


def classify_anatomy(text: str) -> AnatomyType:
    """
    Classify a text segment into its prompt anatomy type.
    Checks patterns in priority order (Role → Constraint → Format →
    Example → Context → Instruction).
    Returns UNKNOWN only if truly no pattern fires.
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return AnatomyType.UNKNOWN

    for anatomy_type, patterns in ANATOMY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return anatomy_type

    # Last resort: if the segment starts with a capital and is a complete
    # sentence (has a verb), call it INSTRUCTION rather than UNKNOWN
    if re.search(r'\b(is|are|was|were|be|have|has|do|does|will|would|can|could|should|may|might)\b', text_lower):
        return AnatomyType.INSTRUCTION

    return AnatomyType.UNKNOWN


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _sentence_tokenize(text: str) -> list[str]:
    """
    Split text into sentence-level tokens for diffing.
    Splits on: sentence-ending punctuation, newlines, colons followed by content.
    """
    # First split on newlines
    lines = text.strip().split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Split on sentence boundaries within each line
        parts = re.split(r'(?<=[.!?])\s+', line)
        result.extend(p.strip() for p in parts if p.strip())
    return result


# ---------------------------------------------------------------------------
# Core differ
# ---------------------------------------------------------------------------

def build_diff(prompt_v1: str, prompt_v2: str) -> PromptDiff:
    """
    Build a structural diff with anatomy tagging.
    Uses sentence-level diffing — each sentence is a unit.
    """
    segs_v1 = _sentence_tokenize(prompt_v1)
    segs_v2 = _sentence_tokenize(prompt_v2)

    matcher = difflib.SequenceMatcher(None, segs_v1, segs_v2, autojunk=False)

    result_v1: list[DiffSegment] = []
    result_v2: list[DiffSegment] = []
    anatomy_changes: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for sent in segs_v1[i1:i2]:
                result_v1.append(DiffSegment(text=sent, tag=DiffTag.UNCHANGED, anatomy=classify_anatomy(sent)))
            for sent in segs_v2[j1:j2]:
                result_v2.append(DiffSegment(text=sent, tag=DiffTag.UNCHANGED, anatomy=classify_anatomy(sent)))

        elif tag == "delete":
            for sent in segs_v1[i1:i2]:
                anat = classify_anatomy(sent)
                result_v1.append(DiffSegment(text=sent, tag=DiffTag.REMOVED, anatomy=anat))
                anatomy_changes.append({"type": anat.value, "tag": "removed", "text": sent})

        elif tag == "insert":
            for sent in segs_v2[j1:j2]:
                anat = classify_anatomy(sent)
                result_v2.append(DiffSegment(text=sent, tag=DiffTag.ADDED, anatomy=anat))
                anatomy_changes.append({"type": anat.value, "tag": "added", "text": sent})

        elif tag == "replace":
            for sent in segs_v1[i1:i2]:
                anat = classify_anatomy(sent)
                result_v1.append(DiffSegment(text=sent, tag=DiffTag.CHANGED, anatomy=anat))
                anatomy_changes.append({"type": anat.value, "tag": "changed_from", "text": sent})
            for sent in segs_v2[j1:j2]:
                anat = classify_anatomy(sent)
                result_v2.append(DiffSegment(text=sent, tag=DiffTag.CHANGED, anatomy=anat))
                anatomy_changes.append({"type": anat.value, "tag": "changed_to", "text": sent})

    return PromptDiff(
        segments_v1=result_v1,
        segments_v2=result_v2,
        anatomy_changes=anatomy_changes,
    )


def summarise_anatomy_changes(diff: PromptDiff) -> dict[str, list[str]]:
    """Group anatomy changes by type for the UI summary panel."""
    summary: dict[str, list[str]] = {}
    for change in diff.anatomy_changes:
        t = change["type"]
        text = change["text"]
        label = (
            f"[{change['tag'].upper()}] "
            f"{text[:80]}{'...' if len(text) > 80 else ''}"
        )
        summary.setdefault(t, []).append(label)
    return summary
