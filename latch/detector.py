"""Prompt-injection detection.

The hard part is not spotting the word "ignore" -- it is that an injection and a
legitimate instruction look identical in isolation. "Ignore the previous
instructions" is a textbook attack; "ignore the previous summary and use this
version" is a normal sentence to a co-writer.

So this module scores *combinations* with bounded proximity, not keyword
presence, and every rule states why it fired. A detector that cannot explain
itself on a false positive is worse than no detector, because the operator turns
it off.

Pure functions. No model call, no network.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Rule:
    """One family of attack, with the textual evidence it looks for."""

    key: str
    label: str
    # Each group is an alternative. A group with several patterns requires ALL of
    # them (AND); the groups themselves are OR'd. Proximity lives inside each
    # pattern, which is what keeps "ignore" + "instructions" apart from
    # "ignore warnings, per the previous guideline".
    groups: tuple[tuple[str, ...], ...]
    weight: int
    category: str
    why: str


RULES: tuple[Rule, ...] = (
    Rule(
        key="instruction_override",
        label="Instruction override",
        groups=(
            (
                r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
                r"\b(?:previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,30}"
                r"\b(?:instruction|instructions|prompt|prompts|rule|rules|"
                r"guideline|guidelines|directive|directives|context|system)\b",
            ),
        ),
        weight=40,
        category="override",
        why="A verb of dismissal aimed at prior instructions.",
    ),
    Rule(
        key="role_hijack",
        label="Role hijack",
        groups=(
            (
                r"\b(?:you are|you're|from now on|act as|pretend|roleplay|behave as)\b[^.\n]{0,50}"
                r"\b(?:dan|developer mode|unrestricted|jailbroken|"
                r"no (?:rules|limits|restrictions)|without (?:rules|limits|restrictions))\b",
            ),
        ),
        weight=35,
        category="override",
        why="Reassigns the assistant's identity to one without constraints.",
    ),
    Rule(
        key="system_prompt_exfil",
        label="System prompt exfiltration",
        groups=(
            (
                r"\b(?:reveal|show|print|repeat|output|display|tell me|divulge)\b[^.\n]{0,40}"
                r"\b(?:system prompt|system message|initial prompt|initial instructions|"
                r"your instructions|your (?:rules|guidelines|directives)|hidden prompt|"
                r"pre-?prompt|configured with)\b",
            ),
        ),
        weight=45,
        category="exfiltration",
        why="Asks for the configuration itself, the highest-value secret in context.",
    ),
    Rule(
        key="delimiter_escape",
        label="Delimiter escape",
        groups=(
            (
                r"(```|<\|im_start\|>|<\|im_end\|>|\[/?INST\]|</?SYS>|</?system>|"
                r"<\|endoftext\|>|<\|end\|>)",
            ),
        ),
        weight=25,
        category="structure",
        why="Emits chat-format control tokens to break out of the data channel.",
    ),
    Rule(
        key="tool_abuse",
        label="Tool / function abuse",
        groups=(
            (
                r"\b(?:call|invoke|execute|run|use)\b[^.\n]{0,25}"
                r"\b(?:the )?(?:function|tool|api|command|shell|bash|exec|eval|subprocess)\b",
            ),
        ),
        weight=30,
        category="action",
        why="Tries to make the model act rather than answer.",
    ),
    Rule(
        key="data_exfil",
        label="Data exfiltration",
        # The destination must be a machine endpoint, not a human address:
        # "send the invoice to billing@acme" is email, a webhook is exfiltration.
        groups=(
            (
                r"\b(?:send|post|upload|exfiltrate|leak|forward|transmit)\b[^.\n]{0,60}"
                r"(?:https?://|webhook|pastebin|ngrok|requestbin|transfer\.sh|"
                r"\d{1,3}(?:\.\d{1,3}){3})",
            ),
        ),
        weight=50,
        category="exfiltration",
        why="Moves data to an attacker-controlled endpoint.",
    ),
    Rule(
        key="credential_harvest",
        label="Credential harvest",
        # A secret noun used as a *modifier* (password policy, token rotation,
        # secret manager) is a question about a control, not a request to emit a
        # secret. That distinction is the difference between a 50-point false
        # positive on ordinary support tickets and a working detector.
        groups=(
            (
                r"\b(?:send|give|show|reveal|print|output|provide|share|paste|repeat|dump)\b"
                r"[^.\n]{0,30}"
                r"\b(?:api[_ -]?key|access key|token|password|secret|credential|"
                r"private key|passphrase)\b"
                r"(?!\s*(?:polic(?:y|ies)|rotation|strength|requirements?|handbook|"
                r"rules|guidelines|manager|reset|expiry|expiration|length|complexity))",
            ),
        ),
        weight=50,
        category="exfiltration",
        why="Asks for a secret to be emitted into the transcript.",
    ),
    Rule(
        key="encoded_payload",
        label="Encoded payload",
        groups=(
            (
                r"\b(?:decode|base64|rot13|hex)\b[^.\n]{0,60}"
                r"\b(?:and (?:then )?(?:run|execute|follow|obey|do)|then)\b",
            ),
        ),
        weight=30,
        category="obfuscation",
        why="Smuggles the real instruction past keyword filters.",
    ),
    Rule(
        key="urgency_coercion",
        label="Urgency / authority coercion",
        # Requires a *threat* alongside the pressure. "Run the migration
        # immediately or the deploy will fail" is a normal ops instruction, so
        # generic words like "fail" and "you must" do not count.
        groups=(
            (
                r"\b(?:urgent|comply|immediately|right now|or else)\b[^.\n]{0,50}"
                r"\b(?:terminated|suspended|locked|deleted|penalty|legal action|"
                r"consequences|revoked)\b",
            ),
        ),
        weight=20,
        category="social",
        why="Manufactures pressure to skip the model's normal caution.",
    ),
    Rule(
        key="conversation_echo",
        label="Conversation / secret echo",
        groups=(
            (
                r"\b(?:repeat|print|output|recite|echo|summarise|summarize|dump)\b[^.\n]{0,40}"
                r"\b(?:conversation|transcript|everything (?:above|before)|"
                r"all (?:messages|text)|what you (?:were|have been) (?:told|given)|tokens you)\b",
            ),
        ),
        weight=40,
        category="exfiltration",
        why="Tries to make the model recite context it should not surface.",
    ),
    Rule(
        key="implicit_override",
        label="Implicit override",
        # Obedience to the attacker without the words "ignore previous".
        groups=(
            (
                r"\b(?:follow|obey|comply with)\b[^.\n]{0,20}\b(?:my|these)\b[^.\n]{0,20}"
                r"\b(?:directives|instructions|commands|rules)\b[^.\n]{0,15}"
                r"\b(?:only|instead|alone)\b",
            ),
            (
                r"\b(?:ignore|disregard|bypass)\b[^.\n]{0,25}"
                r"\b(?:safety|guidelines|guardrails|polic(?:y|ies)|filters?)\b",
            ),
            (r"\banswer\b[^.\n]{0,15}\bfreely\b",),
        ),
        weight=35,
        category="override",
        why="Instructs obedience to the attacker rather than to the system.",
    ),
    Rule(
        key="authority_spoof",
        label="Spoofed authority turn",
        groups=(
            (
                r"\b(?:system|developer|admin(?:istrator)?)\s*"
                r"(?:message|notice|override|instruction|directive)\b",
            ),
            (r"^\s*(?:system|assistant|developer)\s*:",),
        ),
        weight=30,
        category="structure",
        why="Pretends the text came from the operator instead of the data.",
    ),
    Rule(
        key="prompt_leak_probe",
        label="Prompt leak probe",
        groups=(
            (
                r"\b(?:what|which)\b[^.\n]{0,25}\b(?:prompt|instructions|rules)\b[^.\n]{0,20}"
                r"\b(?:were you|are you|do you have|given|configured)\b",
            ),
        ),
        weight=35,
        category="exfiltration",
        why="Probes for the configuration by asking rather than demanding.",
    ),
)

# Zero-width characters used to break keyword matching.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# Homoglyph folding. Deliberately small and documented: a large table starts
# mangling legitimate non-Latin text.
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ν": "v", "α": "a", "ο": "o",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "У": "Y", "Х": "X",
})

# Endpoints that only matter inside retrieved/third-party content.
INDIRECT_MARKERS = (
    "http://", "https://", "www.", "@", "<!--", "-->", "data:", "javascript:",
)


def normalize(text: str) -> str:
    """
    Fold text into the form the rules expect.

    Handles the cheap evasions: full-width characters, combining marks (U+0336
    and friends), zero-width joiners, and Cyrillic/Greek lookalikes. It does not
    defeat a determined attacker -- it removes the accidental ones so the rules
    can see intent.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    if any(unicodedata.category(ch) == "Mn" for ch in t):
        # Drop combining marks: they break keyword matches while looking identical.
        t = "".join(
            ch for ch in unicodedata.normalize("NFD", t)
            if unicodedata.category(ch) != "Mn"
        )
    t = t.translate(_ZERO_WIDTH)
    t = t.translate(_HOMOGLYPHS)
    return t


@dataclass
class Signal:
    rule: str
    label: str
    category: str
    weight: int
    why: str
    excerpt: str


@dataclass
class Verdict:
    score: int
    action: str            # allow | flag | block
    signals: list[Signal] = field(default_factory=list)
    normalized_len: int = 0

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "action": self.action,
            "signals": [
                {
                    "rule": s.rule,
                    "label": s.label,
                    "category": s.category,
                    "weight": s.weight,
                    "why": s.why,
                    "excerpt": s.excerpt,
                }
                for s in self.signals
            ],
        }


# Thresholds in one place so they are reviewable. They sit in the gap measured
# against the adversarial corpus: benign text peaks at 20 (a coercion signal with
# no threat behind it), while the weakest true attack scores 25. A single weak
# signal therefore still only flags; blocking needs a real family hit.
BLOCK_AT = 25
FLAG_AT = 15


def _excerpt(text: str, match: re.Match, pad: int = 40) -> str:
    start = max(0, match.start() - pad)
    end = min(len(text), match.end() + pad)
    body = text[start:end].replace("\n", " ")
    return ("..." if start else "") + body + ("..." if end < len(text) else "")


def scan(text: str, rules: Sequence[Rule] = RULES) -> Verdict:
    """
    Score a piece of text for injection intent.

    Returns a Verdict with a score, an action, and per-rule evidence. The action
    is `block` only at or above BLOCK_AT, so a single weak signal flags rather
    than blocks.
    """
    norm = normalize(text)
    if not norm.strip():
        return Verdict(0, "allow", [], 0)

    flat = re.sub(r"\s+", " ", norm).lower()
    lower = norm.lower()
    signals: list[Signal] = []

    for rule in rules:
        for group in rule.groups:
            match = None
            for pattern in group:
                m = re.search(pattern, lower)
                if m is None:
                    match = None
                    break
                if match is None:
                    match = m
            if match is not None:
                signals.append(
                    Signal(
                        rule=rule.key,
                        label=rule.label,
                        category=rule.category,
                        weight=rule.weight,
                        why=rule.why,
                        excerpt=_excerpt(flat, match),
                    )
                )
                break  # one hit per rule; groups are alternatives

    # Overlapping categories compound: exfiltration plus override is materially
    # worse than either alone, which is a property of the attacks, not a fudge.
    cats = {s.category for s in signals}
    score = sum(s.weight for s in signals)
    if len(cats) >= 2:
        score += 10 * (len(cats) - 1)

    score = min(score, 100)
    action = "block" if score >= BLOCK_AT else "flag" if score >= FLAG_AT else "allow"
    return Verdict(score, action, signals, len(norm))


def scan_indirect(document: str, rules: Sequence[Rule] = RULES) -> Verdict:
    """
    Scan untrusted *retrieved* content (a web page, a PDF, an email body).

    Indirect injection hides in content the model was asked to read, so a bare
    instruction-like sentence is more suspicious here than in a user turn: a web
    page has no reason to address the assistant. That raises the score.
    """
    base = scan(document, rules)
    if base.score == 0:
        return base

    lowered = normalize(document).lower()
    carries_endpoint = any(m in lowered for m in INDIRECT_MARKERS)
    score = min(base.score + (15 if carries_endpoint else 0), 100)
    action = "block" if score >= BLOCK_AT else "flag" if score >= FLAG_AT else "allow"
    return Verdict(score, action, base.signals, base.normalized_len)


def batch(texts: Iterable[str]) -> list[Verdict]:
    return [scan(t) for t in texts]


def worst(verdicts: Sequence[Verdict]) -> Verdict:
    """The most severe verdict in a set, for reducing a batch to one decision."""
    if not verdicts:
        return Verdict(0, "allow")
    return max(verdicts, key=lambda v: v.score)


def metrics(labels: Sequence[bool], predicted: Sequence[bool]) -> dict:
    """
    Precision/recall for a benchmark run.

    Report them together: a detector tuned for recall alone blocks half of all
    legitimate traffic, and one tuned for precision misses the attacks.
    """
    if len(labels) != len(predicted):
        raise ValueError("labels and predictions must be the same length")
    tp = sum(1 for l, p in zip(labels, predicted) if l and p)
    fp = sum(1 for l, p in zip(labels, predicted) if not l and p)
    fn = sum(1 for l, p in zip(labels, predicted) if l and not p)
    tn = sum(1 for l, p in zip(labels, predicted) if not l and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(labels), 4) if labels else 0.0,
    }
