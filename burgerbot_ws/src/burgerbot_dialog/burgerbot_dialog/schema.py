"""The contract with the language model, and the validator that enforces it.

Pure logic, no ROS and no HTTP, so every way a model can return something
unhelpful is a unit test rather than something discovered while a robot is
driving at somebody.

Two decisions shape this file.

*A single JSON object, not tool/function calling.* Local model support for
function calling is uneven and differs between servers, while JSON-schema
constrained decoding is well supported by both vLLM (`guided_json`) and Ollama
(`format`). Asking for one object with four fields is the thing most likely to
work on whatever the reader actually runs.

*Validate defensively anyway.* A server that promises constrained decoding can
still be misconfigured, pointed at a model that ignores the grammar, or simply
be an older build. The validator therefore assumes nothing, and -- this is the
part that matters -- it degrades **per field**. A hallucinated gesture name
drops the gesture and keeps the sentence. Throwing away a good reply because
one enum was wrong is the failure that makes a companion feel broken, and it is
the more likely failure by far, because the enums are the part a small model
gets wrong most often.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Valid expressions, mirroring the constants in ExpressionCommand.msg. Kept as
#: a literal list rather than imported from burgerbot_msgs so this module stays
#: importable, and testable, without a built ROS workspace.
EXPRESSIONS = (
    "neutral", "happy", "curious", "focused", "confused",
    "startled", "sad", "sleepy", "determined", "error", "nervous",
)

#: Valid gestures, mirroring the GESTURES table in burgerbot_expressions.
#: `celebrate` is a full 360 spin and `recoil` is a sharp reverse; both are
#: excluded from what the model may ask for, not because they would fail but
#: because neither is something to do in conversation at arm's length from
#: somebody. The feasibility gate would allow them -- this is a taste
#: judgement, and it belongs here rather than in the gate.
GESTURES = (
    "nod_yes", "shake_no", "wiggle", "curious_tilt",
    "anticipate", "dance", "spin_delight", "bounce",
)

#: Hard cap on a single utterance. A small model asked to be chatty will
#: happily produce three paragraphs, which is tedious to read and would be
#: unbearable once this is spoken aloud. Truncating at a sentence boundary
#: beats both refusing the reply and letting it run.
MAX_SAY_CHARS = 320


@dataclass
class Reply:
    """One validated turn's worth of intent."""

    say: str = ""
    #: Empty means "no opinion", which leaves the face to whatever the
    #: companion and the telemetry sources are already saying about it.
    expression: str = ""
    gesture: str = ""
    #: A place name to be resolved by places.py, not a pose. The model has no
    #: idea where anything is and must not be asked to produce coordinates.
    go_to: str = ""

    def is_empty(self) -> bool:
        return not (self.say or self.expression or self.gesture or self.go_to)


@dataclass
class ParseResult:
    reply: Optional[Reply] = None
    #: Human-readable notes about anything that was wrong but survivable.
    #: Logged rather than raised, and worth watching: a steady stream of
    #: "unknown gesture" means the prompt is not constraining the model and the
    #: schema is not being enforced by the server.
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.reply is not None


#: JSON schema handed to the server for constrained decoding. Deliberately
#: flat: nested objects and arrays are where small models' grammar adherence
#: falls apart first, and nothing here needs the structure.
RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "say": {"type": "string"},
        "expression": {"type": "string", "enum": ["", *EXPRESSIONS]},
        "gesture": {"type": "string", "enum": ["", *GESTURES]},
        "go_to": {"type": "string"},
    },
    "required": ["say"],
    "additionalProperties": False,
}


def parse_reply(raw: str, max_say: int = MAX_SAY_CHARS) -> ParseResult:
    """Turn whatever the model returned into a Reply, or explain why not."""
    result = ParseResult()

    if not raw or not raw.strip():
        result.problems.append("empty response")
        return result

    payload, note = _extract_json_object(raw)
    if note:
        result.problems.append(note)
    if payload is None:
        result.problems.append("no JSON object found in response")
        return result
    if not isinstance(payload, dict):
        result.problems.append(f"expected a JSON object, got {type(payload).__name__}")
        return result

    reply = Reply()

    say = payload.get("say")
    if isinstance(say, str):
        reply.say = _tidy(say, max_say, result.problems)
    elif isinstance(say, (int, float, bool)):
        # Recoverable. A list or dict means the model answered a different
        # question entirely and there is nothing to salvage.
        reply.say = str(say)
        result.problems.append("'say' was not a string")
    elif say is not None:
        result.problems.append(f"'say' was {type(say).__name__}, ignored")

    reply.expression = _enum_field(payload, "expression", EXPRESSIONS, result.problems)
    reply.gesture = _enum_field(payload, "gesture", GESTURES, result.problems)

    go_to = payload.get("go_to")
    if isinstance(go_to, str):
        # Free text on purpose: the set of place names is whatever the user has
        # taught the robot, so it cannot be a fixed enum. places.py decides
        # whether it means anything, and says so out loud when it does not.
        reply.go_to = go_to.strip()
    elif go_to is not None:
        result.problems.append(f"'go_to' was {type(go_to).__name__}, ignored")

    for key in payload:
        if key not in ("say", "expression", "gesture", "go_to"):
            result.problems.append(f"ignored unexpected field '{key}'")

    if reply.is_empty():
        result.problems.append("reply had nothing in it")
        return result

    result.reply = reply
    return result


def _enum_field(payload: dict, key: str, allowed: Tuple[str, ...],
                problems: List[str]) -> str:
    """One constrained field, dropped rather than fatal when it is wrong."""
    value = payload.get(key)
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        problems.append(f"'{key}' was {type(value).__name__}, ignored")
        return ""

    cleaned = value.strip().lower()
    if cleaned in allowed:
        return cleaned

    # Small models produce "Happy", "happy face", "gesture: wiggle". Salvaging
    # those is worth a few lines -- the intent is unambiguous, and the
    # alternative is a robot that stares blankly because of a stray word.
    # Only whole-word matches, so "unhappy" does not become "happy".
    for candidate in allowed:
        if re.search(rf"(?<![a-z]){re.escape(candidate)}(?![a-z])", cleaned):
            problems.append(f"'{key}' was '{value}', read as '{candidate}'")
            return candidate

    problems.append(f"unknown {key} '{value}', ignored")
    return ""


def _tidy(text: str, limit: int, problems: List[str]) -> str:
    """Collapse whitespace, strip markup, and cap the length."""
    cleaned = " ".join(text.split())
    # Stage directions and markdown. Harmless on a screen, but this is written
    # to be spoken eventually, and a synthesiser reads "asterisk waves
    # asterisk" out loud. Cheaper to strip now than to discover later.
    cleaned = re.sub(r"\*+([^*]*)\*+", r"\1", cleaned)
    cleaned = re.sub(r"^\s*\[[^\]]*\]\s*", "", cleaned)
    cleaned = " ".join(cleaned.split())

    if len(cleaned) <= limit:
        return cleaned

    # Cut at the last sentence end inside the limit so the robot stops
    # somewhere deliberate rather than mid-word. Fall back to a hard cut only
    # when there is no sentence break at all in the first two thirds.
    window = cleaned[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    trimmed = window[: cut + 1].strip() if cut > limit // 3 else window.rstrip()
    problems.append(f"truncated a {len(cleaned)} character reply to {len(trimmed)}")
    return trimmed


def _extract_json_object(raw: str) -> Tuple[Optional[Any], str]:
    """Find the JSON object in a response that may be wrapped in anything.

    Tried in order of how clean the response was, so the note returned says
    something useful about how far the model strayed:

      1. The whole string parses. What a well-behaved server returns.
      2. Strip a markdown code fence. Extremely common -- models are trained on
         documentation, and documentation puts JSON in fences.
      3. Scan for the first balanced object. Catches "Sure! Here you go: {...}"
         and anything else with prose on either side.
    """
    stripped = raw.strip()

    try:
        return json.loads(stripped), ""
    except (ValueError, TypeError):
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip()), "response was wrapped in a code fence"
        except (ValueError, TypeError):
            pass

    span = _first_balanced_object(stripped)
    if span is not None:
        try:
            return json.loads(span), "response had text around the JSON"
        except (ValueError, TypeError):
            pass

    return None, ""


def _first_balanced_object(text: str) -> Optional[str]:
    """The first {...} with balanced braces, ignoring braces inside strings.

    Counting braces naively breaks on any reply containing one in its own text
    -- and `say` is natural language, so that does happen. Tracking string
    state and escapes is the minimum needed to be right rather than usually
    right.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
