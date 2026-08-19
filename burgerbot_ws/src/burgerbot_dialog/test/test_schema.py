"""Model output validation. Pure logic, no ROS graph and no network.

Every test here is a shape a language model has actually been observed to
return. The point is not that the parser is elegant, it is that a local 8B
model having a bad day costs you a gesture rather than the sentence.
"""

import json

from burgerbot_dialog.schema import (
    EXPRESSIONS,
    GESTURES,
    RESPONSE_SCHEMA,
    parse_reply,
)


def reply_of(raw, **kwargs):
    result = parse_reply(raw, **kwargs)
    assert result.ok, f"expected a reply, got problems: {result.problems}"
    return result.reply


# ---- the happy path -------------------------------------------------------


def test_a_well_formed_reply_round_trips():
    reply = reply_of(json.dumps({
        "say": "Hello Mark.", "expression": "happy",
        "gesture": "wiggle", "go_to": "kitchen",
    }))
    assert reply.say == "Hello Mark."
    assert reply.expression == "happy"
    assert reply.gesture == "wiggle"
    assert reply.go_to == "kitchen"


def test_only_say_is_required():
    reply = reply_of('{"say": "Hello."}')
    assert reply.say == "Hello."
    assert reply.expression == ""
    assert reply.gesture == ""
    assert reply.go_to == ""


def test_every_declared_expression_is_accepted():
    for name in EXPRESSIONS:
        assert reply_of(f'{{"say":"hi","expression":"{name}"}}').expression == name


def test_every_declared_gesture_is_accepted():
    for name in GESTURES:
        assert reply_of(f'{{"say":"hi","gesture":"{name}"}}').gesture == name


def test_the_schema_offered_to_the_server_matches_what_is_accepted():
    """Otherwise constrained decoding permits things the validator then drops."""
    assert set(RESPONSE_SCHEMA["properties"]["expression"]["enum"]) == {"", *EXPRESSIONS}
    assert set(RESPONSE_SCHEMA["properties"]["gesture"]["enum"]) == {"", *GESTURES}
    assert RESPONSE_SCHEMA["required"] == ["say"]


# ---- degrading per field, not per reply -----------------------------------


def test_an_invented_gesture_loses_the_gesture_and_keeps_the_sentence():
    """The single most important behaviour in this file.

    Small models get the enums wrong far more often than they get the prose
    wrong. Rejecting the whole reply over one bad enum throws away the good
    part and leaves the robot silent, which reads as broken.
    """
    result = parse_reply('{"say": "Of course.", "gesture": "backflip"}')
    assert result.ok
    assert result.reply.say == "Of course."
    assert result.reply.gesture == ""
    assert any("backflip" in p for p in result.problems)


def test_an_invented_expression_loses_only_the_expression():
    result = parse_reply('{"say": "Hi.", "expression": "ecstatic"}')
    assert result.ok
    assert result.reply.say == "Hi."
    assert result.reply.expression == ""


def test_capitalised_and_padded_enum_values_are_normalised():
    assert reply_of('{"say":"hi","expression":"  Happy "}').expression == "happy"
    assert reply_of('{"say":"hi","gesture":"NOD_YES"}').gesture == "nod_yes"


def test_an_enum_wrapped_in_stray_words_is_salvaged():
    assert reply_of('{"say":"hi","expression":"happy face"}').expression == "happy"
    assert reply_of('{"say":"hi","gesture":"gesture: wiggle"}').gesture == "wiggle"


def test_salvage_only_matches_whole_words():
    """'unhappy' must not become 'happy' -- that inverts the meaning."""
    result = parse_reply('{"say":"hi","expression":"unhappy"}')
    assert result.reply.expression == ""


def test_unexpected_fields_are_noted_but_harmless():
    result = parse_reply('{"say":"hi","mood":"blue","volume":11}')
    assert result.ok
    assert result.reply.say == "hi"
    assert sum("unexpected" in p for p in result.problems) == 2


# ---- responses that are not clean JSON ------------------------------------


def test_json_wrapped_in_a_markdown_fence():
    """Models are trained on documentation, and documentation fences JSON."""
    raw = '```json\n{"say": "Hello.", "expression": "happy"}\n```'
    reply = reply_of(raw)
    assert reply.say == "Hello."
    assert reply.expression == "happy"


def test_a_bare_fence_with_no_language_tag():
    assert reply_of('```\n{"say": "Hello."}\n```').say == "Hello."


def test_prose_either_side_of_the_object():
    raw = 'Sure! Here is my response:\n{"say": "Hello."}\nLet me know if that helps.'
    result = parse_reply(raw)
    assert result.ok
    assert result.reply.say == "Hello."
    assert any("text around" in p for p in result.problems)


def test_braces_inside_the_utterance_do_not_end_the_object_early():
    """Naive brace counting breaks here, and `say` is natural language."""
    reply = reply_of('{"say": "Use {curly braces} for that."}')
    assert reply.say == "Use {curly braces} for that."


def test_escaped_quotes_inside_the_utterance():
    reply = reply_of(r'{"say": "He said \"hello\" to me."}')
    assert reply.say == 'He said "hello" to me.'


def test_a_nested_object_in_a_stray_field_does_not_confuse_extraction():
    reply = reply_of('{"say": "hi", "meta": {"a": {"b": 1}}}')
    assert reply.say == "hi"


# ---- responses with nothing usable in them --------------------------------


def test_an_empty_response_is_a_failure_not_a_crash():
    for raw in ("", "   ", "\n"):
        result = parse_reply(raw)
        assert not result.ok
        assert result.problems


def test_plain_prose_with_no_json_is_a_failure():
    result = parse_reply("I would be happy to help you with that!")
    assert not result.ok


def test_malformed_json_is_a_failure_not_an_exception():
    for raw in ('{"say": ', '{"say": "hi",}', '{say: "hi"}', "{'say': 'hi'}"):
        result = parse_reply(raw)
        assert not result.ok
        assert result.problems


def test_a_json_array_is_rejected():
    result = parse_reply('[{"say": "hi"}]')
    assert not result.ok


def test_an_object_with_no_usable_content_is_rejected():
    result = parse_reply('{"expression": "", "gesture": ""}')
    assert not result.ok
    assert any("nothing in it" in p for p in result.problems)


def test_the_parser_never_raises_on_any_input():
    for raw in (None, "", "{", "}" * 500, "\x00\x01", "x" * 100_000, "[]", "null"):
        parse_reply(raw)  # must simply not raise


# ---- length and markup ----------------------------------------------------


def test_a_long_reply_is_truncated_at_a_sentence_boundary():
    long_say = "This is a sentence. " * 60
    result = parse_reply(json.dumps({"say": long_say}), max_say=100)
    assert result.ok
    assert len(result.reply.say) <= 100
    assert result.reply.say.endswith(".")
    assert any("truncated" in p for p in result.problems)


def test_a_long_reply_with_no_sentence_break_is_still_capped():
    result = parse_reply(json.dumps({"say": "word " * 200}), max_say=80)
    assert len(result.reply.say) <= 80


def test_a_short_reply_is_left_alone():
    result = parse_reply('{"say": "Hello."}', max_say=100)
    assert result.reply.say == "Hello."
    assert not any("truncated" in p for p in result.problems)


def test_stage_directions_and_markdown_are_stripped():
    """This is written to be spoken; a synthesiser reads asterisks aloud."""
    assert reply_of('{"say": "*waves* Hello there."}').say == "waves Hello there."
    assert reply_of('{"say": "That is **very** good."}').say == "That is very good."
    assert reply_of('{"say": "[happy] Hello."}').say == "Hello."


def test_newlines_and_runs_of_space_collapse():
    raw = json.dumps({"say": "One.\n\n  Two."})
    assert reply_of(raw).say == "One. Two."


# ---- wrong types ----------------------------------------------------------


def test_a_numeric_say_is_coerced_rather_than_dropped():
    result = parse_reply('{"say": 42}')
    assert result.ok
    assert result.reply.say == "42"


def test_a_structured_say_is_dropped():
    result = parse_reply('{"say": {"text": "hi"}, "gesture": "wiggle"}')
    assert result.ok  # the gesture survives
    assert result.reply.say == ""
    assert result.reply.gesture == "wiggle"


def test_wrong_typed_enums_are_dropped_without_taking_the_reply():
    result = parse_reply('{"say": "hi", "expression": 3, "gesture": ["wiggle"]}')
    assert result.ok
    assert result.reply.say == "hi"
    assert result.reply.expression == ""
    assert result.reply.gesture == ""


def test_go_to_is_free_text_because_place_names_are_taught_not_fixed():
    assert reply_of('{"say":"ok","go_to":"  the kitchen "}').go_to == "the kitchen"
    result = parse_reply('{"say":"ok","go_to":42}')
    assert result.reply.go_to == ""
