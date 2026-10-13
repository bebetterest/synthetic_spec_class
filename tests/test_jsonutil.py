from synthesis.jsonutil import parse_json_object
from synthesis.messages import normalize_messages
from synthesis.types import ParseError


def test_parse_json_object_accepts_a_fence_and_prose():
    fenced = '```json\n{"conversations": []}\n```'
    assert parse_json_object(fenced) == {"conversations": []}
    wrapped = 'Here you go:\n{"ok": true}\nThanks.'
    assert parse_json_object(wrapped) == {"ok": True}


def test_parse_json_object_rejects_a_list():
    try:
        parse_json_object("[1, 2]")
    except ParseError as exc:
        assert "object" in str(exc)
    else:
        raise AssertionError("expected ParseError")


def test_normalize_messages_requires_user_and_assistant():
    messages = normalize_messages(
        [
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        require_user=True,
        require_assistant=True,
    )
    assert [message["role"] for message in messages] == ["developer", "user", "assistant"]
    kept = normalize_messages(
        [
            {"role": "user", "content": "Hi", "trainable": False, "note": "drop"},
            {"role": "assistant", "content": "Hello", "trainable": True},
        ]
    )
    assert kept[0] == {"role": "user", "content": "Hi", "trainable": False}
    assert "note" not in kept[0]
    try:
        normalize_messages([{"role": "user", "content": "Hi", "trainable": "yes"}])
    except ParseError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("expected ParseError")
    try:
        normalize_messages([{"role": "user", "content": "Hi"}], require_assistant=True)
    except ParseError:
        return
    raise AssertionError("expected ParseError")
