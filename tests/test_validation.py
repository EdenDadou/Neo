"""
Tests — Validation d'inputs
==============================
"""

import pytest

from neo_core.validation import (
    validate_message,
    validate_task_description,
    validate_session_id,
    ValidationError,
    InputTooLongError,
    EmptyInputError,
    MAX_MESSAGE_LENGTH,
    MAX_TASK_LENGTH,
)


class TestValidateMessage:
    def test_normal_message(self):
        assert validate_message("Bonjour Neo") == "Bonjour Neo"

    def test_strips_whitespace(self):
        assert validate_message("  hello  ") == "hello"

    def test_empty_raises(self):
        with pytest.raises(EmptyInputError):
            validate_message("")

    def test_whitespace_only_raises(self):
        with pytest.raises(EmptyInputError):
            validate_message("   \n\t  ")

    def test_too_long_raises(self):
        with pytest.raises(InputTooLongError) as exc_info:
            validate_message("A" * (MAX_MESSAGE_LENGTH + 1))
        assert exc_info.value.length == MAX_MESSAGE_LENGTH + 1
        assert exc_info.value.max_length == MAX_MESSAGE_LENGTH

    def test_custom_max_length(self):
        with pytest.raises(InputTooLongError):
            validate_message("ABCDEF", max_length=5)

    def test_non_string_raises(self):
        with pytest.raises(ValidationError):
            validate_message(123)

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_message(None)

    def test_unicode_ok(self):
        assert validate_message("日本語テスト") == "日本語テスト"

    def test_exact_max_length(self):
        msg = "A" * MAX_MESSAGE_LENGTH
        assert validate_message(msg) == msg


class TestValidateTaskDescription:
    def test_normal(self):
        assert validate_task_description("Faire un rapport") == "Faire un rapport"

    def test_too_long(self):
        with pytest.raises(InputTooLongError):
            validate_task_description("X" * (MAX_TASK_LENGTH + 1))

    def test_empty(self):
        with pytest.raises(EmptyInputError):
            validate_task_description("")


class TestValidateSessionId:
    def test_valid_uuid(self):
        assert validate_session_id("abc-123-def") == "abc-123-def"

    def test_strips(self):
        assert validate_session_id("  abc123  ") == "abc123"

    def test_empty_raises(self):
        with pytest.raises(EmptyInputError):
            validate_session_id("")

    def test_too_long_raises(self):
        with pytest.raises(ValidationError):
            validate_session_id("a" * 101)

    def test_invalid_chars_raises(self):
        with pytest.raises(ValidationError):
            validate_session_id("abc 123")

    def test_special_chars_raises(self):
        with pytest.raises(ValidationError):
            validate_session_id("abc;DROP TABLE")

    def test_non_string_raises(self):
        with pytest.raises(ValidationError):
            validate_session_id(42)

    def test_underscores_ok(self):
        assert validate_session_id("session_abc_123") == "session_abc_123"
