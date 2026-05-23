from __future__ import annotations

from text_to_sign_production.workflows.foundation.review import review_field, to_review_value


def test_to_review_value_normalizes_mapping_with_list_value() -> None:
    value = to_review_value({"batch_size_candidates": [64, 128, 256]})

    field = review_field("candidates", value)

    assert field.value == ("batch_size_candidates=[64, 128, 256]",)


def test_to_review_value_normalizes_nested_mapping() -> None:
    value = to_review_value({"nested": {"a": 1, "b": [2, 3]}})

    field = review_field("nested", value)

    assert field.value == ("nested=[a=1, b=[2, 3]]",)


def test_to_review_value_normalizes_list_containing_mapping() -> None:
    value = to_review_value([1, "x", {"a": 2}])

    field = review_field("mixed", value)

    assert field.value == (1, "x", ("a=2",))
