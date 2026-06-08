"""Smoke tests: the vendored core imports and basic helpers work offline.

These must stay green after ``pip install -e ".[dev]"`` (milestone M0 exit
criterion). They exercise no network and need no real API key.
"""
from __future__ import annotations

from drawing_takeoff.core import api_config, pricing, tokenizer
from tests.fixtures.fake_anthropic import FakeClient, FakeMessage, FakeTextBlock, FakeUsage


def test_core_imports_and_default_model():
    assert api_config.REVIEW_MODEL_DEFAULT
    # pricing recognizes the default model
    assert pricing.friendly_model_name(api_config.REVIEW_MODEL_DEFAULT)


def test_image_token_estimate_is_positive():
    est = tokenizer.estimate_image_tokens(1992, 1992, model=api_config.REVIEW_MODEL_DEFAULT)
    assert est > 0


def test_fake_client_injection_roundtrips():
    client = FakeClient(
        lambda kw: FakeMessage(
            content=[FakeTextBlock(text="ok")],
            usage=FakeUsage(input_tokens=3, output_tokens=1),
        )
    )
    resp = client.messages.create(model="x", messages=[])
    assert resp.content[0].text == "ok"
    assert client.messages.calls  # the request was recorded for assertions
