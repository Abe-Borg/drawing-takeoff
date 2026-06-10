"""Hermetic tests for the Models API capability registration.

No network: a ``SimpleNamespace`` stands in for ``client.models.retrieve``. The
live path exists so an env-pinned newer model gets its real capabilities
instead of degrading to the conservative unknown-model defaults.
"""
from __future__ import annotations

from types import SimpleNamespace

from drawing_takeoff.core import api_config


def _models_client(record=None, *, fail: bool = False):
    def retrieve(model_id):
        if fail:
            raise RuntimeError("boom")
        return record

    return SimpleNamespace(models=SimpleNamespace(retrieve=retrieve))


def _record(*, adaptive: bool = True, effort: bool = True, xhigh: bool = True):
    return SimpleNamespace(
        max_tokens=128_000,
        max_input_tokens=1_000_000,
        capabilities={
            "thinking": {"types": {"adaptive": {"supported": adaptive}}},
            "effort": {"supported": effort, "xhigh": {"supported": xhigh}},
        },
    )


def test_unknown_model_registers_from_models_api():
    model = "claude-test-live-1"
    api_config.ensure_model_registered(model, client=_models_client(_record()))
    caps = api_config.model_capabilities(model)
    assert caps.supports_adaptive_thinking and caps.supports_effort and caps.supports_effort_xhigh
    assert caps.context_window == 1_000_000 and caps.max_output_tokens == 128_000
    # the 300k batch beta is not discoverable via the Models API -> stays off
    assert caps.supports_extended_output_beta is False


def test_lookup_failure_degrades_and_is_not_retried():
    model = "claude-test-live-2"
    calls: list[str] = []
    client = _models_client(fail=True)
    orig = client.models.retrieve

    def counting(model_id):
        calls.append(model_id)
        return orig(model_id)

    client.models.retrieve = counting
    api_config.ensure_model_registered(model, client=client)
    api_config.ensure_model_registered(model, client=client)
    assert calls == [model]  # one attempt per process, even on failure
    assert api_config.model_capabilities(model) == api_config._DEFAULT_CAPABILITIES


def test_client_without_models_api_does_not_burn_the_attempt():
    model = "claude-test-live-3"
    api_config.ensure_model_registered(model, client=SimpleNamespace())  # e.g. a test fake
    assert model not in api_config._LIVE_LOOKUP_ATTEMPTED
    # a later, capable client can still resolve it
    api_config.ensure_model_registered(model, client=_models_client(_record()))
    assert api_config.model_capabilities(model).supports_effort


def test_known_models_never_hit_the_models_api():
    client = _models_client(fail=True)  # would raise if consulted
    api_config.ensure_model_registered(api_config.MODEL_OPUS_48, client=client)
    api_config.ensure_model_registered(api_config.MODEL_SONNET_46, client=client)


def test_xhigh_clamp_consults_the_capability_registry():
    # static entries: Sonnet clamps xhigh -> high, Opus keeps it
    assert api_config._clamp_effort_for_model("xhigh", api_config.MODEL_SONNET_46) == "high"
    assert api_config._clamp_effort_for_model("xhigh", api_config.MODEL_OPUS_48) == "xhigh"
    # a live-registered model without xhigh support clamps the same way
    model = "claude-test-live-4"
    api_config.ensure_model_registered(model, client=_models_client(_record(xhigh=False)))
    assert api_config._clamp_effort_for_model("xhigh", model) == "high"
