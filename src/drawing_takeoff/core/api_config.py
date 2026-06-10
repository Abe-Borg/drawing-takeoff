"""Centralized Anthropic API configuration for Spec Critic.

Single place for model identifiers, per-phase output-token caps, batch
beta headers, web-search tool configuration, and request-shape policy
(prompt caching, adaptive thinking, effort).

Model identifiers may be overridden via env vars:
    DRAWING_TAKEOFF_MODEL                — review (default Opus 4.8).
    DRAWING_TAKEOFF_VERIFICATION_MODEL          — verification initial pass
                                              (default Sonnet 4.6).
    DRAWING_TAKEOFF_VERIFICATION_ESCALATION_MODEL — escalation (default Opus 4.8).
    DRAWING_TAKEOFF_TRIAGE_MODEL                — verification triage
                                              (default Haiku 4.5).
    DRAWING_TAKEOFF_LABELING_MODEL              — sheet labeling (default
                                              Sonnet 4.6).
    DRAWING_TAKEOFF_LABELING_DENSE_MODEL        — labeling of dense sheets
                                              (default Opus 4.8).
    DRAWING_TAKEOFF_LABELING_ESCALATION_MODEL   — second-look re-check of
                                              flagged networks (default
                                              Opus 4.8).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model identifiers (centralized)
# ---------------------------------------------------------------------------

MODEL_OPUS_48 = "claude-opus-4-8"
MODEL_SONNET_46 = "claude-sonnet-4-6"
MODEL_HAIKU_45 = "claude-haiku-4-5"

# Review runs on the current Opus flagship; verification routes through
# Sonnet first and reserves Opus for escalation on CRITICAL/HIGH UNVERIFIED
# findings. Defaults track the newest Opus generation (4.8). Override any of
# these via the matching ``DRAWING_TAKEOFF_*_MODEL`` env var.
REVIEW_MODEL_DEFAULT = os.environ.get("DRAWING_TAKEOFF_MODEL", MODEL_OPUS_48)
CROSS_CHECK_MODEL_DEFAULT = MODEL_SONNET_46
VERIFICATION_MODEL_DEFAULT = os.environ.get(
    "DRAWING_TAKEOFF_VERIFICATION_MODEL", MODEL_SONNET_46
)

# Model used when escalating a low-confidence/high-severity verification.
VERIFICATION_ESCALATION_MODEL = os.environ.get(
    "DRAWING_TAKEOFF_VERIFICATION_ESCALATION_MODEL", MODEL_OPUS_48
)

# Verification triage pre-pass (triage.classify_findings_with_haiku) decides
# whether a finding can be locally resolved or needs web verification. The
# task is shallow classification over short inputs; Haiku fits.
TRIAGE_MODEL_DEFAULT = os.environ.get("DRAWING_TAKEOFF_TRIAGE_MODEL", MODEL_HAIKU_45)

# Sheet labeling (legend.label_networks — the M7 set-of-marks call). Sonnet 4.6
# was validated for this vision+reasoning task by the M7 probe at a fraction of
# Opus's cost. Dense sheets route to the Opus tier instead: the set-of-marks
# overlay renders at 2400px on the long edge, which Opus 4.7+ accepts at full
# resolution (2576px image cap) while Sonnet 4.6 downscales to 1568px — and the
# more numbered networks share one sheet, the more that resolution matters for
# reading the marks. The dense threshold lives at the call site (legend.py),
# which knows the network count; only the model choice is policy here.
LABELING_MODEL_DEFAULT = os.environ.get("DRAWING_TAKEOFF_LABELING_MODEL", MODEL_SONNET_46)
LABELING_DENSE_MODEL_DEFAULT = os.environ.get(
    "DRAWING_TAKEOFF_LABELING_DENSE_MODEL", MODEL_OPUS_48
)

# Second-look escalation (legend.second_look_networks): networks the first
# pass flagged are by definition the hard cases, so the close-up re-check
# defaults to the strongest tier — mirroring the verification escalation
# split (Sonnet first, Opus on what survives).
LABELING_ESCALATION_MODEL_DEFAULT = os.environ.get(
    "DRAWING_TAKEOFF_LABELING_ESCALATION_MODEL", MODEL_OPUS_48
)


# Convenience sets for output-cap dispatch. Every Opus family member shares
# the 128k output ceiling and the high-effort escalation tier, so newer Opus
# ids must be listed here too (not just in ``_MODEL_CAPABILITIES``) or they
# fall through to the Sonnet 64k ceiling / medium effort.
OPUS_MODELS = frozenset({MODEL_OPUS_48})
HAIKU_MODELS = frozenset({MODEL_HAIKU_45})


# ---------------------------------------------------------------------------
# Output-token caps
# ---------------------------------------------------------------------------

# Hard ceilings imposed by the model.
MAX_OUTPUT_TOKENS_OPUS = 128_000
MAX_OUTPUT_TOKENS_SONNET = 64_000
MAX_OUTPUT_TOKENS_HAIKU = 64_000

# Extended-output batch beta. Required header to use 300k output in batch.
BATCH_OUTPUT_BETA = "output-300k-2026-03-24"
BATCH_MAX_OUTPUT_TOKENS = 300_000

# Per-phase dynamic caps. These are intentionally lower than the hard model
# ceilings so the app does not blanket-allocate the maximum on every call.
# A single review baseline keeps findings consistent on normal-size specs.
# (Anthropic bills by actual output, so the cap is a fail-fast guard, not a
# cost lever.) The extended 300k path is gated behind the
# ``output-300k-2026-03-24`` beta header for large batch inputs only.
REVIEW_OUTPUT_CAP = 128_000              # baseline review cap
REVIEW_OUTPUT_CAP_BATCH_EXTENDED = 300_000  # batch-only, with 300k beta header
CROSS_CHECK_OUTPUT_CAP = 96_000       # cross-check needs more than verify
# Verdicts are 1-2 sentences per the verifier system prompt; 16k is a
# fail-fast guard, not a billing knob (you pay only for actual output).
VERIFICATION_OUTPUT_CAP = 16_000
# Triage emits a small array of {index, classification, reason}; 8k is more
# than enough even for a 50-finding chunk.
HAIKU_TRIAGE_OUTPUT_CAP = 8_000
# Labeling replies are a small JSON array, but adaptive thinking shares
# max_tokens with the visible output — the old 4k cap would starve the
# thinking that motivates the budget. 16k stays under the SDK's
# streaming-required threshold.
LABELING_OUTPUT_CAP = 16_000

# Token threshold above which a review uses the larger batch cap.
LARGE_REVIEW_INPUT_THRESHOLD = 200_000


# Phase identifiers. Defined here (before the phase→budget registry) so
# the registry can reference them directly. ``thinking_config_for`` and
# ``apply_thinking_config`` further below also consume these.
PHASE_REVIEW = "review"
PHASE_CROSS_CHECK = "cross_check"
PHASE_VERIFICATION = "verification"
PHASE_VERIFICATION_RETRY = "verification_retry"
PHASE_VERIFICATION_CONTINUATION = "verification_continuation"
PHASE_TRIAGE = "triage"
PHASE_LABELING = "labeling"


def output_cap_for_model(model: str, *, requested: int) -> int:
    """Clamp ``requested`` to the model's hard output ceiling."""
    if model in OPUS_MODELS:
        ceiling = MAX_OUTPUT_TOKENS_OPUS
    elif model in HAIKU_MODELS:
        ceiling = MAX_OUTPUT_TOKENS_HAIKU
    else:
        ceiling = MAX_OUTPUT_TOKENS_SONNET
    return min(requested, ceiling)


# Single registry of per-phase output budgets so verification
# retry/continuation and triage all resolve through the same lookup. Each
# phase declares its desired cap; ``phase_output_cap`` clamps that to the
# selected model's ceiling. The phase helpers below stay as thin wrappers
# so callers can keep their existing imports.
#
# Verification retry/continuation reuse the verification cap by default —
# the verdict envelope is unchanged across retries, so granting more output
# only invites the model to ramble. If a future investigation shows
# continuations need more headroom, this is the one place to tune it.
_PHASE_OUTPUT_BUDGET: dict[str, int] = {
    PHASE_REVIEW: REVIEW_OUTPUT_CAP,
    PHASE_CROSS_CHECK: CROSS_CHECK_OUTPUT_CAP,
    PHASE_VERIFICATION: VERIFICATION_OUTPUT_CAP,
    PHASE_VERIFICATION_RETRY: VERIFICATION_OUTPUT_CAP,
    PHASE_VERIFICATION_CONTINUATION: VERIFICATION_OUTPUT_CAP,
    PHASE_TRIAGE: HAIKU_TRIAGE_OUTPUT_CAP,
    PHASE_LABELING: LABELING_OUTPUT_CAP,
}


def phase_output_cap(phase: str, *, model: str) -> int:
    """Return the centralized per-phase max_tokens budget for ``model``.

    Every phase resolves its output cap here so review, batch review,
    cross-check, verification, verification retry, verification continuation,
    and triage all share one registry. Unknown phases fall back to the
    verification cap, the most conservative value in the registry — a future
    phase that forgets to register loses headroom instead of accidentally
    inheriting the 128k review cap.
    """
    requested = _PHASE_OUTPUT_BUDGET.get(phase, VERIFICATION_OUTPUT_CAP)
    return output_cap_for_model(model, requested=requested)


def triage_max_tokens(*, model: str = TRIAGE_MODEL_DEFAULT) -> int:
    return phase_output_cap(PHASE_TRIAGE, model=model)


def review_max_tokens(*, model: str = REVIEW_MODEL_DEFAULT, allow_extended_output: bool = False) -> int:
    """Return a per-call max_tokens for a review request.

    Review runs exclusively through the Message Batches API, so every
    request shares the same baseline cap on normal-size specs.
    ``allow_extended_output`` selects the 300k batch-only path; the beta
    header is checked at the call site by
    :func:`assert_extended_output_allowed`.
    """
    if allow_extended_output:
        return min(BATCH_MAX_OUTPUT_TOKENS, REVIEW_OUTPUT_CAP_BATCH_EXTENDED)
    return phase_output_cap(PHASE_REVIEW, model=model)


def cross_check_max_tokens(*, model: str = CROSS_CHECK_MODEL_DEFAULT) -> int:
    return phase_output_cap(PHASE_CROSS_CHECK, model=model)


def verification_max_tokens(*, model: str = VERIFICATION_MODEL_DEFAULT, phase: str = PHASE_VERIFICATION) -> int:
    """Return a per-call max_tokens for a verification request.

    ``phase`` defaults to ``PHASE_VERIFICATION``; pass
    ``PHASE_VERIFICATION_RETRY`` or ``PHASE_VERIFICATION_CONTINUATION`` to
    pick up retry-specific or continuation-specific budgets from the central
    registry. Today all three resolve to the same cap; the parameter exists
    so a future tuning pass touches one place.
    """
    return phase_output_cap(phase, model=model)


def assert_extended_output_allowed(
    *, max_tokens: int, betas: Iterable[str] | None, model: str | None = None
) -> None:
    """Guard against extended output without the required beta header.

    The Anthropic API rejects output above a model's baseline ceiling when
    the extended-output beta is not set, but the failure surfaces deep in the
    request lifecycle. Plan Sprint 2 item 8: fail fast at the call site instead.

    The threshold is the *selected model's* baseline (non-beta) output ceiling
    (TRUST_AUDIT P2-3), derived from the single :func:`output_cap_for_model`
    source of truth — Opus 128k, Sonnet/Haiku 64k. Passing ``model`` makes the
    guard correct for Sonnet (whose 64k baseline is below the old hardcoded
    128k threshold, so a 64k–128k Sonnet request without the beta would have
    slipped past). When ``model`` is omitted the guard falls back to the
    highest baseline ceiling (Opus 128k) so it never *over*-fires on a
    legitimate sub-ceiling request — the API stays the backstop for that case.
    """
    ceiling = (
        output_cap_for_model(model, requested=BATCH_MAX_OUTPUT_TOKENS)
        if model
        else MAX_OUTPUT_TOKENS_OPUS
    )
    if max_tokens <= ceiling:
        return
    beta_set = set(betas or ())
    if BATCH_OUTPUT_BETA not in beta_set:
        raise ValueError(
            f"Requested max_tokens={max_tokens:,} exceeds the baseline output "
            f"ceiling ({ceiling:,}" + (f" for model '{model}'" if model else "") + ") "
            f"and requires beta header '{BATCH_OUTPUT_BETA}'. Refusing to submit without it."
        )


# ---------------------------------------------------------------------------
# Model capability policy
# ---------------------------------------------------------------------------
#
# Whitelist-style registry of per-model capabilities. The Anthropic API
# rejects requests that include feature parameters the selected model does
# not support — most notably ``thinking`` against Haiku 4.5, which produces
# an API error.
#
# To add a new model: register it in ``_MODEL_CAPABILITIES``. Unknown model
# IDs fall through to ``_DEFAULT_CAPABILITIES``, which disables every
# capability flag — intentional. Stripping a feature from a future model is
# strictly safer than sending an invalid request that fails deep in the
# request lifecycle. The degradation is no longer silent, though: an
# unrecognized id logs one WARNING (see :func:`model_capabilities`) so a
# stale whitelist that quietly under-powers a newer/better model is visible
# to the operator rather than hidden.


@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model feature support. Drives request-shape decisions."""

    supports_adaptive_thinking: bool
    max_output_tokens: int
    supports_extended_output_beta: bool  # 300k batch-only beta header
    context_window: int
    # Whether the model accepts ``output_config.effort``. The
    # parameter controls token eagerness and tool-call behavior. Sending
    # it to an unsupported model returns an API error, so the policy in
    # :func:`effort_config_for` must consult this flag before attaching
    # the field. Default ``False`` so unknown models silently omit it.
    supports_effort: bool = False
    # Whether the model accepts the ``xhigh`` effort level (Opus-tier only
    # today). :func:`_clamp_effort_for_model` consults this so a phase that
    # defaults to ``xhigh`` degrades to ``high`` instead of a 400 on models
    # without it. Default ``False`` — clamping is always safe.
    supports_effort_xhigh: bool = False


_MODEL_CAPABILITIES: dict[str, ModelCapabilities] = {
    MODEL_OPUS_48: ModelCapabilities(
        # Claude Opus 4.8 capability profile per Anthropic's "What's new in
        # Claude Opus 4.8" and the models overview: 1M-token context window on
        # the Claude API, 128k max output, the ``output-300k-2026-03-24`` batch
        # beta (shared with Sonnet 4.6), extended/adaptive thinking, and the
        # ``effort`` parameter (default high). Registered explicitly so
        # selecting it via ``DRAWING_TAKEOFF_*_MODEL`` unlocks full capabilities
        # instead of falling through to the conservative unknown-model defaults.
        supports_adaptive_thinking=True,
        max_output_tokens=MAX_OUTPUT_TOKENS_OPUS,
        supports_extended_output_beta=True,
        context_window=1_000_000,
        supports_effort=True,
        supports_effort_xhigh=True,
    ),
    MODEL_SONNET_46: ModelCapabilities(
        supports_adaptive_thinking=True,
        max_output_tokens=MAX_OUTPUT_TOKENS_SONNET,
        # Sonnet 4.6 supports the ``output-300k-2026-03-24`` beta
        # on Message Batches. The prior ``False`` value predated that
        # capability rollout and forced the batch path to gate extended
        # output by Opus-only family membership.
        supports_extended_output_beta=True,
        context_window=1_000_000,
        supports_effort=True,
    ),
    MODEL_HAIKU_45: ModelCapabilities(
        # Anthropic models overview lists Haiku 4.5 without adaptive
        # thinking support; sending ``thinking`` to it returns an API error.
        supports_adaptive_thinking=False,
        max_output_tokens=MAX_OUTPUT_TOKENS_HAIKU,
        supports_extended_output_beta=False,
        context_window=200_000,
        # The Anthropic effort docs list Haiku 4.5 without effort support.
        # Omit ``output_config.effort`` for Haiku to keep request shapes
        # safe across model swaps (e.g. triage).
        supports_effort=False,
    ),
}


# Unknown models: every capability flag defaults to False so we never
# construct an invalid request payload. Output cap defaults to the Sonnet
# ceiling, the most conservative of the supported models that still leaves
# room for a meaningful response.
_DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_adaptive_thinking=False,
    max_output_tokens=MAX_OUTPUT_TOKENS_SONNET,
    supports_extended_output_beta=False,
    context_window=200_000,
    supports_effort=False,
)


# Falling through to ``_DEFAULT_CAPABILITIES`` keeps a misconfigured
# ``DRAWING_TAKEOFF_*_MODEL`` from constructing an invalid request, but the
# degradation used to be *silent*: an operator who pinned a newer/better model
# than the whitelist knew about got quietly smaller requests (no extended
# thinking, no effort tuning, a 64k output cap instead of 128k/300k, a 200k
# context window instead of 1M, no batch extended-output beta) with no signal
# anywhere. We now emit one WARNING per unrecognized id so the quality loss is
# visible. Deduped via a module-level set because ``model_capabilities`` sits
# on a per-request hot path and must not spam the log.
_WARNED_UNKNOWN_MODELS: set[str] = set()


def _warn_unknown_model(model: str) -> None:
    """Emit a one-time WARNING that ``model`` fell through to safe defaults."""
    if model in _WARNED_UNKNOWN_MODELS:
        return
    _WARNED_UNKNOWN_MODELS.add(model)
    _log.warning(
        "Model id %r is not in the capability whitelist (_MODEL_CAPABILITIES "
        "in src/core/api_config.py); degrading to conservative defaults: no "
        "adaptive thinking, no effort tuning, %s-token output cap, %s-token "
        "context window, no 300k extended-output beta. If this is a "
        "known-good model, add it to the whitelist to unlock its full "
        "capabilities.",
        model,
        f"{_DEFAULT_CAPABILITIES.max_output_tokens:,}",
        f"{_DEFAULT_CAPABILITIES.context_window:,}",
    )


def model_capabilities(model: str) -> ModelCapabilities:
    """Return the capability record for ``model`` (or safe defaults).

    Known ids resolve from ``_MODEL_CAPABILITIES``. Unknown ids fall through
    to ``_DEFAULT_CAPABILITIES`` *and* trigger a one-time WARNING (see
    :func:`_warn_unknown_model`) so the conservative degradation is never
    silent — the failure mode the trust audit (P0-3) flagged, where a
    deliberately-selected newer model gets quietly worse requests.
    """
    caps = _MODEL_CAPABILITIES.get(model)
    if caps is not None:
        return caps
    _warn_unknown_model(model)
    return _DEFAULT_CAPABILITIES


# ---------------------------------------------------------------------------
# Models API registration (live capability discovery)
# ---------------------------------------------------------------------------
#
# The static whitelist above goes stale the day Anthropic ships a model: an
# operator who pins a newer id via ``DRAWING_TAKEOFF_*_MODEL`` used to get the
# conservative defaults (no thinking, no effort, 64k cap) plus a warning. The
# Models API (``client.models.retrieve``) reports the real capability tree, so
# an unknown id is now resolved live — once per process — and registered with
# its actual capabilities. The static table stays as the offline fallback and
# the source of truth for what the API can't report (the 300k batch beta).

# Ids we already asked the Models API about (hit or miss) — the lookup sits on
# the per-request path, so it must run at most once per id per process.
_LIVE_LOOKUP_ATTEMPTED: set[str] = set()


def _capabilities_from_models_api(info) -> ModelCapabilities:
    """Map a Models API record onto :class:`ModelCapabilities`.

    ``info.capabilities`` is the API's untyped nested dict with a
    ``supported`` bool at each leaf; a missing branch reads as unsupported.
    The 300k extended-output batch beta is not discoverable here, so it stays
    off — strictly safer (the request simply won't ask for extended output).
    """
    tree = dict(info.capabilities or {})

    def supported(*path: str) -> bool:
        node: object = tree
        for key in (*path, "supported"):
            if not isinstance(node, dict):
                return False
            node = node.get(key)
        return bool(node)

    return ModelCapabilities(
        supports_adaptive_thinking=supported("thinking", "types", "adaptive"),
        max_output_tokens=int(info.max_tokens),
        supports_extended_output_beta=False,
        context_window=int(info.max_input_tokens),
        supports_effort=supported("effort"),
        supports_effort_xhigh=supported("effort", "xhigh"),
    )


def ensure_model_registered(model: str, *, client=None) -> None:
    """Best-effort: resolve an unknown ``model`` via the Models API.

    Known ids and previously-attempted ids return immediately, so the call is
    a set lookup on the hot path. ``client`` is duck-typed; one without a
    ``models.retrieve`` (e.g. the test fakes) is skipped silently and the id
    falls through to the existing unknown-model warning path. A lookup failure
    likewise warns and falls back — capability discovery must never sink a
    request.
    """
    if model in _MODEL_CAPABILITIES or model in _LIVE_LOOKUP_ATTEMPTED or client is None:
        return
    retrieve = getattr(getattr(client, "models", None), "retrieve", None)
    if not callable(retrieve):
        return
    _LIVE_LOOKUP_ATTEMPTED.add(model)
    try:
        caps = _capabilities_from_models_api(retrieve(model))
    except Exception as exc:
        _log.warning(
            "Models API lookup for %r failed (%s); the id will degrade to the "
            "conservative unknown-model defaults.", model, exc,
        )
        return
    _MODEL_CAPABILITIES[model] = caps
    _log.info("Registered %r from the Models API: %s", model, caps)


def model_supports_adaptive_thinking(model: str) -> bool:
    """Whether ``model`` accepts the ``thinking`` request parameter."""
    return model_capabilities(model).supports_adaptive_thinking


def model_supports_effort(model: str) -> bool:
    """Whether ``model`` accepts the ``output_config.effort`` parameter.

    Callers MUST check this before attaching
    ``output_config={"effort": ...}`` to a request. Unsupported models
    (Haiku 4.5, unknown / future models) silently omit the field — the
    field is opt-in per model, so omitting it is always safe.
    """
    return model_capabilities(model).supports_effort


def model_supports_extended_output_beta(model: str) -> bool:
    """Whether ``model`` is eligible for the 300k batch-output beta.

    The extended-output decision must read from the capability
    registry rather than testing ``model in OPUS_MODELS``. Sonnet 4.6
    supports the ``output-300k-2026-03-24`` beta on Message Batches,
    which the family-style check incorrectly excluded.
    """
    return model_capabilities(model).supports_extended_output_beta


# Phase identifiers (declared above so the phase→budget registry can use
# them) gate per-phase request decisions. ``_PHASES_NO_THINKING`` is the
# extension point for phases that should never request thinking regardless
# of model capability — currently only the Haiku triage classifier, which
# is a shallow batch-classification pass.
_PHASES_NO_THINKING: frozenset[str] = frozenset({PHASE_TRIAGE})


def thinking_config_for(*, model: str, phase: str) -> dict | None:
    """Return the ``thinking`` request parameter for ``(model, phase)``.

    Returns ``None`` when the parameter should be omitted entirely —
    either the phase opts out, or the model does not support adaptive
    thinking. Callers should branch on ``is None``; the Anthropic API
    rejects ``thinking=null``.
    """
    if phase in _PHASES_NO_THINKING:
        return None
    if not model_supports_adaptive_thinking(model):
        return None
    return {"type": "adaptive"}


def apply_thinking_config(kwargs: dict, *, model: str, phase: str) -> dict:
    """Insert the ``thinking`` key into ``kwargs`` only when applicable.

    Mutates and returns ``kwargs`` for fluent use. The key is omitted
    entirely (not set to ``None``) when thinking is not applicable, because
    the Anthropic API rejects ``thinking=null``.
    """
    config = thinking_config_for(model=model, phase=phase)
    if config is not None:
        kwargs["thinking"] = config
    return kwargs


# ---------------------------------------------------------------------------
# Output-config effort policy
# ---------------------------------------------------------------------------
#
# The Anthropic API accepts an ``output_config.effort`` parameter on
# supported models. The value tunes how eagerly the model produces tokens
# and how aggressively it pursues tool calls. The documented levels are
# ``low`` / ``medium`` / ``high`` / ``xhigh`` (plus ``max``). The review and
# cross-check phases use ``xhigh`` — Anthropic recommends it as the starting
# point for coding/agentic work on Opus 4.8, and per-spec review is the
# deepest-reasoning phase in the pipeline. We still don't use ``max`` (it
# overshoots without a measured benefit for this workload), and verification
# stays at medium/high so the verdict envelope doesn't balloon.
#
# ``xhigh`` is Opus-4.8-only. Sonnet 4.6's supported set is
# ``{low, medium, high, max}`` — it rejects ``xhigh`` at submit with a 400
# ("This model does not support effort level 'xhigh'"). So ``supports_effort``
# being a coarse boolean is not enough: a phase that defaults to ``xhigh`` but
# runs on Sonnet (the cross-check phase always does; review does when
# ``DRAWING_TAKEOFF_MODEL`` is overridden) must clamp down to ``high`` or
# the request fails. :func:`effort_config_for` does this clamp via
# :func:`_clamp_effort_for_model`.
#
# Effort is a request-policy decision, not a prompt one. Centralizing it
# here keeps every request site (review / batch review / cross-check /
# verification / retry / continuation) reaching for the same lever via
# :func:`apply_effort_config`. Unsupported models silently omit the
# parameter via :func:`model_supports_effort`.
#
# Default policy:
#
# - Sonnet verification (PHASE_VERIFICATION{,_RETRY,_CONTINUATION}): medium.
# - Opus verification (i.e. escalation): high.
# - Opus deep review (PHASE_REVIEW, PHASE_CROSS_CHECK): xhigh.
# - Sonnet deep review (cross-check always; review when overridden): xhigh is
#   Opus-only, so the clamp drops it to high.
# - Triage (Haiku): omit (Haiku does not support effort).
# - Unknown model: omit.

EFFORT_MEDIUM = "medium"
EFFORT_HIGH = "high"
EFFORT_XHIGH = "xhigh"

# Phases whose request paths route through ``output_config.effort``. Triage
# is intentionally omitted — it defaults to Haiku which does not support
# effort, and the workload is a small classification pass that does not
# benefit from elevated effort.
_PHASE_DEFAULT_EFFORT: dict[str, str] = {
    PHASE_REVIEW: EFFORT_XHIGH,
    PHASE_CROSS_CHECK: EFFORT_XHIGH,
    PHASE_VERIFICATION: EFFORT_MEDIUM,
    PHASE_VERIFICATION_RETRY: EFFORT_MEDIUM,
    PHASE_VERIFICATION_CONTINUATION: EFFORT_MEDIUM,
    # Labeling is a single-shot vision classification — ``high`` is Anthropic's
    # recommended minimum for intelligence-sensitive work and is accepted by
    # both the Sonnet default and the Opus dense tier (no clamp needed);
    # ``xhigh`` is tuned for coding/agentic loops, not one structured reply.
    PHASE_LABELING: EFFORT_HIGH,
}

# Verification phases get the model-aware bump: Opus on verification is
# always the escalation tier, so the policy lifts effort to ``high``.
_VERIFICATION_PHASES: frozenset[str] = frozenset(
    {
        PHASE_VERIFICATION,
        PHASE_VERIFICATION_RETRY,
        PHASE_VERIFICATION_CONTINUATION,
    }
)

# Effort levels gated to models whose capability record carries
# ``supports_effort_xhigh`` (the Opus tier today). Sonnet 4.6's supported set
# is ``{low, medium, high, max}``; it rejects ``xhigh`` at submit with a 400
# ("This model does not support effort level 'xhigh'"). Membership in this set
# is the trigger for :func:`_clamp_effort_for_model` to downgrade to ``high``.
_XHIGH_EFFORT_LEVELS: frozenset[str] = frozenset({EFFORT_XHIGH})


def _clamp_effort_for_model(level: str, model: str) -> str:
    """Clamp an effort ``level`` down to what ``model`` accepts.

    ``xhigh`` is accepted only where the capability registry says so (static
    whitelist or a Models API registration); everywhere else it falls back to
    ``high`` — the deepest level Sonnet 4.6 accepts (we don't use ``max``).
    Every other level passes through unchanged. This is what keeps the
    cross-check phase (``xhigh`` default, but always Sonnet) from 400-ing at
    submit, and protects a Sonnet-overridden review phase the same way.
    """
    if level in _XHIGH_EFFORT_LEVELS and not model_capabilities(model).supports_effort_xhigh:
        return EFFORT_HIGH
    return level


def effort_config_for(*, model: str, phase: str) -> dict | None:
    """Return the ``output_config`` dict for ``(model, phase)``, or ``None``.

    Returns ``None`` (i.e. "omit the field") when:

    - the model does not support effort (Haiku, unknown / future models),
    - the phase has no registered default (triage — defaults to Haiku,
      which already short-circuits above).

    Otherwise returns ``{"effort": <level>}`` where the level is ``high``
    for Opus on a verification phase (the escalation tier) or the phase
    default from :data:`_PHASE_DEFAULT_EFFORT`, clamped to what ``model``
    supports (``xhigh`` → ``high`` on non-Opus models — see
    :func:`_clamp_effort_for_model`).
    """
    if not model_supports_effort(model):
        return None

    if phase in _VERIFICATION_PHASES:
        # Opus on a verification phase is the escalation tier — every
        # initial verification call routes to Sonnet by default.
        if model in OPUS_MODELS:
            return {"effort": EFFORT_HIGH}
        return {"effort": EFFORT_MEDIUM}

    level = _PHASE_DEFAULT_EFFORT.get(phase)
    if level is None:
        return None
    return {"effort": _clamp_effort_for_model(level, model)}


def apply_effort_config(kwargs: dict, *, model: str, phase: str) -> dict:
    """Insert ``output_config`` into ``kwargs`` only when applicable.

    Mutates and returns ``kwargs`` for fluent use. The key is omitted
    entirely (not set to ``None``) when effort is not applicable, because
    the Anthropic API rejects ``output_config=null``.

    Mirrors :func:`apply_thinking_config` so request builders pair the
    two helpers the same way per directive 4 ("Pair effort decisions
    with thinking decisions where appropriate").
    """
    config = effort_config_for(model=model, phase=phase)
    if config is not None:
        kwargs["output_config"] = config
    return kwargs


# ---------------------------------------------------------------------------
# Prompt caching (centralized phase-aware policy)
# ---------------------------------------------------------------------------
#
# Each phase declares whether its system prompt and tool list are stable /
# large / repeated enough to benefit from caching. Caching is enabled for
# high-value phases (review, batch review, cross-check, verification +
# retry/continuation) and disabled for triage where the prompt is below
# the Anthropic cache minimum (2048 tokens for Haiku) so a cache write
# would be paid for nothing.


@dataclass(frozen=True)
class CachePolicy:
    """Per-phase cache policy.

    ``cache_system`` and ``cache_tools`` independently control whether the
    system prompt and the trailing tool block carry ``cache_control``
    breakpoints.
    """

    cache_system: bool
    cache_tools: bool

    @property
    def caches_anything(self) -> bool:
        return self.cache_system or self.cache_tools


_DEFAULT_PHASE_CACHE_POLICY = CachePolicy(cache_system=True, cache_tools=True)

_PHASE_CACHE_POLICY: dict[str, CachePolicy] = {
    PHASE_REVIEW: CachePolicy(cache_system=True, cache_tools=True),
    PHASE_CROSS_CHECK: CachePolicy(cache_system=True, cache_tools=True),
    PHASE_VERIFICATION: CachePolicy(cache_system=True, cache_tools=True),
    PHASE_VERIFICATION_RETRY: CachePolicy(cache_system=True, cache_tools=True),
    PHASE_VERIFICATION_CONTINUATION: CachePolicy(cache_system=True, cache_tools=True),
    # Triage: ~375-token system prompt called in batches of up to 20,
    # below the 2048-token Haiku cache minimum so repeated calls cannot
    # hit. Skip caching to avoid the cache-write cost.
    PHASE_TRIAGE: CachePolicy(cache_system=False, cache_tools=False),
    # Labeling: ~500-token system prompts, below the cache minimum on both
    # the Sonnet default (2048) and the Opus dense tier (4096) — a cache
    # write would be paid for nothing.
    PHASE_LABELING: CachePolicy(cache_system=False, cache_tools=False),
}


def cache_policy_for(phase: str | None) -> CachePolicy:
    """Return the per-phase :class:`CachePolicy`.

    Unknown phases fall back to the conservative default (cache both
    system prompt and tools).
    """
    if phase is None:
        return _DEFAULT_PHASE_CACHE_POLICY
    return _PHASE_CACHE_POLICY.get(phase, _DEFAULT_PHASE_CACHE_POLICY)


def _cache_control_block() -> dict:
    """Return the standard 1-hour ephemeral cache_control block.

    Spec Critic batch + verification waves run for 30 minutes to several
    hours, well beyond the 5-minute default ephemeral cache TTL. The
    1-hour TTL costs 2x the cache write but typically pays back inside
    the second wave of a batch verification cycle, where the same system
    prompt is sent hundreds of times.
    """
    return {"type": "ephemeral", "ttl": "1h"}


def system_prompt_with_cache(prompt: str, *, phase: str | None = None):
    """Return a system payload with a cache breakpoint when policy permits.

    Per the Anthropic prompt-caching docs, including the same cache_control
    blocks in every request in a batch lets later items hit the cache
    created by earlier items.
    """
    policy = cache_policy_for(phase)
    if not policy.cache_system:
        return prompt
    return [
        {
            "type": "text",
            "text": prompt,
            "cache_control": _cache_control_block(),
        }
    ]


def tools_with_cache(tools: list[dict], *, phase: str | None = None) -> list[dict]:
    """Attach a cache breakpoint to the last tool definition.

    Tool schemas are stable across verification calls. Caching the trailing
    tool block lets the rest of the request (system prompt + tool defs)
    share one cache prefix. The system prompt has its own breakpoint via
    :func:`system_prompt_with_cache`, so changing only a tool definition
    invalidates only the tools-level cache entry.

    ``phase`` selects the per-phase policy. When the policy disables tool
    caching for the phase (e.g. triage where the prompt is below the cache
    minimum), the tool list is returned unchanged.
    """
    if not tools:
        return tools
    policy = cache_policy_for(phase)
    if not policy.cache_tools:
        return tools
    last = dict(tools[-1])
    last["cache_control"] = _cache_control_block()
    return [*tools[:-1], last]


# ---------------------------------------------------------------------------
# Service tier (priority capacity)
# ---------------------------------------------------------------------------


def batch_service_tier() -> str:
    """Return the ``service_tier`` parameter for batch request params.

    ``auto`` opts batch requests into priority capacity when available,
    falling back to standard.
    """
    return "auto"


# ---------------------------------------------------------------------------
# Anthropic token-counting preflight
# ---------------------------------------------------------------------------

def token_count_preflight_enabled() -> bool:
    """Whether to call Anthropic's count_tokens endpoint before submission.

    Always True. The GUI also runs an exact count for the largest spec
    when the file list changes; the pipeline call here is the moment-of-
    truth guard before a real submission.
    """
    return True


# ---------------------------------------------------------------------------
# Web-search tool configuration
# ---------------------------------------------------------------------------

# Source-quality blocklist for ``web_search_20260209``. Mixing
# ``allowed_domains`` and ``blocked_domains`` is not supported by the tool,
# so this is blocked-only; California priority sources are documented in the
# verifier system prompt as guidance rather than encoded as an allow-list.
#
# Domains are listed bare (no scheme/path) and the tool treats each entry as
# "this apex and every subdomain", so adding ``simple.wikipedia.org`` when
# ``wikipedia.org`` is already on the list adds nothing.
#
# Categories (kept as inline comment groups so the intent of each line is
# obvious; do *not* hand-sort across categories without checking that no
# entry's category interpretation changes):
#   - Aggregators / Q&A: forums where contractor-grade evidence is rare.
#   - LLM-assistant outputs: another model's answer is not a citable source.
#   - Trade forums: useful peer chatter, not authoritative for code.
#   - DIY / home-improvement content farms.
#   - Social: unsuitable for a defensible engineering review.
#   - General encyclopedias: tertiary sources.
#
# TODO: explore a category-based blocking helper so each entry is annotated
# with its category and the report can explain *why* a citation was rejected
# (deferred for now; the immediate fix here is just deduplicating the
# obvious subdomain overlap). Any change to this list should be exercised
# against the verifier's grounding tests in
# ``tests/test_source_grounding_invariant.py``.
_WEB_SEARCH_BLOCKED_DOMAINS = [
    # Aggregators / Q&A
    "reddit.com", "quora.com", "medium.com",
    "stackexchange.com", "stackoverflow.com",
    "answers.yahoo.com", "fixya.com",
    # LLM-assistant outputs
    "chatgpt.com", "perplexity.ai", "openai.com", "gemini.google.com",
    "claude.ai", "you.com", "phind.com", "copilot.microsoft.com",
    "poe.com", "character.ai", "jasper.ai", "writesonic.com",
    # Trade forums (peer chatter, not authoritative for code compliance)
    "diychatroom.com", "forums.jlconline.com", "hvac-talk.com",
    "inspectionnews.net", "inspectorsforum.com", "contractortalk.com",
    # DIY / home-improvement / lead-gen content farms
    "doityourself.com", "homeadvisor.com", "thumbtack.com", "angi.com",
    "ehow.com", "wikihow.com", "about.com", "thespruce.com", "bobvila.com",
    "familyhandyman.com", "hunker.com", "sapling.com", "reference.com",
    "leaf.tv", "sciencing.com", "bizfluent.com", "pocketsense.com",
    # Social
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "linkedin.com", "pinterest.com", "youtube.com", "threads.net",
    # General encyclopedias (tertiary). ``wikipedia.org`` already covers
    # every subdomain (``simple.wikipedia.org``, ``en.wikipedia.org``, ...).
    "wikipedia.org", "britannica.com",
]

# Fallback budget for severities outside the known set.
DEFAULT_VERIFICATION_MAX_USES = 5

# Per-severity search budgets. High-stakes claims get more rope; editorial
# gripes get less. Applied identically to real-time and batch verification
# paths so the budget shape doesn't depend on which mode you ran in.
_SEVERITY_MAX_USES: dict[str, int] = {
    "CRITICAL": 8,
    "HIGH": 7,
    "MEDIUM": 5,
    "GRIPES": 3,
}


def web_search_max_uses_for_severity(severity: str | None) -> int:
    """Return the per-severity web_search budget.

    Falls back to ``DEFAULT_VERIFICATION_MAX_USES`` for unknown severities so
    a misclassified finding still gets a reasonable budget.
    """
    sev = (severity or "").strip().upper()
    return _SEVERITY_MAX_USES.get(sev, DEFAULT_VERIFICATION_MAX_USES)


def build_web_search_tool(*, max_uses: int = DEFAULT_VERIFICATION_MAX_USES) -> dict:
    return {
        "type": "web_search_20260209",
        "name": "web_search",
        "blocked_domains": list(_WEB_SEARCH_BLOCKED_DOMAINS),
        "max_uses": max_uses,
        "user_location": {
            "type": "approximate",
            "country": "US",
            "region": "California",
        },
    }


# ---------------------------------------------------------------------------
# Web-fetch tool configuration
# ---------------------------------------------------------------------------
#
# The ``web_fetch_20260209`` server tool is the companion to ``web_search``:
# it pulls the full text of a previously-seen URL (URLs are required to have
# appeared in a prior web_search result block in the same conversation
# context, so the model cannot fetch arbitrary URLs it invented). Per
# Anthropic's pricing docs, web_fetch carries no per-request surcharge —
# the caller pays only for the tokens the fetched content consumes — so
# the safety knob here is ``max_uses`` plus ``max_content_tokens``, not a
# billing rate.
#
# Used by STANDARD_REASONING and DEEP_REASONING verification modes only;
# STRICT_STRUCTURED / LOCAL_SKIP intentionally omit the tool because those
# modes are explicitly cheap/narrow and don't benefit from a deep dive into
# a single source page.

# Per-request fetch budget. Lower than the search budget by design — a
# verification call typically needs at most one or two full-page fetches
# to confirm a borderline claim; more than that is a sign the model is
# spinning rather than converging.
DEFAULT_VERIFICATION_MAX_FETCHES = 3

# Truncation ceiling on fetched-page content. Large code-publisher pages
# (up.codes / iccsafe.org / nfpa.org) can easily exceed 100k tokens of
# rendered text; we cap at 50k so a single fetch cannot blow the
# verification input window. The model gets enough context to find the
# clause it cares about without forcing the verifier to truncate the
# response.
WEB_FETCH_MAX_CONTENT_TOKENS = 50_000


def build_web_fetch_tool(*, max_uses: int = DEFAULT_VERIFICATION_MAX_FETCHES) -> dict:
    """Build the web_fetch server-tool dict for a verification request.

    Tool type pinned to ``web_fetch_20260209`` per Anthropic's web-fetch
    server-tool spec. Web fetch is generally available and needs no
    ``anthropic-beta`` header — the tool dict alone enables it, and sending a
    (retired) beta value such as ``web-fetch-2026-02-09`` is rejected with
    HTTP 400 ``invalid_request_error``.

    The ``citations`` field is enabled so cited URLs land in the
    assistant message's source-grounding partition the same way web_search
    citations do; ``max_content_tokens`` caps the truncation length so
    one fetch on a giant code-publisher page cannot dominate the verifier
    response window. ``blocked_domains`` mirrors the web_search blocklist
    so the two tools share one source-quality policy — a domain we won't
    search is a domain we won't fetch either.
    """
    return {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
        "blocked_domains": list(_WEB_SEARCH_BLOCKED_DOMAINS),
        "max_uses": max_uses,
        "citations": {"enabled": True},
        "max_content_tokens": WEB_FETCH_MAX_CONTENT_TOKENS,
    }


# Web fetch is generally available and takes NO ``anthropic-beta`` header —
# the tool dict above is sufficient to enable it. The verification request
# builder therefore attaches no beta header for web_fetch. Sending a retired
# beta value such as ``web-fetch-2026-02-09`` is rejected by the API with
# HTTP 400 ``invalid_request_error: Unexpected value(s) ... for the
# anthropic-beta header`` — an unrecognized beta value is not silently
# ignored, so it must not be sent at all.


# ---------------------------------------------------------------------------
# Cache-token usage extraction (for diagnostics)
# ---------------------------------------------------------------------------

def extract_cache_usage(usage) -> dict[str, int]:
    """Pull cache-related fields off an Anthropic usage object.

    Returns a dict with keys ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` (zero when absent). The Anthropic SDK
    exposes these on ``Message.usage`` when prompt caching is in effect.
    """
    if usage is None:
        return {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    return {
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    }


# ---------------------------------------------------------------------------
# Cache diagnostics (beta, opt-in observability)
# ---------------------------------------------------------------------------
#
# The ``cache-diagnosis-2026-04-07`` beta lets a request carry a
# ``diagnostics.previous_message_id`` and receive a ``diagnostics`` object on
# the response that fingerprints the current and previous request and reports
# the first point of prompt-prefix divergence — i.e. *why* a cache hit did not
# occur. It is a debugging aid for the cache-breakpoint-stability invariant
# this app cares about, NOT a request-shape change, so it stays default-off and
# is requested only when an operator is actively investigating a miss.
#
# Constraints worth remembering at the call site:
#   - First-party Claude API only (unavailable on Bedrock / Vertex).
#   - Needs a *previous* message id to diff against, so it produces signal only
#     on sequential same-prefix synchronous calls (the verification
#     continuation loop), never on the Batch API (batch items have no prior
#     message id to reference).

ENV_CACHE_DIAGNOSTICS = "DRAWING_TAKEOFF_CACHE_DIAGNOSTICS"
CACHE_DIAGNOSTICS_BETA = "cache-diagnosis-2026-04-07"

# Mirrors the disable-token convention used by the diagnostics / cache modules.
_DISABLE_TOKENS = frozenset({"0", "false", "no", "off"})


def cache_diagnostics_enabled() -> bool:
    """Whether to request prompt-cache diagnostics. Default OFF.

    Opt-in via ``DRAWING_TAKEOFF_CACHE_DIAGNOSTICS`` set to any truthy,
    non-disable value. Off by default because it is a beta, first-party-only
    observability feature that only an operator chasing a cache miss needs;
    leaving it off keeps the request byte-identical to today.
    """
    raw = os.environ.get(ENV_CACHE_DIAGNOSTICS)
    if raw is None:
        return False
    val = raw.strip().lower()
    return val != "" and val not in _DISABLE_TOKENS


def cache_diagnostics_params(
    previous_message_id: str | None,
) -> tuple[dict | None, dict | None]:
    """Return ``(extra_body, extra_headers)`` to request cache diagnostics.

    Returns ``(None, None)`` unless cache diagnostics is enabled AND a
    ``previous_message_id`` is supplied — the feature is meaningless without a
    prior message to diff against, so an isolated call cleanly no-ops.

    The body param rides the SDK ``extra_body`` seam and the beta rides
    ``extra_headers`` (``anthropic-beta``) so this stays correct on SDK
    versions that do not yet model ``diagnostics`` natively — the same
    transport-seam discipline the verification request builder already uses.
    """
    if not previous_message_id or not cache_diagnostics_enabled():
        return None, None
    extra_body = {"diagnostics": {"previous_message_id": previous_message_id}}
    extra_headers = {"anthropic-beta": CACHE_DIAGNOSTICS_BETA}
    return extra_body, extra_headers


def extract_cache_diagnostics(message) -> dict | None:
    """Pull the beta ``diagnostics`` object off a response message, if present.

    Defensive by construction: the SDK ``Message`` model is configured
    ``extra="allow"``, so an unmodeled ``diagnostics`` field round-trips as an
    attribute. Returns ``None`` when absent (the common case, or the feature
    disabled) or on any access/serialization error — a diagnostics read must
    never sink a verification.
    """
    try:
        diag = getattr(message, "diagnostics", None)
    except Exception:
        return None
    if diag is None:
        return None
    if isinstance(diag, dict):
        return diag
    dumper = getattr(diag, "model_dump", None)
    if callable(dumper):
        try:
            return dumper()
        except Exception:
            return None
    return None
