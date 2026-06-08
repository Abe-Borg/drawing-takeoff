"""
Token counting and limit management for Claude API calls.

Uses tiktoken with cl100k_base for approximate preflight estimates.
These counts are used for guardrails, not exact billing.

Token limits (v2.3.0):
    - Claude Opus 4.8 context window: 1,000,000 tokens
    - Opus 4.8 max output: 128,000 tokens
    - Sonnet 4.6 max output: 64,000 tokens
    - Per-spec recommended input limit: 500,000 tokens
      (practical limit — individual specs are reviewed one at a time)
    - Cross-check recommended input limit: ~822,000 tokens
      (1,000,000 context - 128,000 output reserve - 50,000 overhead)

The per-spec limit (RECOMMENDED_MAX) is intentionally conservative
relative to the 1M context window. Per-spec review calls send a single
spec at a time, and the token gauge in the GUI displays the largest
spec's call size against this limit.

The cross-check limit (CROSS_CHECK_RECOMMENDED_MAX) is much higher
because the cross-checker sends ALL spec content in a single call.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import tiktoken

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model limits
# ---------------------------------------------------------------------------

# Claude Opus 4.8 context window (1M tokens, no beta header required).
MAX_CONTEXT_TOKENS = 1_000_000


# ---------------------------------------------------------------------------
# Per-spec review limits (used by GUI token gauge and per-spec pipeline)
# ---------------------------------------------------------------------------

# Practical per-call input limit for per-spec reviews.
# Individual specs are reviewed one at a time — this is the budget for a
# single (system prompt + project context + spec content) API call.
# Conservative relative to the 1M window and intended as a practical guardrail.
RECOMMENDED_MAX = 500_000

# Hard cap on the Project Context block. The context is sent on every per-spec
# review call, every cross-check call, and every verification call, so it
# multiplies cost quickly. 100K tokens leaves ~400K of the per-spec budget for
# the spec itself.
PROJECT_CONTEXT_MAX_TOKENS = 100_000


# ---------------------------------------------------------------------------
# Cross-check limits (v2.2.0)
# ---------------------------------------------------------------------------

# Cross-check uses Sonnet 4.6 with full spec content and adaptive thinking.
# With thinking enabled, thinking tokens + text output share the max_tokens budget.
# We keep a 128K output reserve (matches the api_config cross-check cap before
# the per-model clamp) so the input budget stays stable across model changes.
# Budget: 1M context - 128K output reserve - 50K overhead = 822K
CROSS_CHECK_OVERHEAD = 50_000
CROSS_CHECK_OUTPUT_BUDGET = 128_000
CROSS_CHECK_RECOMMENDED_MAX = (
    MAX_CONTEXT_TOKENS - CROSS_CHECK_OUTPUT_BUDGET - CROSS_CHECK_OVERHEAD
)


def exceeds_per_call_limit(spec_tokens: int, overhead_tokens: int) -> bool:
    """Check if a single spec would exceed the per-call token limit.

    Backward-compatible wrapper: no safety factor applied. New code that
    needs model-aware behavior should call
    :func:`exceeds_per_call_limit_for_model` instead.
    """
    return (overhead_tokens + spec_tokens) > RECOMMENDED_MAX


# ---------------------------------------------------------------------------
# Model-specific safety multipliers for the local cl100k_base estimate
# ---------------------------------------------------------------------------
#
# cl100k_base is OpenAI's tokenizer and does not exactly match Claude's
# tokenization. The undercount is usually modest for English prose
# (≤10%) but can be larger for structured spec text full of section
# numbers, table cells, and unicode punctuation. Without a safety factor
# the local estimate looks reassuring even when the real Claude count
# would breach the per-call budget — the goal is that local tokenizer
# estimates no longer create false confidence.
#
# The multipliers below are intentionally conservative. They are only
# consulted on the fallback path when the Anthropic ``count_tokens``
# endpoint is unavailable; once we have an exact count, that becomes
# the authoritative gate (directive 3).
_DEFAULT_LOCAL_SAFETY_FACTOR = 1.20  # unknown models — widest margin
_LOCAL_SAFETY_FACTORS: dict[str, float] = {
    # Opus / Sonnet share Claude's main tokenizer; the cl100k_base
    # undercount is small but non-zero.
    "claude-opus-4-8": 1.10,
    "claude-sonnet-4-6": 1.10,
    # Haiku 4.5 tokenization tends to undercount cl100k a bit more on
    # structured construction-spec text in practice. Pad more.
    "claude-haiku-4-5": 1.15,
}


def local_estimate_safety_factor(model: str | None) -> float:
    """Return the cl100k→Claude safety multiplier for ``model``.

    The factor is a conservative multiplier ≥ 1.0 applied to the local
    cl100k_base count whenever it is used as a budget gate. Unknown
    models fall back to ``_DEFAULT_LOCAL_SAFETY_FACTOR`` (the widest
    margin) so a future model never silently sails through a budget
    check that would have been blocked under a known model.

    The ``≥ 1.0`` floor is *enforced*, not merely assumed: a sub-1.0 entry
    slipping into ``_LOCAL_SAFETY_FACTORS`` (a typo, or a misguided attempt
    to trim the margin) would turn the safety pad into a *danger pad* — it
    would shrink the estimate below the raw local count, undercount the
    Claude token total, and let an over-budget spec sail through the
    fallback gate. Clamping here honors the contract for every caller, not
    just :func:`safe_local_estimate`.
    """
    factor = _LOCAL_SAFETY_FACTORS.get(model or "", _DEFAULT_LOCAL_SAFETY_FACTOR)
    return max(1.0, factor)


def safe_local_estimate(local_tokens: int, *, model: str | None) -> int:
    """Return ``local_tokens`` padded by the model-specific safety factor."""
    factor = local_estimate_safety_factor(model)
    # Round up — the factor is a safety margin, not a midpoint estimate.
    return math.ceil(local_tokens * factor)


def exceeds_per_call_limit_for_model(
    spec_tokens: int,
    overhead_tokens: int,
    *,
    model: str | None,
) -> bool:
    """Model-aware version of :func:`exceeds_per_call_limit`.

    Applies the model-specific safety factor to ``spec_tokens + overhead``
    before comparing against ``RECOMMENDED_MAX``. Use this when the local
    cl100k_base count is the only signal available (e.g. the API preflight
    failed or was disabled). When an exact Anthropic count is available,
    bypass this helper and compare the exact count directly to
    ``RECOMMENDED_MAX`` — the exact number is authoritative (directive 3).
    """
    padded = safe_local_estimate(overhead_tokens + spec_tokens, model=model)
    return padded > RECOMMENDED_MAX


def get_encoder():
    """Get the tokenizer used for approximate token estimates."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in a text string (local cl100k_base estimate)."""
    encoder = get_encoder()
    return len(encoder.encode(text))


# ---------------------------------------------------------------------------
# Image / vision token estimation
# ---------------------------------------------------------------------------
#
# Claude bills an image at approximately ``width * height / 750`` tokens (per
# the vision docs), after any resize down to the model's native resolution, and
# clamped to a per-model token cap. The published cost tables match
# ``ceil(w*h/750)`` with no extra padding, so we mirror that exactly:
#
#   * Opus 4.7 / 4.8 (high-resolution): up to 4784 tokens, long edge <= 2576 px.
#   * Other models (Sonnet 4.6 / Haiku 4.5 / unknown): up to 1568 tokens,
#     long edge <= 1568 px.
#
# These are local *estimates* for budgeting (mirroring the documented formula);
# the authoritative number is still Anthropic's ``count_tokens`` endpoint, which
# accepts image/document blocks like any other content.

_IMAGE_TOKEN_DIVISOR = 750

_IMAGE_TOKEN_CAP_HIRES = 4784      # Opus 4.7 / 4.8
_IMAGE_LONG_EDGE_HIRES = 2576
_IMAGE_TOKEN_CAP_DEFAULT = 1568    # Sonnet 4.6 / Haiku 4.5 / unknown
_IMAGE_LONG_EDGE_DEFAULT = 1568


def _image_caps_for_model(model: str | None) -> tuple[int, int]:
    """Return ``(token_cap, long_edge_cap_px)`` for ``model``.

    Reads the Opus high-resolution tier from the api_config whitelist so the
    capability source of truth stays single. Imported lazily to avoid any
    import-order coupling at module load.
    """
    try:
        from .api_config import OPUS_MODELS

        if model in OPUS_MODELS:
            return _IMAGE_TOKEN_CAP_HIRES, _IMAGE_LONG_EDGE_HIRES
    except Exception:  # pragma: no cover - defensive; fall back to safe default
        pass
    return _IMAGE_TOKEN_CAP_DEFAULT, _IMAGE_LONG_EDGE_DEFAULT


def estimate_image_tokens(width_px: int, height_px: int, *, model: str | None) -> int:
    """Estimate the billed token cost of one image of ``width_px x height_px``.

    Mirrors the documented vision pricing: resize down so the long edge fits the
    model's native resolution (preserving aspect ratio), then ``ceil(w*h/750)``,
    clamped to the per-model token cap. Returns an integer >= 0.
    """
    if width_px <= 0 or height_px <= 0:
        return 0
    token_cap, long_edge_cap = _image_caps_for_model(model)
    w = float(width_px)
    h = float(height_px)
    longest = max(w, h)
    if longest > long_edge_cap:
        scale = long_edge_cap / longest
        w *= scale
        h *= scale
    tokens = math.ceil((w * h) / _IMAGE_TOKEN_DIVISOR)
    return min(token_cap, tokens)


def estimate_image_tokens_total(
    sizes: list[tuple[int, int]], *, model: str | None
) -> int:
    """Sum :func:`estimate_image_tokens` over a list of ``(width, height)`` sizes."""
    return sum(estimate_image_tokens(w, h, model=model) for w, h in sizes)


def count_tokens_via_api(
    *,
    model: str,
    system: Any,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    client: Any = None,
) -> Optional[int]:
    """Exact token count via Anthropic's count_tokens endpoint.

    Returns the input-token total for the given request shape, or ``None`` on
    failure (network error, missing API key, SDK version mismatch). Callers
    should treat ``None`` as "preflight unavailable" and fall back to the
    local estimate rather than blocking submission.

    Plan section 6.3: keep the local estimate for UI responsiveness, use this
    helper before batch submission when exact routing/guardrail decisions
    matter.
    """
    if client is None:
        try:
            from ..client import get_client as _get_client
            client = _get_client()
        except Exception as exc:  # pragma: no cover - exercised via tests
            _log.warning("count_tokens_via_api: no client available (%s)", exc)
            return None
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        result = client.messages.count_tokens(**kwargs)
        # MessageTokensCount has an input_tokens attribute.
        return int(getattr(result, "input_tokens", 0) or 0)
    except Exception as exc:
        _log.warning("count_tokens_via_api failed: %s", exc)
        return None
