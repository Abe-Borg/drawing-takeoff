"""Model pricing + request cost estimation (USD).

A small, dependency-free pricing table so the app can show a spend estimate
before launching an expensive run (e.g. the drawing-analysis cost-confirm
dialog). Rates are USD per million tokens, current as of 2026-06. Image/vision
input is billed as ordinary input tokens, so no separate image rate is needed;
the Batch API bills at 50% of standard, exposed via the ``batch=`` flag.

**Rates drift — verify against the published pricing before relying on a figure.**
Unknown model ids return ``None`` from :func:`price_for` /
:func:`estimate_request_cost` (the caller shows scale without a dollar figure
rather than guessing a wrong number).
"""
from __future__ import annotations

from dataclasses import dataclass

# Batch API bills at half of standard, per Anthropic's published pricing.
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class ModelPrice:
    """USD per **million** tokens."""

    input_per_mtok: float
    output_per_mtok: float
    label: str  # human-friendly name for dialogs


# Keyed by the bare model id. A dated/fast/-suffixed variant resolves via the
# startswith fallback in ``price_for`` (e.g. "claude-haiku-4-5-20251001").
MODEL_PRICING: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(5.00, 25.00, "Opus 4.8"),
    "claude-opus-4-7": ModelPrice(5.00, 25.00, "Opus 4.7"),
    "claude-opus-4-6": ModelPrice(5.00, 25.00, "Opus 4.6"),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00, "Sonnet 4.6"),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00, "Haiku 4.5"),
}


def price_for(model: str) -> ModelPrice | None:
    """Resolve a model id to its price, tolerating dated/suffixed variants.

    Exact match first, then the longest known-prefix match so a variant like
    ``claude-haiku-4-5-20251001`` or ``claude-opus-4-8-fast`` still resolves.
    Returns ``None`` for an unrecognized id.
    """
    if not model:
        return None
    exact = MODEL_PRICING.get(model)
    if exact is not None:
        return exact
    # Only a *delimited* variant resolves to a base price — "claude-opus-4-8-fast"
    # or a dated "...-4-5-20251001", but NOT a different model whose id merely
    # starts with a known one (e.g. a future "claude-opus-4-80" must stay
    # unknown → None, not silently priced as 4.8). Longest match wins.
    best_key = ""
    for key in MODEL_PRICING:
        if model.startswith(key + "-") and len(key) > len(best_key):
            best_key = key
    return MODEL_PRICING[best_key] if best_key else None


def friendly_model_name(model: str) -> str:
    """Human-friendly label (``"Opus 4.8"``) for dialogs; falls back to the id."""
    price = price_for(model)
    return price.label if price is not None else model


def estimate_request_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    model: str,
    batch: bool = False,
) -> float | None:
    """Estimated USD cost of a request, or ``None`` if the model is unknown.

    Image/vision input counts as ordinary input tokens, so callers fold image
    tokens into ``input_tokens``. ``batch=True`` applies the 50% Batch discount.
    """
    price = price_for(model)
    if price is None:
        return None
    factor = BATCH_DISCOUNT if batch else 1.0
    cost = (
        (input_tokens / 1_000_000) * price.input_per_mtok
        + (output_tokens / 1_000_000) * price.output_per_mtok
    )
    return cost * factor
