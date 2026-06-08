"""Hermetic tests for scale-label arithmetic and the dimension verifier.

No PDF, no network: ``scale`` is pure string/number work.
"""
from __future__ import annotations

import math

import pytest

from drawing_takeoff import scale


@pytest.mark.parametrize(
    "label, ppf",
    [
        ('1/8" = 1\'-0"', 9.0),      # the sample sheet
        ('1/4"=1\'-0"', 18.0),       # plan's worked example
        ('1/2" = 1\'-0"', 36.0),
        ('3/32" = 1\'-0"', 6.75),    # 72 * 3/32
        ('1" = 1\'-0"', 72.0),
        ('3" = 1\'-0"', 216.0),
        ('1-1/2" = 1\'-0"', 108.0),  # mixed number on the paper side
        ('1" = 20\'-0"', 3.6),       # engineering scale
        ('1" = 20\'', 3.6),          # engineering, no inches
        ('1"=30\'', 2.4),
        ('⅛" = 1′-0″', 9.0),  # ⅛ + unicode prime/double-prime
    ],
)
def test_points_per_foot_from_label(label, ppf):
    assert scale.points_per_foot_from_label(label) == pytest.approx(ppf)


@pytest.mark.parametrize("bad", ["NTS", "AS NOTED", "scale varies", ""])
def test_points_per_foot_rejects_unparseable(bad):
    with pytest.raises(ValueError):
        scale.points_per_foot_from_label(bad)


def test_verify_against_dimension_exact_and_off():
    # 108 pt at 9 pt/ft is exactly 12 ft.
    assert scale.verify_against_dimension(108.0, 12.0, 9.0) == pytest.approx(0.0)
    assert scale.verify_against_dimension(90.0, 10.0, 9.0) == pytest.approx(0.0)
    # 110 pt reads 12.22 ft -> +1.85%.
    assert scale.verify_against_dimension(110.0, 12.0, 9.0) == pytest.approx(1.852, abs=0.01)
    # short read is negative.
    assert scale.verify_against_dimension(106.0, 12.0, 9.0) < 0


def test_verify_against_dimension_guards_zero():
    with pytest.raises(ValueError):
        scale.verify_against_dimension(100.0, 0.0, 9.0)
    with pytest.raises(ValueError):
        scale.verify_against_dimension(100.0, 12.0, 0.0)


@pytest.mark.parametrize(
    "token, kwargs, feet",
    [
        ('12\'-0"', {}, 12.0),
        ('10\'-10"', {}, 10.0 + 10 / 12),
        ("12'", {}, 12.0),
        ("1′-6″", {}, 1.5),                       # unicode prime/double-prime
        ("14-10", {"allow_bare_hyphen": True}, 14.0 + 10 / 12),
        ("4-6", {"allow_bare_hyphen": True}, 4.5),
    ],
)
def test_parse_feet_inches(token, kwargs, feet):
    assert scale.parse_feet_inches(token, **kwargs) == pytest.approx(feet)


@pytest.mark.parametrize("token", ["N-145", "FP4.01", "14-10", "hello"])
def test_parse_feet_inches_rejects_bare_hyphen_by_default(token):
    # Bare N-N is ambiguous with node ids, so it's opt-in only.
    assert scale.parse_feet_inches(token) is None


def test_detect_scale_label_in_titleblock_text():
    text = "GENERAL NOTES\nFIRE PROTECTION\nSCALE\n1/8\" = 1'-0\"\nFP2.20"
    found = scale.detect_scale_label(text)
    assert found is not None
    assert scale.points_per_foot_from_label(found) == pytest.approx(9.0)


def test_detect_scale_label_ignores_stray_dimension():
    # A lone "= 1'" elsewhere must not be mistaken for a scale label.
    assert scale.detect_scale_label("provide a 1' clearance and = 1' typ") is None
    assert scale.detect_scale_label("no scale here") is None
