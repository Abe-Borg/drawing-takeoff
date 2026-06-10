"""Hermetic tests for the M3 legend step — the project's first tool-use call.

No network, no PyMuPDF: a ``FakeClient`` returns a scripted ``tool_use`` block
(the structured style->system mapping) and ``legend.label_styles`` is asserted to
build the forced-tool request and parse the response, flagging anything the model
omits rather than trusting it.
"""
from __future__ import annotations

import pytest

from drawing_takeoff import legend, measure
from drawing_takeoff.models import GeometryPath, Network, SheetGeometry, SheetRef, SystemLabel, TextWord
from tests.fixtures.fake_anthropic import FakeClient, FakeMessage, FakeToolUseBlock, FakeUsage


def _line(p0, p1, *, color, width):
    return GeometryPath(
        items=(("l", p0, p1),),
        stroke_color=color,
        fill_color=None,
        width=width,
        dashes="[] 0",
        closed=False,
        bbox=(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])),
        kind="stroke",
    )


@pytest.fixture
def geom():
    # Heavy black: 3 disconnected runs (300pt total) -> ranks first (s0).
    # Gray: 1 run (100pt) -> s1.
    paths = [
        _line((0, 0), (100, 0), color=(0.0, 0.0, 0.0), width=1.3),
        _line((0, 20), (100, 20), color=(0.0, 0.0, 0.0), width=1.3),
        _line((0, 40), (100, 40), color=(0.0, 0.0, 0.0), width=1.3),
        _line((0, 60), (100, 60), color=(0.667, 0.667, 0.667), width=0.5),
    ]
    return SheetGeometry(
        ref=SheetRef("synthetic", 0),
        page_width_pt=216,
        page_height_pt=144,
        paths=paths,
        words=[TextWord("FP", (5, 5, 15, 15))],
        scale_label='1/8" = 1\'-0"',
        points_per_foot=9.0,
    )


def _responder_only_s0(kwargs):
    # The request must force the single structured tool.
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_system_labels"}
    assert any(t["name"] == "record_system_labels" for t in kwargs["tools"])
    return FakeMessage(
        content=[
            FakeToolUseBlock(
                name="record_system_labels",
                input={
                    "labels": [
                        {
                            "style_id": "s0",
                            "system": "Fire-protection sprinkler pipe",
                            "measurable": True,
                            "confidence": "high",
                            "ambiguous": False,
                            "reasoning": "Heavy black pen, many long runs — the primary FP system.",
                        }
                        # s1 deliberately omitted to exercise the flag-don't-guess default.
                    ]
                },
            )
        ],
        usage=FakeUsage(input_tokens=120, output_tokens=40),
    )


def test_label_styles_parses_tool_use_and_injects_client(geom):
    client = FakeClient(_responder_only_s0)
    labels = legend.label_styles(geom, client=client, discipline="fire protection")

    black = next(k for k in labels if k.stroke_color == (0.0, 0.0, 0.0))
    gray = next(k for k in labels if k.stroke_color == (0.667, 0.667, 0.667))

    assert labels[black].system == "Fire-protection sprinkler pipe"
    assert labels[black].measurable is True
    assert labels[black].confidence == "high"
    assert labels[black].trusted is True

    # The omitted style is flagged ambiguous, never silently trusted.
    assert labels[gray].ambiguous is True
    assert labels[gray].measurable is False
    assert labels[gray].trusted is False

    # The request was recorded (one structured call, no network).
    assert len(client.messages.calls) == 1
    sent = client.messages.calls[0]
    assert sent["model"]
    assert "fire protection" in sent["system"].lower() or "construction" in sent["system"].lower()


def test_label_styles_passes_images_when_provided(geom):
    captured = {}

    def responder(kwargs):
        captured["content"] = kwargs["messages"][0]["content"]
        return _responder_only_s0(kwargs)

    # one fake swatch + a fake legend image (bytes need not be real PNGs here)
    black_key = _line((0, 0), (1, 0), color=(0.0, 0.0, 0.0), width=1.3).style_key
    legend.label_styles(
        geom,
        client=FakeClient(responder),
        style_images={black_key: b"\x89PNG-fake"},
        legend_image=b"\x89PNG-legend",
    )
    kinds = [b.get("type") for b in captured["content"]]
    assert kinds.count("image") == 2  # swatch + legend image both attached


def test_cli_requires_api_key(monkeypatch, capsys):
    # The legend CLI is the LLM step — with no key it must bail before any work
    # (no PDF read, no network), not crash.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    rc = legend.main(["does-not-matter.pdf"])
    assert rc == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_label_styles_empty_geometry_returns_empty():
    empty = SheetGeometry(ref=SheetRef("x", 0), page_width_pt=10, page_height_pt=10, points_per_foot=9.0)
    # No client call should be needed when there are no candidate styles.
    sentinel = FakeClient(lambda kw: (_ for _ in ()).throw(AssertionError("should not call the API")))
    assert legend.label_styles(empty, client=sentinel) == {}


# ---- M7: network labeling + System x Size assembly ------------------------
def _black(p0, p1):
    return _line(p0, p1, color=(0.0, 0.0, 0.0), width=1.3)


def _network(nid, *segments, ppf=9.0):
    runs = []
    for p0, p1 in segments:
        runs += measure.stitch_runs([_black(p0, p1)], ppf=ppf)
    return Network(id=nid, runs=tuple(runs), ppf=ppf)


def _net_geom(words=None):
    return SheetGeometry(
        ref=SheetRef("synthetic", 0), page_width_pt=216, page_height_pt=144,
        words=words or [], scale_label='1/8" = 1\'-0"', points_per_foot=9.0,
    )


def _net_responder(kwargs):
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_network_labels"}
    assert any(t["name"] == "record_network_labels" for t in kwargs["tools"])
    return FakeMessage(content=[FakeToolUseBlock(
        name="record_network_labels",
        input={"labels": [
            {"network_id": "N0", "system": "Fire-protection sprinkler", "is_pipe": True,
             "confidence": "high", "ambiguous": False, "reasoning": "main with branches"},
            {"network_id": "N1", "system": "Matchline / non-pipe", "is_pipe": False,
             "confidence": "high", "ambiguous": False, "reasoning": "spans the sheet, no branches"},
        ]},
    )], usage=FakeUsage(input_tokens=100, output_tokens=50))


def test_label_networks_keys_on_ids_and_maps_is_pipe():
    nets = [_network("N0", ((0, 0), (100, 0))), _network("N1", ((0, 50), (100, 50)))]
    labels = legend.label_networks(_net_geom(), nets, client=FakeClient(_net_responder), discipline="fire protection")
    assert labels["N0"].measurable is True and labels["N0"].system.startswith("Fire-protection")
    assert labels["N0"].trusted is True
    assert labels["N1"].measurable is False and labels["N1"].trusted is False


def test_label_networks_flags_unreturned_network():
    nets = [_network("N0", ((0, 0), (100, 0))), _network("N1", ((0, 50), (100, 50)))]

    def resp(kw):
        return FakeMessage(content=[FakeToolUseBlock(name="record_network_labels", input={"labels": [
            {"network_id": "N0", "system": "FP sprinkler", "is_pipe": True,
             "confidence": "high", "ambiguous": False, "reasoning": "x"}]})])

    labels = legend.label_networks(_net_geom(), nets, client=FakeClient(resp))
    assert labels["N1"].ambiguous is True and labels["N1"].measurable is False


def test_system_size_takeoff_buckets_trusted_and_reviews_rest():
    geom = _net_geom(words=[TextWord("2", (50, 2, 56, 8))])     # 2" tag on N0's run
    n0 = _network("N0", ((0, 0), (100, 0)))
    n1 = _network("N1", ((0, 50), (100, 50)))
    n2 = _network("N2", ((0, 90), (60, 90)))
    labels = {
        "N0": SystemLabel(system="FP sprinkler", measurable=True, confidence="high", ambiguous=False),
        "N1": SystemLabel(system="Matchline", measurable=False, confidence="high", ambiguous=False),
        "N2": SystemLabel(system="FP sprinkler", measurable=True, confidence="low", ambiguous=True),
    }
    by, review = legend.system_size_takeoff([n0, n1, n2], labels, geom, ppf=9.0)
    assert by[("FP sprinkler", '2"')] == pytest.approx(100 / 9)   # N0 -> 2"
    assert any("N1" in r for r in review)    # non-pipe excluded
    assert any("N2" in r for r in review)    # ambiguous flagged, not counted


def test_build_system_size_report_smoke():
    geom = _net_geom(words=[TextWord("2", (50, 2, 56, 8))])
    nets = [_network("N0", ((0, 0), (100, 0)))]
    labels = {"N0": SystemLabel(system="FP sprinkler", measurable=True, confidence="high", ambiguous=False)}
    report = legend.build_system_size_report(nets, labels, geom, ppf=9.0)
    assert "M7 takeoff" in report and "FP sprinkler" in report and '2"' in report


def test_takeoff_tables_summary_detail_review():
    geom = _net_geom(words=[TextWord("2", (50, 2, 56, 8))])
    n0 = _network("N0", ((0, 0), (100, 0)))
    n1 = _network("N1", ((0, 50), (100, 50)))
    labels = {
        "N0": SystemLabel(system="FP sprinkler", measurable=True, confidence="high", ambiguous=False),
        "N1": SystemLabel(system="Matchline", measurable=False, confidence="high", ambiguous=False),
    }
    tables = legend.takeoff_tables([n0, n1], labels, geom, ppf=9.0)
    assert tables["by_system_size"][("FP sprinkler", '2"')] == pytest.approx(100 / 9)
    assert [r["network"] for r in tables["detail"]] == ["N0", "N1"]
    assert tables["detail"][0]["counted"] is True and tables["detail"][0]["is_pipe"] is True
    assert tables["detail"][1]["counted"] is False     # N1 non-pipe excluded
    assert any("N1" in r for r in tables["review"])


def test_takeoff_tables_detail_keeps_unsized_and_review_has_confirm():
    geom = _net_geom(words=[TextWord("2", (10, 2, 16, 8))])   # 2" tag near the start only
    # one network: a run near the tag (sized 2") + a run far from any tag (unsized);
    # its bbox spans the page width, so it also triggers the CONFIRM advisory.
    n0 = _network("N0", ((0, 0), (50, 0)), ((200, 0), (250, 0)))
    labels = {"N0": SystemLabel(system="FP sprinkler", measurable=True, confidence="high", ambiguous=False)}
    tables = legend.takeoff_tables([n0], labels, geom, ppf=9.0)
    assert '2"' in tables["detail"][0]["sizes"] and "unsized" in tables["detail"][0]["sizes"]
    assert any("CONFIRM" in r for r in tables["review"])
