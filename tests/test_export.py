"""Hermetic tests for the takeoff export documents (pure builders + writer)."""
from __future__ import annotations

from drawing_takeoff import export
from drawing_takeoff.models import StyleKey, TakeoffItem, TakeoffResult

_PIPE = StyleKey((0.0, 0.0, 0.0), 1.3, "[] 0")
_GRAY = StyleKey((0.667, 0.667, 0.667), 0.58, "[] 0")


def _result() -> TakeoffResult:
    items = [
        TakeoffItem("Sprinkler pipe", 1404.2, "LF", "FP2.20#p0", _PIPE, 9.0, confidence="high", run_count=282),
        TakeoffItem("Sprinkler pipe", 891.2, "LF", "FP2.21#p0", _PIPE, 9.0, confidence="medium", run_count=144),
        TakeoffItem("Maybe branch", 146.0, "LF", "FP2.21#p0", _GRAY, 9.0, confidence="low",
                    ambiguous=True, reasoning="gray, ambiguous vs background"),
    ]
    r = TakeoffResult(items=items, sheet_count=2, diagnostics=["FP2.20#p0: scale=1/8\" ppf=9"])
    r.per_system_totals = {"Sprinkler pipe": 2295.4}
    return r


def test_build_documents_shape_and_aggregation():
    docs = dict(export.build_takeoff_documents(_result()))
    assert set(docs) == {"takeoff_by_system.csv", "takeoff_detail.csv", "diagnostics.txt"}

    by_system = docs["takeoff_by_system.csv"]
    assert "Sprinkler pipe" in by_system
    assert "2295.4" in by_system           # the two trusted pipe rows summed
    assert "Maybe branch" not in by_system  # the flagged style is NOT counted

    detail = docs["takeoff_detail.csv"]
    assert "FP2.20#p0" in detail and "FP2.21#p0" in detail
    assert "YES" in detail                  # the ambiguous row is flagged

    diag = docs["diagnostics.txt"]
    assert "FLAGGED" in diag and "Maybe branch" in diag


def test_write_export_creates_timestamped_folder(tmp_path):
    folder = export.write_takeoff_export(_result(), tmp_path, project_name="My Job 1")
    assert folder.exists() and folder.name.startswith("My_Job_1_")
    assert {p.name for p in folder.iterdir()} == {
        "takeoff_by_system.csv", "takeoff_detail.csv", "diagnostics.txt"
    }
    assert "Sprinkler pipe" in (folder / "takeoff_by_system.csv").read_text()


def test_build_takeoff_workbook_has_summary_detail_review():
    tables = {
        "by_system_size": {
            ("Fire-protection sprinkler", '1"'): 100.0,
            ("Fire-protection sprinkler", '1-1/2"'): 250.0,
            ("Fire-protection sprinkler", "unsized"): 30.0,
        },
        "detail": [
            {"network": "N0", "system": "Fire-protection sprinkler", "is_pipe": True, "counted": True,
             "confidence": "high", "ambiguous": False, "linear_feet": 380.0, "pct_page": 40,
             "sizes": '1"=100, 1-1/2"=250', "reasoning": "main with branches"},
        ],
        "review": ["N1: excluded — non-pipe (Matchline) — 200 LF"],
    }
    wb = export.build_takeoff_workbook(tables)
    assert wb.sheetnames == ["Summary", "Detail", "Review"]

    summary = list(wb["Summary"].iter_rows(values_only=True))
    assert summary[0] == ("System", "Size", "Linear Feet")
    assert ("Fire-protection sprinkler", '1"', 100.0) in summary
    # sizes sort ascending; 'unsized' last; a bold total row sums the system
    assert summary.index(("Fire-protection sprinkler", '1"', 100.0)) < \
        summary.index(("Fire-protection sprinkler", "unsized", 30.0))
    assert any(row[0] == "Fire-protection sprinkler — total" and row[2] == 380.0 for row in summary)

    detail = list(wb["Detail"].iter_rows(values_only=True))
    assert detail[0][0] == "Network" and detail[1][0] == "N0"
    assert any("N1" in str(row[0]) for row in wb["Review"].iter_rows(values_only=True) if row[0])
