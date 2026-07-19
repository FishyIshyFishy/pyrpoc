from pyrpoc_next.structs.status import CompatibilityReport, IssueSeverity


def test_report_blocked_only_on_halt():
    report = CompatibilityReport()
    assert not report.blocked
    report.add(IssueSeverity.warn, "a display is streaming-only")
    assert not report.blocked
    report.add(IssueSeverity.halt, "no display can show this modality's data")
    assert report.blocked
    assert len(report.issues) == 2
