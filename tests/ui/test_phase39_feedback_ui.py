from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_chat_feedback_buttons_call_backend():
    html = (ROOT / "static" / "chat.html").read_text(encoding="utf-8")

    assert "/api/feedback/chat" in html
    assert "function submitFeedback" in html
    assert "responseRequestId" in html
    assert "Feedback saved" in html


def test_admin_feedback_panels_use_admin_endpoints():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "feedbackTable" in html
    assert "feedbackSummaryTable" in html
    assert "/api/admin/feedback?limit=30" in html
    assert "/api/admin/feedback/summary" in html
    assert "feedback_up" in html
    assert "feedback_down" in html


def test_admin_analytics_dashboard_wires_backend_endpoint():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "view-analytics" in html
    assert "Analytics Dashboard" in html
    assert "function refreshAnalytics" in html
    assert "/api/admin/analytics" in html
    assert "analyticsTimelineTable" in html
