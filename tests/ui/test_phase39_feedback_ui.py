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
    assert "aHealthScore" in html
    assert "analyticsInsights" in html
    assert "function analyticsHealth" in html
    assert "function renderAnalyticsInsights" in html
    assert "Agent Evaluation Center" in html
    assert "/api/admin/evaluations/runs" in html
    assert "agentEvalRunsTable" in html
    assert "agentEvalResultsTable" in html
    assert "function runAgentEvaluation" in html


def test_admin_dev_identity_is_collapsible_and_debug_gated():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "Admin Request Identity" not in html
    assert 'id="adminDevIdentity" hidden' in html
    assert 'id="authUserId"' in html
    assert 'id="authRoles"' in html
    assert 'id="authChannel"' in html
    assert 'id="btnSaveAuth"' in html
    assert "debug_auth_inputs_enabled" in html


def test_admin_role_based_shell_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "roleShellTitle" in html
    assert "TAB_ACCESS" in html
    assert "function loadViewerProfile" in html
    assert "function canAccessTarget" in html
    assert "/api/me" in html
    assert "view-access-denied" in html


def test_support_workspace_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "view-support-workspace" in html
    assert "Support Workspace" in html
    assert "support-email-subview" in html
    assert "support-cases-subview" in html
    assert "function activateSupportSubtab" in html
    assert "supportWorkspaceCasesTable" in html
    assert "supportWorkspaceEmailsTable" in html
    assert "supportWorkspaceActionsTable" in html
    assert "workspaceCaseContextBody" in html
    assert "workspaceCaseContextView" in html
    assert "function renderReadableCaseContext" in html
    assert "Raw JSON Debug" in html
    assert "workspaceEmailThreadBody" in html
    assert "workspaceEmailReplyBody" in html
    assert "btnWorkspaceSendSupportEmailReply" in html
    assert "workspacePublicReplyBody" in html
    assert "btnWorkspacePublicReply" in html
    assert "btnWorkspacePublicReplyResolve" in html
    assert "function sendPublicSupportReply" in html
    assert "btnWorkspaceCloseCase" in html
    assert "function closeSupportCase" in html
    assert "renderSupportWorkspaceCases" in html
    assert "renderSupportWorkspaceActions" in html


def test_knowledge_workspace_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "view-knowledge-workspace" in html
    assert "Knowledge Workspace" in html
    assert "kwNewKbName" in html
    assert "kwSelectedKbName" in html
    assert "btnKwRenameKb" in html
    assert "kwDriveSourceName" in html
    assert "btnKwCreateKb" in html
    assert "btnKwCreateDriveSource" in html
    assert "function renameSelectedKb" in html
    assert "body:JSON.stringify({ name })" in html
    assert "knowledgeJobSummary" in html
    assert "knowledgeWorkspaceKbFilesTable" in html
    assert "knowledgeWorkspaceLibraryTable" in html
    assert "knowledgeWorkspaceDriveSourcesTable" in html
    assert "knowledgeWorkspaceSourcesTable" in html
    assert "Knowledge Quality Workflow" in html
    assert "Knowledge Review Queue" in html
    assert "knowledgeReviewQueueTable" in html
    assert "knowledgeReviewQueueFilter" in html
    assert "function refreshKnowledgeReviewQueue" in html
    assert "/api/kbs/${selectedKbId()}/review-queue" in html
    assert "knowledgeQualityTable" in html
    assert "kwQualityScore" in html
    assert "kwStaleDocs" in html
    assert "function refreshKnowledgeQuality" in html
    assert "/api/kbs/${selectedKbId()}/quality" in html
    assert "function updateKbFileLifecycle" in html
    assert "function showKbFileDiff" in html
    assert "data-kb-lifecycle" in html
    assert "data-kb-diff" in html
    assert "renderKnowledgeWorkspace" in html
    assert "'view-knowledge-workspace'" in html


def test_admin_legacy_views_are_hidden_backing_views():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert 'class="view legacy-backfill" hidden aria-hidden="true"' in html
    for view_id in [
        "view-kb",
        "view-drive",
        "view-email",
        "view-support-cases",
        "view-actions",
        "view-jobs",
        "view-audit",
    ]:
        assert f'id="{view_id}" class="view legacy-backfill" hidden aria-hidden="true"' in html


def test_operations_workspace_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "view-operations-workspace" in html
    assert "Operations Workspace" in html
    assert "operationsPendingActionsTable" in html
    assert "operationsBackgroundJobsTable" in html
    assert "Durable Workflows" in html
    assert "operationsWorkflowRunsTable" in html
    assert "operationsWorkflowStatus" in html
    assert "function refreshWorkflowRuns" in html
    assert "function viewWorkflowRun" in html
    assert "data-workflow-retry" in html
    assert "data-workflow-resume" in html
    assert "operationsSyncSchedulesTable" in html
    assert "Notification Center" in html
    assert "operationsNotificationsTable" in html
    assert "operationsWebhookDeliveriesTable" in html
    assert "Webhook Subscriptions" in html
    assert "operationsWebhookSubscriptionsTable" in html
    assert "btnSaveWebhookSubscription" in html
    assert "function refreshWebhookSubscriptions" in html
    assert "function saveWebhookSubscription" in html
    assert "function testWebhookSubscription" in html
    assert "function refreshNotifications" in html
    assert "function refreshWebhookDeliveries" in html
    assert "function markNotificationRead" in html
    assert "function retryWebhookDelivery" in html
    assert "Agent Trace Timeline" in html
    assert "/api/admin/support-tickets/${caseId}/timeline" in html
    assert "function renderCaseTimeline" in html
    assert "Generate Draft" in html
    assert "/api/admin/support-tickets/${selectedSupportCaseId}/draft-reply" in html
    assert "function generateSupportDraftReply" in html
    assert "operationsChatLogsTable" in html
    assert "operationsFeedbackTable" in html
    assert "operationsAuthAuditTable" in html
    assert "renderOperationsWorkspace" in html
    assert "'view-operations-workspace'" in html


def test_mcp_security_hardening_ui_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "mcpScopes" in html
    assert "Tool Scopes" in html
    assert "blocked_by_policy" in html
    assert "required_scopes" in html
    assert "X-MCP-Scopes" in html
    assert "MCP Sessions" in html
    assert "mcpSessionsTable" in html
    assert "require_client_token" in html
    assert "tool_quota_rules" in html


def test_internal_user_portal_is_wired():
    html = (ROOT / "static" / "internal.html").read_text(encoding="utf-8")

    assert "Internal User Portal" in html
    assert "/api/chat" in html
    assert "/api/chat/kbs" in html
    assert "/api/feedback/chat" in html
    assert "/api/support-tickets" in html
    assert "function createTicket" in html
    assert "function refreshTickets" in html
    assert "function viewTicket" in html
    assert "ticketDetail" in html
    assert "data-ticket-view" in html
    assert "Support Replies" in html
    assert "/api/support-tickets/${ticketId}/notes" in html
    assert "Reply to Support" in html
    assert "This ticket is closed." in html
    assert "function sendTicketReply" in html
    assert "data-ticket-reply" in html
    assert "function sendChat" in html
    assert "currentEvent === 'token'" in html
    assert "data.text || data.token" in html
    assert "currentEvent === 'start'" in html


def test_modal_loading_empty_state_cleanup_is_wired():
    admin_html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")
    portal_html = (ROOT / "static" / "internal.html").read_text(encoding="utf-8")

    assert "adminModal" in admin_html
    assert "function modalConfirm" in admin_html
    assert "function modalPrompt" in admin_html
    assert "function loadingRow" in admin_html
    assert "function emptyRow" in admin_html
    assert "button.loading" in admin_html
    assert "loadingRow" in portal_html
    assert "empty-state" in portal_html
    assert "setButtonLoading" in portal_html


def test_table_headers_are_sticky_in_admin_and_portal():
    admin_html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")
    portal_html = (ROOT / "static" / "internal.html").read_text(encoding="utf-8")

    for html in [admin_html, portal_html]:
        assert ".table th { position:sticky;" in html
        assert "top:0;" in html
        assert ".scroll { max-height:360px; overflow:auto; position:relative; }" in html


def test_admin_refresh_is_role_aware():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "TASK_ACCESS" in html
    assert "function canAccessTask" in html
    assert "async function refreshIfAllowed" in html
    assert "async function allAllowed" in html
    assert "refreshIfAllowed('support'" in html
    assert "refreshIfAllowed('operations'" in html
    assert "refreshIfAllowed('analytics'" in html
    assert "refreshIfAllowed('audit'" in html
    assert "refreshIfAllowed('mcp'" in html
    assert "refreshIfAllowed('system'" in html
