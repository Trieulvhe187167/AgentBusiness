from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_chat_feedback_buttons_call_backend():
    html = (ROOT / "static" / "chat.html").read_text(encoding="utf-8")

    assert "/api/feedback/chat" in html
    assert "function submitFeedback" in html
    assert "responseRequestId" in html
    assert "Feedback saved" in html
    assert "feedbackModal" in html
    assert "feedbackReason" in html
    assert "reason_code" in html


def test_chat_sprint1_answer_to_ticket_and_collapsed_debug_ui_are_wired():
    html = (ROOT / "static" / "chat.html").read_text(encoding="utf-8")

    assert "evidence-collapsed" in html
    assert "devSettingsToggle" in html
    assert "Auth settings" in html
    assert "function createTicketFromAnswer" in html
    assert "/api/support-tickets" in html
    assert "source_chat_request_id" in html
    assert "source_citations" in html


def test_portal_ticket_timeline_and_plain_sla_language_are_wired():
    html = (ROOT / "static" / "internal.html").read_text(encoding="utf-8")

    assert "function plainTicketStatus" in html
    assert "function slaText" in html
    assert "function renderTimeline" in html
    assert "Dang cho phe duyet" in html
    assert "ticket-timeline" in html
    assert "<h3>Timeline</h3>" in html


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
    assert "capProvider" in html
    assert "LLM Provider Capabilities" in html
    assert "analyticsInsights" in html
    assert "function analyticsHealth" in html
    assert "function renderAnalyticsInsights" in html
    assert "AI Ops Snapshot" in html
    assert "aiOpsSummary" in html
    assert "aiOpsEvalTrendTable" in html
    assert "aiOpsReplayChatId" in html
    assert "btnRunAiOpsReplay" in html
    assert "function refreshAiOps" in html
    assert "function runAiOpsReplay" in html
    assert "/api/admin/ai-ops/summary" in html
    assert "/api/admin/ai-ops/replay/chat-logs" in html
    assert "Agent Evaluation Center" in html
    assert "/api/admin/evaluations/runs" in html
    assert "Golden Dataset" in html
    assert "goldenDatasetTable" in html
    assert "function refreshGoldenDataset" in html
    assert "function createGoldenDatasetItem" in html
    assert "function uploadGoldenDatasetCsv" in html
    assert "function createGoldenEvalSchedule" in html
    assert "/api/admin/evaluations/golden-dataset" in html
    assert "/api/admin/evaluations/golden-dataset/upload" in html
    assert 'value="golden_dataset"' in html
    assert "payload.alert_drop_threshold" in html
    assert "evalLlmJudge" in html
    assert "payload.llm_judge" in html
    assert "agent_eval_run" in html
    assert "answer_similarity" in html
    assert "recall_at_k" in html
    assert "citation_accuracy" in html
    assert "judge_score" in html
    assert "agentEvalRunsTable" in html
    assert "agentEvalResultsTable" in html
    assert "function runAgentEvaluation" in html
    assert "Knowledge Gaps" in html
    assert "knowledgeGapsTable" in html
    assert "knowledgeQualityDebt" in html
    assert "Source Needed" in html
    assert "Patch Pending" in html
    assert "Owner" in html
    assert "Priority" in html
    assert "function refreshKnowledgeGaps" in html
    assert "function triageKnowledgeGap" in html
    assert "renderKnowledgeQualityDebt" in html
    assert "function suggestKnowledgeGapFaq" in html
    assert "function createKnowledgeGapReportSchedule" in html
    assert "/api/admin/knowledge-gaps" in html
    assert "/api/admin/knowledge-gaps/quality-debt" in html
    assert "/suggest-faq" in html
    assert "knowledge_gap_report" in html
    assert "btnScheduleKnowledgeGapReport" in html
    assert "data-gap-triage" in html
    assert "data-gap-suggest" in html
    assert "data-gap-status" in html
    assert "/api/admin/pending-actions/${actionId}/events" in html
    assert "data-action-events" in html
    assert "function viewApprovalEvents" in html
    assert "idempotency_key" in html


def test_admin_rag_quality_cockpit_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "view-rag-quality" in html
    assert "RAG Quality & Evaluation Cockpit" in html
    assert "ragQualityDays" in html
    assert "ragQualityScope" in html
    assert "btnRefreshRagQuality" in html
    assert "btnRunCockpitGoldenEval" in html
    assert "rqEmbeddingStack" in html
    assert "rqRerankerStack" in html
    assert "rqVectorStack" in html
    assert "rqCorrectiveStack" in html
    assert "rqRetrievalBudget" in html
    assert "Latest Eval Metric Breakdown" in html
    assert "ragQualityMetricGrid" in html
    assert "Baseline Regression Gate" in html
    assert "ragQualityRegressionTable" in html
    assert "ragQualityRunsTable" in html
    assert "ragQualityFailuresTable" in html
    assert "ragQualityGoldenTable" in html
    assert "function renderRagQualityCockpit" in html
    assert "function renderRagQualityMetrics" in html
    assert "function ragQualityRegressionRowsHtml" in html
    assert "function refreshRagQualityCockpitData" in html
    assert "function runCockpitGoldenEvaluation" in html
    assert "/api/system?kb_id=${selectedKbId()}" in html
    assert "/api/admin/analytics?${params.toString()}" in html
    assert "/api/admin/evaluations/runs?limit=20" in html
    assert "/api/admin/evaluations/golden-dataset${query}" in html


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


def test_admin_navigation_information_architecture_is_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert 'class="tab-group" data-group="Knowledge"' in html
    assert 'class="tab-group" data-group="Operations"' in html
    assert 'class="tab-group" data-group="System"' in html
    assert "tab-group-label" in html
    assert "mobile-module-switcher" in html
    assert 'id="adminModuleSelect"' in html
    assert "document.getElementById('adminModuleSelect').addEventListener('change'" in html
    assert "option.disabled = !allowed" in html
    assert "group.hidden = !hasVisibleTab" in html
    assert "kb-danger-zone" in html
    assert "Danger Zone" in html
    assert "Destructive Knowledge Base actions live here" in html
    assert 'id="btnDeleteKb"' in html


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
    assert "knowledgeFlow" in html
    assert "knowledge-stage" in html
    assert "knowledgeStageSetupSummary" in html
    assert "knowledgeStageSourcesSummary" in html
    assert "knowledgeStageIndexingSummary" in html
    assert "knowledgeStageQualitySummary" in html
    assert "data-knowledge-stage-target" in html
    assert "function renderKnowledgeFlow" in html
    assert "function scrollToKnowledgeStage" in html
    assert "knowledgeBulkSelectAll" in html
    assert "btnKnowledgeBulkIngest" in html
    assert "btnKnowledgeBulkReviewed" in html
    assert "btnKnowledgeBulkDetach" in html
    assert "data-knowledge-file-select" in html
    assert "function renderKnowledgeBulkState" in html
    assert "function bulkIngestKnowledgeFiles" in html
    assert "function bulkMarkKnowledgeFilesReviewed" in html
    assert "function bulkDetachKnowledgeFiles" in html
    assert "fileDetailDrawer" in html
    assert "fileDetailTitle" in html
    assert "fileDetailBody" in html
    assert "btnCloseFileDetail" in html
    assert "data-file-detail" in html
    assert "data-file-detail-source" in html
    assert "function findFileDetail" in html
    assert "function fileDetailHtml" in html
    assert "function openFileDetail" in html
    assert "function closeFileDetail" in html
    assert "document.getElementById('fileDetailBody').addEventListener('click'" in html
    assert "knowledgeProgressPanel" in html
    assert "knowledgeIndexProgressBar" in html
    assert "knowledgeProgressSummary" in html
    assert "knowledgeNextActions" in html
    assert "knowledgeProgressNeedsIngest" in html
    assert "function knowledgeProgressStats" in html
    assert "function knowledgeNextActionsHtml" in html
    assert "function renderKnowledgeProgress" in html
    assert "function fileNextActionNote" in html
    assert "knowledgeWorkspaceKbFilesTable" in html
    assert "knowledgeWorkspaceLibraryTable" in html
    assert "btnViewAllKbFiles" in html
    assert "btnViewAllLibraryFiles" in html
    assert "view-file-browser" in html
    assert "fileBrowserTable" in html
    assert "fileBrowserSearch" in html
    assert "function openFileBrowser" in html
    assert "Review & Quality Details" in html
    assert "view-knowledge-details" in html
    assert "btnOpenReviewQueueDetails" in html
    assert "btnOpenQualityWorkflowDetails" in html
    assert "btnOpenSourceQualityDetails" in html
    assert "btnKnowledgeDetailsBack" in html
    assert "function openKnowledgeDetails" in html
    assert "knowledgeDetailsReviewQueue" in html
    assert "knowledgeDetailsQualityWorkflow" in html
    assert "knowledgeDetailsSourceQuality" in html
    assert "data-file-menu" in html
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
    assert "function showFileVersions" in html
    assert "function replaceFileContent" in html
    assert "function rollbackFileVersion" in html
    assert "function showFileVersionDiff" in html
    assert "/api/files/${fileId}/content" in html
    assert "/api/files/${fileId}/versions" in html
    assert "data-kb-lifecycle" in html
    assert "data-kb-diff" in html
    assert "data-file-versions" in html
    assert "data-file-replace" in html
    assert "data-file-rollback" in html
    assert "data-file-version-diff" in html
    assert "renderKnowledgeWorkspace" in html
    assert "'view-knowledge-workspace'" in html
    assert "'view-knowledge-details'" in html
    assert "'view-file-browser'" in html
    assert ".png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp" in html


def test_admin_workspace_form_style_and_view_all_tables_are_wired():
    html = (ROOT / "static" / "admin.html").read_text(encoding="utf-8")

    assert "--brand:#007d84" in html
    assert ".tabs { position:fixed;" in html
    assert ".knowledge-strip { display:grid; grid-template-columns:repeat(5,minmax(0,1fr));" in html
    assert ".card-footer-action" in html
    assert ".link-action" in html
    assert "All Files In Selected KB" in html
    assert "All Source Library Files" in html
    assert "document.getElementById('btnViewAllKbFiles').addEventListener('click', () => openFileBrowser('kb'))" in html
    assert "document.getElementById('btnViewAllLibraryFiles').addEventListener('click', () => openFileBrowser('library'))" in html
    assert "document.querySelector('#fileBrowserTable tbody').addEventListener('click'" in html


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
    assert "/api/admin/support-tickets/${selectedSupportCaseId}/canned-action" in html
    assert "workspaceNextAction" in html
    assert "workspaceDraftReview" in html
    assert "workspaceCannedAction" in html
    assert "refund_requires_approval" in html
    assert "cancel_requires_approval" in html
    assert "function renderNextActionCard" in html
    assert "function renderDraftReviewPacket" in html
    assert "function applySupportCannedAction" in html
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
    assert "MCP Quotas" in html
    assert "mcpQuotaTable" in html
    assert "MCP Deny Audit" in html
    assert "mcpDenyTable" in html
    assert "mcpDryRun" in html
    assert "mcpRecentDenies" in html
    assert "tool_dry_run" in html
    assert "quota_dashboard" in html
    assert "recent_denies" in html
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
