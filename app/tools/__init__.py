"""
Default tool registry for the AgentBusiness upgrade path.
"""

from __future__ import annotations

from app.tools.admin_tools import build_get_kb_stats_tool, build_list_kbs_tool
from app.tools.business_tools import (
    build_find_recent_orders_tool,
    build_get_online_member_count_tool,
    build_get_order_status_tool,
)
from app.tools.drive_tools import (
    build_create_google_drive_source_tool,
    build_delete_google_drive_source_tool,
    build_get_google_drive_sync_status_tool,
    build_list_google_drive_sources_tool,
    build_sync_google_drive_source_tool,
)
from app.tools.email_tools import (
    build_create_ticket_from_email_tool,
    build_list_support_emails_tool,
    build_read_email_thread_tool,
    build_send_email_reply_tool,
)
from app.tools.kb_tools import build_search_kb_tool
from app.tools.registry import ToolRegistry
from app.tools.support_tools import build_create_support_ticket_tool


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_search_kb_tool())
    registry.register(build_create_support_ticket_tool())
    registry.register(build_get_order_status_tool())
    registry.register(build_find_recent_orders_tool())
    registry.register(build_get_online_member_count_tool())
    registry.register(build_list_kbs_tool())
    registry.register(build_get_kb_stats_tool())
    registry.register(build_list_google_drive_sources_tool())
    registry.register(build_create_google_drive_source_tool())
    registry.register(build_sync_google_drive_source_tool())
    registry.register(build_get_google_drive_sync_status_tool())
    registry.register(build_delete_google_drive_source_tool())
    registry.register(build_list_support_emails_tool())
    registry.register(build_read_email_thread_tool())
    registry.register(build_create_ticket_from_email_tool())
    registry.register(build_send_email_reply_tool())
    return registry


tool_registry = build_default_tool_registry()

__all__ = ["build_default_tool_registry", "tool_registry"]
