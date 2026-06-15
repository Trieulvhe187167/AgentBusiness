# UI/UX Audit & Design Inventory - 2026-06

## Scope

This audit covers the current local UI surfaces for Agent Business / CampusRAG:

- Admin operations UI: `/admin`
- Chat UI: `/chat`
- Internal employee portal: `/portal`

The goal is to create a Phase 0 design inventory and UX backlog before changing production UI.

## Capture Notes

Screenshots were captured locally with Microsoft Edge headless against `http://127.0.0.1:8080`.

Runtime used for capture:

- `RAG_LLM_PROVIDER=none`
- hashing embedding fallback
- `RAG_VECTOR_BACKEND=numpy`

This keeps UI capture lightweight. It means model/provider health shown in screenshots is an audit-time state, not a production recommendation.

## Evidence

| Step | Screenshot | Surface | Health |
| --- | --- | --- | --- |
| 1 | [01-admin-knowledge-workspace.png](screenshots/01-admin-knowledge-workspace.png) | Admin Knowledge Workspace | Functional, dense, action-heavy |
| 2 | [02-admin-analytics-evaluations.png](screenshots/02-admin-analytics-evaluations.png) | Admin Analytics/Evaluations | Functional, but quality and eval concepts compete |
| 3 | [03-admin-system-runtime.png](screenshots/03-admin-system-runtime.png) | Admin System Runtime | Useful for debugging, too raw for daily ops |
| 4 | [04-chat-rag-surface.png](screenshots/04-chat-rag-surface.png) | Chat | Clear first-use surface, trace/citation side panel underused before chat |
| 5 | [05-internal-support-portal.png](screenshots/05-internal-support-portal.png) | Internal Portal | Good task split, support list hierarchy needs tightening |
| 6 | [06-admin-mobile-knowledge.png](screenshots/06-admin-mobile-knowledge.png) | Admin mobile viewport | Usable start, but important text and nav overflow |

## Design Inventory

### Current Visual Language

- Admin uses a light warm operational theme:
  - `--bg:#fbf7ef`
  - `--panel:#fffdf8`
  - `--brand:#007d84`
  - `--radius:10px`
- Chat uses a separate dark assistant theme:
  - `--bg:#0f1117`
  - `--brand:#4f8ef7`
  - right-side citations/trace panel
- Internal portal uses a softer employee-support theme:
  - `--bg:#eef3ea`
  - `--panel:#fffdf7`
  - `--brand:#0f6f64`
  - `--radius:20px`

The three surfaces feel related, but not yet like one cohesive product. Admin and Portal share warm neutrals; Chat diverges strongly into a dark technical product.

### Reusable UI Patterns Already Present

- Fixed admin sidebar with role-aware module buttons
- Top identity shell / dev identity editor
- KB selector and KB status badges
- Metric cards / stat tiles
- Tables with sticky headers
- Status pills for jobs, tickets, actions, feedback, and eval states
- Large operational cards
- Upload/drop zones
- Chat messages with citation and trace panels
- Support ticket create/list/detail pattern
- Toast notifications

### Main Product Workflows

1. Knowledge operator:
   - select KB
   - upload or attach source
   - ingest/reindex
   - inspect source quality
2. RAG quality operator:
   - run search calibration
   - inspect runtime config
   - create golden dataset
   - run eval and inspect failures
3. Support operator:
   - inspect tickets/emails
   - classify/handle/escalate
   - reply or resolve
4. Employee:
   - ask KB question
   - create support ticket
   - track own tickets
5. System/admin:
   - inspect runtime readiness
   - manage users/access
   - inspect logs/jobs/MCP/webhooks

## Key Findings

### P0 - Fix Before Large UI Expansion

1. Admin mobile text overflow
   - Evidence: Step 6.
   - The `KB Version` value overflows its stat card and visually collides with the viewport edge.
   - Impact: mobile/tablet admin users cannot trust stat cards for long identifiers.
   - Recommendation: truncate long hashes with middle ellipsis, add copy action, and keep full value in tooltip/details.

2. Admin mobile navigation is visually clipped
   - Evidence: Step 6.
   - Bottom of the screenshot shows horizontal nav content partially clipped.
   - Impact: users may not realize all modules are available or may have trouble switching sections.
   - Recommendation: replace mobile horizontal tab strip with a compact module menu or sticky segmented drawer.

3. System runtime is too raw for most operators
   - Evidence: Step 3.
   - Runtime JSON is useful, but it dominates the System page without a summarized health layer.
   - Impact: non-engineer admins must parse raw JSON to answer basic questions like "is RAG healthy?"
   - Recommendation: add a "Runtime Summary" above JSON with provider, embedding fingerprint, reranker, corrective RAG, vector backend, cache, and budget statuses.

4. RAG quality signals are split across Analytics, Calibrate, System, and raw JSON
   - Evidence: Steps 2 and 3.
   - Eval, runtime budget, provider state, and retrieval quality are not presented as one cockpit.
   - Impact: after model/reranker changes, it is hard to understand whether quality improved.
   - Recommendation: create a dedicated "RAG Quality" view or subnav inside Analytics.

### P1 - Improve Daily Operator Flow

5. Admin Knowledge Workspace is powerful but crowded
   - Evidence: Step 1.
   - KB selector, global stats, create KB, selected controls, ingest actions, upload, source library, Drive, and quality controls all appear in one long operational page.
   - Impact: new admins may not know the next best action.
   - Recommendation: split into task bands:
     - Setup: create/select KB
     - Sources: upload/attach/sync
     - Indexing: ingest/reindex/job status
     - Quality: stale docs/review queue/source health

6. Primary and destructive actions sit close together
   - Evidence: Step 1.
   - `Set Default` and `Delete KB` are adjacent in the selector area.
   - Impact: higher risk of accidental destructive action.
   - Recommendation: keep destructive KB actions inside an overflow menu or danger zone with confirmation.

7. Analytics headline metric is not contextual enough
   - Evidence: Step 2.
   - System health score is prominent, but the reason behind high average latency is not actionable.
   - Impact: users see a large latency number without knowing whether retrieval, reranker, LLM, or cache caused it.
   - Recommendation: add latency breakdown cards matching Phase 6 fields: embedding, vector query, reranker, LLM, cache hit.

8. Empty/loading states are generic
   - Evidence: Steps 1, 2, and source code inventory.
   - Many tables say "No files loaded", "No analytics loaded", or "No evaluation runs loaded".
   - Impact: unclear whether the user should click refresh, create data, change KB, or fix an error.
   - Recommendation: use action-specific empty states with one next action.

9. Chat dev controls are too prominent for normal users
   - Evidence: Step 4.
   - `X-User-Id`, `X-Roles`, KB, and language controls occupy large composer space.
   - Impact: production users may see internal concepts before their first message.
   - Recommendation: hide dev identity fields behind a settings drawer unless auth mode is dev and user is admin.

10. Chat trace and citations panels start empty but visually heavy
    - Evidence: Step 4.
    - The right panel takes around one quarter of the page before any answer exists.
    - Impact: initial state feels more technical than helpful.
    - Recommendation: collapse trace panel by default before the first answer, or show a short explanation of what will appear there after a response.

### P2 - Polish And Consistency

11. Product naming is inconsistent
    - Evidence: Admin says `CampusRAG Admin`, Chat says `Business Agent`, Portal says `Internal User Portal`.
    - Impact: users may not recognize these as one system.
    - Recommendation: choose one product umbrella and use role-specific subtitles.

12. Icon language is inconsistent
    - Evidence: Admin uses letter icons, Chat uses text/icon buttons, Portal uses mostly text.
    - Impact: navigation feels handcrafted rather than systematized.
    - Recommendation: adopt a small icon set and define module icons once.

13. Cards are sometimes doing page-section work
    - Evidence: Admin and Portal.
    - Impact: too many nested/stacked borders reduce scan speed.
    - Recommendation: reserve cards for repeated objects or bounded tools; use page bands/sections for larger workflows.

14. Status colors are mostly consistent but not centralized
    - Evidence: CSS across `admin.html`, `chat.html`, and `internal.html`.
    - Impact: future states may drift.
    - Recommendation: extract shared status tokens and component rules.

## Accessibility Risks Visible From Screenshots

- Some link/button labels rely on color and proximity rather than clear grouping, especially in Admin top-right links.
- Small uppercase table headers may be hard to read for low-vision users.
- Dense tables likely need keyboard focus review; screenshots cannot verify tab order.
- Chat and Admin use strong color contrast in many places, but several muted notes are small and low-emphasis.
- Mobile Admin needs text wrapping/truncation fixes for long identifiers.
- Full compliance cannot be claimed from screenshots alone; keyboard navigation, focus order, screen reader names, and form validation need interactive testing.

## Recommended UI Roadmap

## Implementation Status

### Completed In Current UI Upgrade

UI Phase 1 - RAG Quality Cockpit has been implemented as a first-class Admin module.

Evidence in code:

- `static/admin.html`
  - Adds the `RAG Quality` module in the Admin sidebar.
  - Adds the `RAG Quality & Evaluation Cockpit` view.
  - Shows quality gate, runtime readiness, eval coverage, latency budget, retrieval stack, runtime budget controls, latest eval metric breakdown, baseline regression gate, latest runs, fail/warn inspector, and golden dataset coverage.
  - Adds refresh and run-golden-eval actions.
- `app/main.py` and `app/models.py`
  - Expose reranker provider/model/top_n, retrieval mode, and Qdrant hybrid status from `/api/system`.
- `tests/ui/test_phase39_feedback_ui.py`
  - Locks the cockpit wiring, tables, actions, and endpoint usage.
- `tests/agent/test_phase11_runtime_contract.py`
  - Locks the runtime fields used by the cockpit.

Verification already run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ui\test_phase39_feedback_ui.py tests\agent\test_phase11_runtime_contract.py tests\api\test_phase47_agent_evaluations.py
```

Result: 25 passed.

Interactive browser verification:

- Started the FastAPI app on `http://127.0.0.1:8080`.
- Opened `/admin` in the in-app browser.
- Activated the `RAG Quality` module and verified the active view switched to `view-rag-quality`.
- Confirmed the cockpit rendered embedding, reranker, vector backend, corrective RAG cards, control buttons, and evaluation tables.
- Confirmed no local app console errors were reported during this check.

UI Phase 2 - Admin Navigation And Information Architecture has also been implemented in the Admin shell.

Evidence in code:

- `static/admin.html`
  - Groups the Admin sidebar into Knowledge, Operations, and System sections.
  - Adds a mobile `Admin Module` selector so small viewports no longer depend on a clipped horizontal tab strip.
  - Keeps role-aware module gating for both desktop buttons and mobile select options.
  - Moves `Delete KB` into a dedicated Knowledge Base `Danger Zone`.
- `tests/ui/test_phase39_feedback_ui.py`
  - Locks the grouped navigation, mobile module selector, role-aware option gating, and danger-zone wiring.

UI Phase 3 - Knowledge Workspace Task Flow has a first implementation slice in the Admin shell.

Evidence in code:

- `static/admin.html`
  - Adds a `Knowledge Flow` stage strip with Setup, Sources, Indexing, and Quality stages.
  - Adds jump targets for daily KB setup, source upload/library work, ingest/reindex controls, and quality review.
  - Updates each stage summary from live workspace state: active KB, attached/library/Drive sources, pending ingest attention, and review/quality counts.
  - Adds bulk file selection with ingest selected, mark reviewed, and detach selected actions for files attached to the active KB.
  - Adds a source/file detail drawer for KB and library files, with status, lifecycle, upload, chunk, rows/pages, stale notes, version actions, and attach/detach/ingest actions.
  - Adds indexing progress summary, progress meter, failed/stale/needs-ingest counters, and prioritized next actions.
  - Adds per-file next-action notes so ingest, stale, failed, draft, reviewed, and ready states are easier to act on.
- `tests/ui/test_phase39_feedback_ui.py`
  - Locks the Knowledge Flow stage strip, summaries, jump targets, bulk file controls, file detail drawer, progress summary, next-action notes, and render/click wiring.

Known remaining UI issues from Phase 0:

- Knowledge Workspace now has stage guidance, bulk file actions, source/file detail drawer, and richer indexing progress states.
- Chat still exposes dev controls prominently in local/dev mode.
- Shared design-system extraction is still pending.

### UI Phase 1 - RAG Quality Cockpit

Status: implemented.

Build a dedicated admin section for the backend improvements already implemented:

- current embedding provider/model/fingerprint
- reranker provider/model and fallback status
- vector backend and Qdrant hybrid mode
- corrective RAG status
- runtime budget and latency breakdown
- latest golden eval result and gate status
- failed example inspector
- eval metric breakdown and baseline regression table

### UI Phase 2 - Admin Navigation And Information Architecture

Status: implemented.

- Convert module navigation into grouped sections:
  - Knowledge
  - Operations
  - System
- Add a compact mobile module menu.
- Move dangerous KB actions to a protected danger zone.

### UI Phase 3 - Knowledge Workspace Task Flow

Status: stage flow, bulk file actions, source/file detail drawer, and indexing progress states implemented.

- Add clear task stages: setup, sources, indexing, quality.
- Add bulk file actions.
- Add source/file detail drawer. Done.
- Improve upload/ingest progress and stale-doc explanations. Done for workspace-level indexing progress and per-file next-action notes.

### UI Phase 4 - Chat Experience

- Hide dev controls for normal users.
- Show citations and trace after the first answer, or collapse them by default.
- Add response quality feedback reasons.
- Make corrective RAG and reranker details readable for admins without exposing them to all users.

### UI Phase 5 - Portal Support Flow

- Improve ticket list hierarchy.
- Add clearer ticket detail timeline.
- Connect "answer not enough" from chat to support ticket creation.
- Show SLA/status in plain employee language.

### UI Phase 6 - Shared Design System

- Extract common tokens from the three HTML files.
- Standardize:
  - buttons
  - badges
  - stat cards
  - tables
  - empty states
  - modals/drawers
  - status pills
  - focus states

## Next Implementation Slice

Recommended first build after this audit:

1. Add `RAG Quality` as a first-class Admin module.
2. Start with read-only UI using existing `/api/system`, `/api/admin/evaluations/runs`, `/api/admin/analytics`, and debug retrieval endpoints.
3. Add latency breakdown cards using Phase 6 SSE/runtime data where available.
4. Add a compact "latest eval vs baseline" panel.
5. Add a failed-example table before designing charts.

This gives the newest RAG backend work a visible operational home without disrupting upload/chat/support flows.
