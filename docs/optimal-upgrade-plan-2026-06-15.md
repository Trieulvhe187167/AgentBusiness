# Ke hoach cai tien toi uu cho Agent for Business

Date: 2026-06-15

## Tom tat dieu hanh

Du an hien tai da vuot qua muc chatbot RAG co ban. Nen dinh vi san pham thanh **Business AI Operations Agent**: hoi dap co citation, quan tri tri thuc, dieu phoi ticket/email, phe duyet hanh dong rui ro, evaluation gate, va quan sat chat luong van hanh.

Huong toi uu khong phai viet lai. Huong dung la:

1. Bien cac nang luc RAG/eval/workflow da co thanh trai nghiem san pham on dinh.
2. Tang chat luong truy xuat bang benchmark, Qdrant hybrid, reranker, source diversification, va corrective RAG.
3. Dong goi support workflow thanh vong lap tu chat -> ticket -> agent draft -> approval -> customer reply.
4. Nang cap observability va governance de moi thay doi model, cache, tool deu do duoc.
5. Chi mo rong sang multimodal/visual RAG sau khi text RAG co baseline tot.

## Hien trang tu codebase

### Diem manh da co

- FastAPI app co cac surface `/chat`, `/admin`, `/portal`.
- RAG da co KB scope, ingestion da dinh dang, citations, query expansion, conversation memory, knowledge gaps.
- Vector backend da co `numpy`, `chroma`, va `qdrant`; Qdrant hybrid la opt-in.
- Embedding provider profile da ho tro `sentence_transformers`, `tei`, `openai_compatible`, instruction/prefix, fingerprint, dimension guard.
- Reranker da co `bm25_lite` va `cross_encoder`; model mac dinh la `Qwen/Qwen3-Reranker-0.6B`.
- Evaluation da co golden dataset, retrieval metrics, gate regression, latency aggregate, va optional LLM-as-judge.
- Agent runtime da co routing, tool registry, audit logs, pending actions, approval events, durable agent/workflow runs.
- Admin UI da co RAG Quality cockpit, Operations workspace, Knowledge flow, support case workspace.
- OpenAI Responses da co streaming, tool continuation opt-in, token/cached-token logging, prompt cache controls.

### Diem can uu tien tiep

- Chat UX van con dev-heavy: can an dev auth fields, collapse citation/trace khi chua co cau tra loi, them feedback reason flow.
- Portal support flow can ro hon ve ticket timeline, SLA, va chuyen "cau tra loi chua du" thanh ticket.
- Shared design system chua tach khoi 3 file HTML lon.
- RAG quality da co nhieu control nhung can benchmark thuc te va production profile mac dinh.
- Observability co OpenTelemetry foundation nhung chua co dashboard/debug workflow chuan cho latency, retrieval, reranker, LLM, tool.
- Visual document RAG moi nen la track rieng, khong nen chen vao core text RAG ngay.

## Nghien cuu ngoai va ham y

- OpenAI khuyen dung Responses API khi mot model call kem tools/application logic la du, va dung Agents SDK khi ung dung so huu orchestration, tool execution, approvals, state. Dieu nay khop voi kien truc hien tai: giu server-owned orchestration, chi them adapter khi can.
- OpenAI tools hien co huong built-in tools, function calling, tool search, remote MCP; file search la optional hosted RAG backend. Voi du an nay, hosted file search nen la backend phu, khong thay local tenant-scoped KB.
- OpenAI prompt caching giam latency va chi phi khi prompt prefix lap lai; du an da co cached-token logging, nen viec tiep theo la prompt discipline va dashboard cache hit/cost saved.
- Qdrant hybrid search dung dense + sparse + RRF, sau do reranking. Day la huong phu hop nhat cho tai lieu business co ma don hang, policy name, keyword chinh xac, va paraphrase song ngu.
- MCP specification va security guidance nhan manh least privilege, authorization, tools/resources scoped. MCP cua du an da dung dung huong, can tiep tuc them compatibility va security test khi expose them tool.
- Phoenix/Arize tap trung tracing qua OpenTelemetry, eval, datasets, experiments; co the dung lam backend quan sat neu muon dashboard AI ops chuyen nghiep.
- Qwen3 Embedding/Reranker, BGE-M3, Jina Embeddings v4 la cac ung vien retrieval hien dai. Qwen3/BGE-M3 phu hop text multilingual; Jina/ColPali phu hop visual document track.

Nguon tham khao:

- OpenAI Agents SDK: https://developers.openai.com/api/docs/guides/agents
- OpenAI tools: https://developers.openai.com/api/docs/guides/tools
- OpenAI file search: https://developers.openai.com/api/docs/guides/tools-file-search
- OpenAI prompt caching: https://developers.openai.com/api/docs/guides/prompt-caching
- Qdrant hybrid queries: https://qdrant.tech/documentation/search/hybrid-queries/
- Qdrant hybrid search with reranking: https://qdrant.tech/documentation/tutorials-basics/reranking-hybrid-search/
- MCP specification 2025-06-18: https://modelcontextprotocol.io/specification/2025-06-18
- MCP authorization: https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization
- MCP security best practices: https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices
- Phoenix docs: https://arize.com/docs/phoenix
- Phoenix LLM evals: https://arize.com/docs/phoenix/evaluation/llm-evals
- Qwen3 Embedding: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen3 Reranker: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- Jina Embeddings v4: https://huggingface.co/jinaai/jina-embeddings-v4
- Hugging Face multimodal RAG cookbook: https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms

## Roadmap uu tien

### Nguyen tac van hanh bo sung

- Golden dataset va baseline eval phai chay song song voi Sprint 1, khong doi UI polish xong moi bat dau. Baseline co the chay offline ngay voi config hien tai.
- Moi sprint co error budget va rollback plan rieng: neu metric giam qua nguong, rollback config/code nhanh truoc khi tiep tuc mo rong.
- Moi thay doi UX lon phai co it nhat 1 user test session thuc te voi admin/support/employee flow lien quan.
- Moi KPI phai co owner va reporting cadence; neu khong co nguoi chiu trach nhiem do, KPI khong duoc tinh la committed.
- Bat ky MCP/external tool exposure nao cung can security review va penetration test co ban truoc khi mo cho client ngoai.

### Sprint 1: Product polish cho Chat va Portal

Muc tieu: bien app tu "engineering demo rat manh" thanh trai nghiem nguoi dung that.

Cong viec:

- An dev auth fields trong `/chat` neu khong o dev/admin mode; dua vao settings drawer.
- Collapse citations/agent trace panel cho den khi co cau tra loi dau tien.
- Them feedback reason modal: wrong source, outdated, incomplete, hallucination, too slow, other.
- Them nut "Cau tra loi chua du? Tao ticket" tu bot answer sang portal/support ticket.
- Portal ticket detail them timeline ro: created, classified, assigned, agent drafted, waiting approval, replied, resolved.
- Hien SLA bang ngon ngu nhan vien: "Dang xu ly", "Can bo sung thong tin", "Dang cho phe duyet", "Qua han".
- Chay 1 user test session thuc te sau prototype dau tien:
  - 1 admin/knowledge operator,
  - 1 support operator,
  - 1 employee/end user.
- Ghi lai friction points: lan dau vao chat, tim citation, gui feedback, tao ticket, xem ticket status.

Acceptance criteria:

- Nguoi dung thuong khong thay header/internal auth concepts.
- Moi negative feedback co reason_code co the dung cho knowledge gap/eval.
- Tu chat co the tao ticket kem cau hoi, cau tra loi, citations, session_id, chat_log_id.
- User test xac nhan it nhat 80% task chinh hoan thanh khong can huong dan truc tiep.
- Cac UX issue P0/P1 tu user test duoc log thanh backlog truoc khi Sprint 1 dong.

### Sprint 2: Production RAG profile va benchmark bat buoc

Muc tieu: moi thay doi retrieval/model/cache deu co bang chung tot-xau.

Cong viec:

- Chay song song voi Sprint 1: lap golden dataset va baseline eval khong can doi UI.
- Chay baseline golden eval tren config hien tai, luu run ID lam moc.
- Tao preset `local_cpu`, `local_gpu`, `production_qdrant_hybrid`.
- Dat Qdrant hybrid + source diversification + reranker thanh production profile opt-in.
- Them calibration wizard trong RAG Quality: chon baseline, chay config moi, so sanh recall_at_k, MRR, citation_accuracy, answer score, latency p95.
- Them "promote config" flow: chi cho promote neu gate pass hoac co override reason.
- Them seeded golden dataset template 80-150 cau hoi cho business support song ngu.
- Them error budget cho promote RAG config:
  - `golden_avg_score` khong giam qua 2 diem.
  - `citation_accuracy` khong giam qua 3 diem.
  - `recall_at_k` khong giam qua 3 diem.
  - `latency_p95_ms` khong tang qua 25% neu khong co override.
- Them rollback plan:
  - Luu previous config snapshot, collection name, embedding fingerprint, reranker provider/model, thresholds.
  - Co nut/command "revert to previous production RAG profile".
  - Neu index moi loi, quay ve collection cu va disable candidate profile.

Acceptance criteria:

- Khong promote embedding/reranker/vector backend moi neu chua co eval comparison.
- RAG Quality hien ro latency p50/p95 theo retrieval, reranker, LLM, cache.
- Production profile co tai lieu `.env` mau va rebuild index checklist.
- Moi promoted config co rollback target va owner chap thuan.
- Baseline eval dau tien duoc tao trong Sprint 1/2 song song, truoc khi bat ky config retrieval moi nao duoc promote.

### Sprint 3: Knowledge governance loop

Muc tieu: bien knowledge gaps va feedback thanh viec can lam cho admin.

Trang thai trien khai batch 1:

- Da chuan hoa queue status theo vong doi `new`, `triaged`, `source_needed`, `patch_pending`, `fixed`, `ignored`.
- Da them owner, priority, due date, status reason cho knowledge gap cluster.
- Feedback down tu chat tu dong tao knowledge review queue item, khong phu thuoc top_score cao/thap.
- Suggest FAQ chuyen gap sang `patch_pending` va tao pending action `create_faq_entry` de admin approve.
- Da them API quality debt theo KB: active/overdue gaps, patch actions, stale docs, failed ingest, zero chunk.
- Weekly knowledge gap report da gom them quality debt summary.
- Admin UI da hien quality debt, owner, priority, due date va action triage/source-needed/fixed/ignored.

Cong viec:

- Gom feedback down + fallback + low retrieval score thanh "Knowledge Review Queue".
- Them workflow: gap -> suggested FAQ/source patch -> pending action -> admin approve -> attach/update source -> reingest -> eval affected questions.
- Them owner, priority, due date cho gap cluster.
- Them stale document policy: source changed, old version active, failed ingest, zero chunk, low quality.
- Weekly digest cho knowledge operator: top gaps, stale docs, failed jobs, eval regressions.

Acceptance criteria:

- Moi gap co trang thai: new, triaged, source_needed, patch_pending, fixed, ignored.
- Fix gap co the tao source update va kick off reingest/eval.
- Admin co the xem "quality debt" theo KB.

### Sprint 4: Support automation end-to-end

Muc tieu: dung agent de giam thoi gian xu ly ticket nhung van co approval boundary.

Trang thai trien khai batch 1:

- Da them `next_action` cho support ticket/list/context de operator thay buoc tiep theo: assign owner, generate draft, review approval, SLA breach, human follow-up.
- Draft reply output da co review packet gom evidence used, customer-facing reply, internal risk, va approval boundary.
- Admin Support Workspace da hien next action, draft review packet, citations, va canned action controls.
- Da them canned actions:
  - ask for more info -> public note + waiting_customer,
  - resolve with KB answer -> public resolution + resolved,
  - escalate to team -> assign/escalate with note,
  - refund/cancel requires approval -> tao `support_case_review` pending action va chuyen case sang waiting_approval.
- High-risk canned actions khong execute truc tiep; van di qua pending approval queue.

Cong viec:

- Support case timeline lam surface chinh trong Admin workspace.
- Auto-draft reply dua tren KB citations, email/thread context, ticket notes, va policy.
- Draft diff: hien "evidence used", "customer-facing reply", "internal risk".
- Approval queue uu tien theo risk, SLA, customer impact.
- Auto-resume workflow sau approval/execution da co nen can UI lam ro "next action".
- Them canned action: ask for more info, escalate to team, resolve with KB answer, refund/cancel requires approval.

Acceptance criteria:

- Ticket tu email/chat co du context de support agent khong phai mo nhieu panel.
- Moi outbound reply deu co audit trail va approval neu risk cao.
- SLA breach tao notification va escalation package.

### Sprint 5: Observability va AI Ops

Muc tieu: debug duoc tai sao cau tra loi cham/sai/ton tien.

Trang thai trien khai batch 1:

- Da them AI Ops summary API `/api/admin/ai-ops/summary` gom cost/token, cached-token reuse, p50/p95 latency, tool error budget, approval backlog age, eval gate trend, va alert severity.
- Da them safe replay API `/api/admin/ai-ops/replay/chat-logs/{chat_log_id}` voi mode `retrieval_only`, khong goi LLM, khong tao side-effect, co redaction policy cho user/contact/tenant/org identifiers.
- Admin Analytics da co AI Ops Snapshot: billable input estimate, cache reuse, p95 latency, tool error budget, approval age, latest eval gate, alert cards, eval gate trend table.
- Admin Analytics da co Safe Chat Replay: nhap chat_log_id va top_k de chay lai retrieval, xem top_score/predicted_mode/result snippets da redact.
- Da them test API cho AI Ops alerts, eval trend, replay redaction, va role guard; UI wiring duoc bao ve trong static regression test.

Cong viec:

- Chuan hoa OpenTelemetry spans: request, retrieval, vector query, reranker, corrective rewrite, LLM, tool, workflow step.
- Them local trace viewer hoac Phoenix exporter profile.
- Admin analytics them cost/cached-token estimates, cache hit rate, tool failure rate, approval wait time, eval trend.
- Them request replay debug: tu chat_log_id chay lai retrieval-only, answer-only, eval-only.
- Them redaction policy cho traces va logs.

Acceptance criteria:

- Mot cau tra loi cham co the truy ra cham o retrieval, reranker, LLM, tool, hay queue.
- Trace khong leak secret/contact direct identifiers.
- Eval failures link toi trace/chat/citation.

### Sprint 6: MCP va external ecosystem hardening

Muc tieu: expose tool cho external client an toan va de dung.

Trang thai trien khai batch 1:

- Da them `tools/dryRun` cho MCP JSON-RPC: validate MCP policy, internal auth, schema, quota status ma khong execute handler va khong ghi tool audit nhu execution that.
- High-risk tool dry-run tra ve decision chi tiet (`high_risk_denied_by_default`) thay vi chay tool; invalid args tra ve validation error trong dry-run result.
- Da mo rong MCP resources/templates:
  - `kb://{kb_id}/source-health`,
  - `support://tickets/recent`,
  - `support://tickets/{ticket_id}/timeline`,
  - `eval://runs/recent`,
  - `eval://runs/{run_id}`.
- Resource reads moi tiep tuc enforce admin role, resource scope, va tenant/org isolation.
- Admin MCP status da co quota dashboard, recent deny audit, va capabilities `tool_dry_run`, `quota_dashboard`, `deny_audit`.
- Admin UI MCP Server da hien MCP Quotas va MCP Deny Audit de operator thay client nao bi quota/deny va ly do.
- Test suite da them coverage cho dry-run khong consume quota, policy/validation denial, resource templates moi, support/eval/source-health reads, quota dashboard, va deny audit.

Cong viec:

- Tool descriptions viet lai de model chon tool dung, khong theo kieu marketing.
- Them resource templates huu ich: KB summary, source health, ticket context, eval run detail.
- Version compatibility tests cho MCP spec moi neu nang tu 2025-06-18.
- Them per-tool quota dashboard va deny reason.
- Them "dry run" cho high-risk tools.
- Them penetration test co ban cho MCP/external tool surface:
  - auth bypass,
  - tenant/org scope bypass,
  - prompt/tool injection,
  - over-broad resource discovery,
  - quota bypass,
  - high-risk tool execution without approval.

Acceptance criteria:

- External MCP client chi thay tool/resource duoc scope.
- High-risk tool khong execute truc tiep neu chua approve.
- Admin xem duoc MCP client, quota, failures, denied scopes.
- MCP security test suite va manual pentest checklist pass truoc khi expose them external client.

### Sprint 7: Visual document RAG track

Muc tieu: xu ly PDF scan, bang bieu, invoice, catalogue, slide co layout phuc tap.

Cong viec:

- Tao visual index phu thay vi thay core text RAG.
- Benchmark Jina Embeddings v4 hoac ColPali/ColQwen cho PDF/image page retrieval.
- Luu page image thumbnails, visual hit metadata, bounding/page citation.
- Merge evidence: text chunk + visual page hit + table row.
- Chi bat cho KB co nhieu scanned PDF/form/table image.

Acceptance criteria:

- Cau hoi ve bang/hinh/page layout co citation den page/region.
- Visual track khong lam cham text-only KB.
- Co cost/latency guard rieng.

## Thu tu uu tien de xuat

| Uu tien | Hang muc | Ly do |
| --- | --- | --- |
| P0 | Chat/Portal product polish | Tac dong truc tiep den nguoi dung va demo. |
| P0 | Mandatory eval benchmark before RAG changes | Tranh cai tien cam tinh lam chat luong giam. |
| P1 | Knowledge governance loop | Bien feedback/gap thanh vong lap cai thien du lieu. |
| P1 | Support automation end-to-end | Gan voi gia tri business ro: giam thoi gian xu ly ticket. |
| P1 | AI Ops observability | Can de van hanh production va debug loi. |
| P2 | MCP ecosystem polish | Tot cho mo rong, nhung phai sau khi tool surface on dinh. |
| P2 | Visual document RAG | Gia tri cao neu data co scan/form, nhung ton compute va complexity. |

## KPI can theo doi

| KPI group | Metrics | Owner de xuat | Cadence |
| --- | --- | --- | --- |
| Answer quality | golden avg score, groundedness, citation accuracy, recall_at_k, MRR | AI/RAG owner | Weekly, va truoc/sau moi RAG config change |
| Retrieval | no-result rate, low-confidence rate, reranker improvement delta | AI/RAG owner | Weekly |
| UX | chat-to-ticket conversion, feedback down rate, feedback reason mix, user test completion rate | Product/UX owner | Moi sprint |
| Support | first response time, resolution time, approval wait time, SLA breach count | Support ops owner | Daily ops, weekly review |
| Ops | failed job rate, stale document count, reingest success rate | Platform/ops owner | Daily |
| Cost/latency | p50/p95 latency, LLM tokens, cached token ratio, cache hit rate | Platform/AI ops owner | Weekly, daily neu production traffic cao |
| Safety | denied tool calls, pending approval count, high-risk execution without approval = 0 | Security/approval owner | Weekly, immediate alert cho violation |

## Error budget va rollback template

Moi sprint can dien cac truong nay truoc khi bat dau:

- Scope: feature/config nao se thay doi, surface nao bi anh huong.
- Primary metric: metric nao phai tot hon hoac khong duoc giam.
- Error budget: nguong giam/tang chap nhan duoc, vi du score -2 diem hoac latency +25%.
- Rollback trigger: dieu kien nao bat buoc revert.
- Rollback target: commit/config/index/collection/profile nao quay ve.
- Owner: ai quyet dinh promote, ai quyet dinh rollback.
- Verification: test/eval/user test nao phai pass sau rollback.

## Quyet dinh kien truc nen giu

- Giu backend-owned ToolRegistry, authorization, audit, pending action, workflow engine.
- Khong migrate sang framework agent moi trong ngan han; chi them adapter sau khi contract noi bo on dinh.
- Giu local/tenant-scoped KB la source of truth; hosted file search chi la backend optional.
- Khong bat multimodal mac dinh; chay visual document RAG nhu track rieng.
- Moi model/index change phai qua golden eval va config snapshot.

## Next sprint cu the nen lam

1. Chat/Portal polish: hide dev controls, collapse trace, feedback reasons, create ticket from answer.
2. Golden dataset baseline song song: tao 80-150 Q&A, chay baseline eval, luu config snapshot va run ID.
3. RAG Quality calibration wizard: baseline vs candidate config, promote config with gate va rollback target.
4. Knowledge Review Queue: combine negative feedback, knowledge gaps, stale docs, failed ingest.
5. Support timeline upgrade: one case view with draft reply, evidence, approval state, SLA.

Day la goi viec can bang nhat: vua tang trai nghiem demo, vua tao nen tang do luong chat luong, vua dua du an gan hon voi business value that.
