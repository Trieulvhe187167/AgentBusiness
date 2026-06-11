# Ke hoach nang cap RAG agent theo cong nghe moi

Date: 2026-06-05

## Muc tieu

Nang cap chat/RAG agent hien tai theo huong tang do chinh xac truy xuat, giam hallucination, ho tro tot tieng Viet/tieng Anh, va san sang cho tai lieu da phuong thuc nhu PDF scan, bang bieu, hinh anh. Khong nen viet lai toan bo. Nen nang cap theo cac diem mo-rong da co san: `app/embeddings.py`, `app/reranker.py`, `app/vector_store.py`, `app/rag.py`, `app/evaluations.py`, va `app/agent.py`.

## Baseline hien tai cua du an

- Embedding mac dinh: `paraphrase-multilingual-MiniLM-L12-v2`, co hashing fallback.
- Retrieval: numpy, Chroma, va Qdrant tuy chon.
- Qdrant da co hybrid dense+sparse voi named vectors, BM25 sparse model, prefetch, va RRF fusion.
- Reranking hien tai la BM25-lite trong `app/reranker.py`, chua co neural cross-encoder/reranker.
- Query expansion hien tai la rule-based VI/EN synonym trong `app/query_expander.py`.
- Evaluation da co golden dataset, recall, MRR, citation accuracy, LLM-as-judge tuy chon.
- Agent runtime da co route RAG/tool/memory/clarify/fallback, tool audit, pending actions, va agent run steps.

Ket luan: ROI cao nhat la nang cap embedding + neural reranking + eval gate truoc, sau do moi mo rong agentic RAG.

## Phat hien moi va ung vien cong nghe

### 1. Qwen3 Embedding va Qwen3 Reranker

Nguon:

- https://huggingface.co/papers/2506.05176
- https://huggingface.co/Qwen/Qwen3-Embedding-8B
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

Diem dang chu y:

- Co dong embedding va reranking 0.6B, 4B, 8B.
- Ho tro 100+ ngon ngu, paper/community note neu ro 119 languages.
- Context 32K, phu hop document chunk dai va query co nhieu ngu canh.
- Embedding co Matryoshka/flexible output dimensions, co the dung kich thuoc vector nho hon khi can giam storage/latency.
- Reranker co ban 0.6B la ung vien thuc dung de them vao pipeline hien tai ma khong can GPU qua lon.
- Model card khuyen nghi custom instruction cho task/language, co the tang 1-5% trong nhieu retrieval scenario.

Ap dung cho du an:

- Thu nghiem `Qwen/Qwen3-Embedding-0.6B` lam ung vien dau tien vi can bang chat luong/chi phi.
- Thu nghiem `Qwen/Qwen3-Reranker-0.6B` sau khi retrieve top 30-100 candidates tu Qdrant hybrid.
- Neu co GPU tot, benchmark them `Qwen/Qwen3-Embedding-4B` hoac `8B`; khong nen mac dinh ngay.

Rang buoc ky thuat:

- Can kiem tra/pin `transformers>=4.51.0`.
- `sentence-transformers==3.3.1` hien tai co the dung Qwen qua SentenceTransformer/CrossEncoder, nhung phai test warm-up va inference tren Windows.
- Doi embedding model se doi vector dimension, can rebuild vector index hoac dung collection moi.

### 2. BGE-M3 va BGE reranker

Nguon:

- https://huggingface.co/BAAI/bge-m3

Diem dang chu y:

- BGE-M3 van la baseline manh cho multilingual retrieval.
- Mot model co dense, sparse, va ColBERT-style multi-vector signal.
- Ho tro 100+ ngon ngu, sequence length 8192, dimension 1024.
- Model card khuyen nghi pipeline hybrid retrieval + reranking.

Ap dung cho du an:

- Dung `BAAI/bge-m3` lam baseline on-prem de so sanh voi Qwen3.
- Voi Qdrant hien tai, co the tiep tuc dung BM25 sparse cua Qdrant, sau do benchmark them sparse output tu BGE-M3 neu muon nang cap sau.
- Neu Qwen3 chay cham/khong on dinh tren may local, BGE-M3 la fallback san xuat hop ly.

### 3. Jina Embeddings v4 cho multimodal/multilingual retrieval

Nguon:

- https://huggingface.co/jinaai/jina-embeddings-v4

Diem dang chu y:

- Universal embedding model cho multimodal va multilingual retrieval.
- Thiet ke cho complex document retrieval, gom visual documents co charts, tables, illustrations.
- Ho tro text, image, visual document; dense single-vector va late-interaction multi-vector.
- 30+ ngon ngu, max sequence length 32768, dense dimension mac dinh 2048 va Matryoshka dimensions.

Ap dung cho du an:

- Chua nen thay embedding text mac dinh bang Jina v4 ngay.
- Nen tao track rieng cho "visual document RAG" neu du an hay xu ly PDF scan, anh bang bieu, hoa don, catalog.
- Vi current wrapper `SentenceTransformer(source)` chua truyen `trust_remote_code=True`, can them cau hinh rieng neu dung Jina v4 local.

### 4. ColPali / visual document retrieval

Nguon:

- https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms

Diem dang chu y:

- Cach tiep can chuyen PDF thanh anh, index bang ColPali/Byaldi, retrieve trang/tai lieu bang visual relevance.
- Phu hop tai lieu co layout quan trong: bang, hinh minh hoa, catalogue, huong dan lap rap, invoice, form.

Ap dung cho du an:

- Du an da co OCR va image parser, nen co the them visual index phu cho PDF/anh thay vi thay pipeline text.
- Retrieval hop nhat: text chunks tu pipeline hien tai + visual page hits tu ColPali/Jina, sau do rerank/assemble evidence.

### 5. Qdrant hybrid va multi-representation search

Nguon:

- https://qdrant.tech/documentation/search/hybrid-queries/
- https://qdrant.tech/documentation/tutorials-search-engineering/multi-representation-search/

Diem dang chu y:

- Hybrid dense+sparse giup ket hop semantic understanding va exact word matching.
- RRF fusion phu hop khi co nhieu representation cua cung document.
- Multi-representation search co the index title, abstract/summary, body chunks, tags bang cac named vector khac nhau.

Ap dung cho du an:

- Qdrant hybrid da co trong `app/vector_store.py`; can chuyen tu "co tinh nang" sang "co benchmark + default production profile".
- Nen them title/heading/category boosts va group-by-source logic de tranh nhieu chunk trung lap tu cung file.

### 6. Agentic RAG, corrective RAG, evaluation trajectory

Nguon:

- https://docs.langchain.com/oss/python/langgraph/agentic-rag
- https://docs.langchain.com/langsmith/evaluation-approaches

Diem dang chu y:

- Agentic RAG cho phep LLM quyet dinh co retrieve hay tra loi truc tiep.
- Pipeline hay gap: generate query, retrieve, grade documents, rewrite question neu context kem, generate answer.
- Agent evaluation can danh gia final response, single step, va trajectory/tool path.

Ap dung cho du an:

- Du an da co agent route va eval, khong can migrate sang LangGraph ngay.
- Nen them mot CRAG-style node nho vao `app/rag.py`: grade retrieved docs va rewrite/retrieve lai neu top evidence kem.
- Nen them eval cho route/tool decision trong `app/agent.py` truoc khi agent tu dong nhieu buoc hon.

## Kien truc de xuat

Pipeline de xuat cho production RAG:

1. Normalize query + language detection.
2. Query expansion:
   - rule-based hien tai cho slang/diacritics,
   - optional LLM query rewrite khi query ngan, follow-up, hoac retrieval lan dau kem.
3. Candidate retrieval:
   - Qdrant dense top N,
   - Qdrant sparse/BM25 top N,
   - RRF fusion.
4. Optional multi-representation retrieval:
   - chunk text,
   - title/heading/category,
   - visual page hit neu file la PDF/image.
5. Neural reranker:
   - Qwen3-Reranker-0.6B hoac BGE-reranker-v2-m3,
   - rerank top 30-100, tra top 5-10 cho answer.
6. Evidence assembly:
   - dedupe by file/page/row/chunk,
   - cap token budget,
   - uu tien chunk co citation ro rang,
   - group ket qua theo source de tranh trung lap.
7. Answer generation:
   - strict grounded prompt hien tai,
   - claim-level citation check,
   - numeric guardrail hien tai.
8. Corrective loop:
   - neu reranker score/evidence coverage thap, rewrite query va retrieve lai 1 lan,
   - neu van thap, fallback/clarify va record knowledge gap.
9. Evaluation + observability:
   - log retrieval candidates, reranker scores, selected evidence, latency,
   - nightly golden eval truoc/sau model/config change.

## Roadmap trien khai chi tiet

### Phase 0: Snapshot va benchmark baseline

Muc tieu: biet hien tai tot/xau o dau truoc khi doi model.

Viec can lam:

- Tao mot eval run baseline voi embedding hien tai, backend hien tai, top_k hien tai.
- Tang golden dataset len toi thieu 80-150 cau hoi gom:
  - cau hoi tieng Viet co dau/khong dau,
  - cau hoi tieng Anh,
  - product/order/policy exact IDs,
  - cau hoi follow-up,
  - cau hoi can citation theo page/row,
  - negative/unknown questions.
- Luu config cua moi eval run: embedding model, vector backend, hybrid on/off, top_k, reranker provider, thresholds.
- Them report so sanh trong admin/eval output: recall_at_k, MRR, citation_accuracy, answer_similarity, groundedness/judge score, latency p50/p95.

Acceptance criteria:

- Co baseline reproducible.
- Moi thay doi retrieval/model deu co before/after eval.

Implementation baseline:

- Golden eval run `config_json` phai co `rag_config_snapshot` de dong bang embedding model, chunking, vector backend, hybrid settings, top_k, thresholds, answer settings, LLM provider, va cache flags.
- Golden eval aggregate metrics phai co `latency_p50_ms`, `latency_p95_ms`, `latency_avg_ms`, va `retrieved_count_avg`.
- Chay baseline qua API de dung dung auth/routing/retrieval path:

```powershell
python scripts/run_baseline_eval.py --base-url http://127.0.0.1:8080 --kb-id 1 --limit 150
```

- Neu golden dataset chua du 80 cau hoi nhung can smoke test nhanh:

```powershell
python scripts/run_baseline_eval.py --base-url http://127.0.0.1:8080 --kb-id 1 --limit 20 --allow-small
```

- Truoc benchmark embedding moi, chay baseline current config va luu `run_id`, metrics, va config snapshot. Khi doi model/backend, chay lai golden eval va so voi baseline run ID.

### Phase 1: Them embedding provider profile va model compatibility

Muc tieu: co the thu Qwen3/BGE/Jina ma khong sua code tung lan.

Viec can lam:

- Them settings:
  - `RAG_EMBEDDING_PROVIDER=sentence_transformers|tei|openai_compatible`
  - `RAG_EMBEDDING_MODEL=...`
  - `RAG_EMBEDDING_DIMENSION=auto|1024|2048|4096`
  - `RAG_EMBEDDING_TRUST_REMOTE_CODE=false`
  - `RAG_EMBEDDING_QUERY_INSTRUCTION=...`
  - `RAG_EMBEDDING_DOCUMENT_INSTRUCTION=...`
- Cap nhat `app/embeddings.py`:
  - ho tro instruction-aware query prompt cho Qwen3,
  - ho tro `trust_remote_code` chi khi flag bat,
  - log model dimension va model fingerprint,
  - expose model fingerprint cho cache/reindex.
- Cap nhat requirements:
  - pin/test `transformers>=4.51.0`,
  - giu `sentence-transformers` version tuong thich,
  - neu TEI thi them doc/docker profile rieng, khong bat buoc core.
- Them index compatibility guard:
  - neu dimension/model fingerprint doi, can bao rebuild index/collection moi.

Config thu nghiem de xuat:

```dotenv
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
RAG_EMBEDDING_QUERY_INSTRUCTION=Given a business support question in Vietnamese or English, retrieve passages from internal knowledge base documents that answer the question.
RAG_VECTOR_BACKEND=qdrant
RAG_QDRANT_COLLECTION_NAME=kb_chunks_qwen3_06b_v1
```

Acceptance criteria:

- Warm-up thanh cong voi model moi hoac fail ro rang ve dependency/hardware.
- Re-ingest va retrieve khong bi mismatch dimension.
- Golden eval chay duoc voi it nhat 2 embedding profiles.

Implementation baseline:

- Them provider profile:
  - `RAG_EMBEDDING_PROVIDER=sentence_transformers|tei|openai_compatible`
  - `RAG_EMBEDDING_BASE_URL=...` cho remote providers
  - `RAG_EMBEDDING_DIMENSION=0|1024|...`
  - `RAG_EMBEDDING_TRUST_REMOTE_CODE=false`
  - `RAG_EMBEDDING_QUERY_INSTRUCTION=...`
  - `RAG_EMBEDDING_DOCUMENT_INSTRUCTION=...`
- Embedding fingerprint gom provider, model/path, dimension, trust flag, prefixes, va instructions. Fingerprint duoc dung cho embedding cache, retrieval cache scope, eval snapshot, va ingest signature.
- Remote provider khong fallback ve hashing. Neu endpoint sai/chua san sang, startup/warm-up fail ro rang.
- Khi fingerprint doi, tao Qdrant collection moi hoac rebuild index truoc khi benchmark.

### Phase 2: Neural reranker abstraction

Muc tieu: thay BM25-lite bang reranking that su, nhung giu BM25-lite lam fallback.

Viec can lam:

- Refactor `app/reranker.py` thanh provider:
  - `bm25_lite`,
  - `sentence_transformers_cross_encoder`,
  - optional `tei_rerank` hoac local HTTP reranker.
- Them settings:
  - `RAG_RERANKER_PROVIDER=bm25_lite|cross_encoder|none`
  - `RAG_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B`
  - `RAG_RERANKER_TOP_N=50`
  - `RAG_RERANKER_BATCH_SIZE=8`
  - `RAG_RERANKER_TIMEOUT_SECONDS=10`
  - `RAG_RERANKER_MIN_SCORE=...`
  - `RAG_RERANKER_WEIGHT=...`
- Implement scoring:
  - retrieve top N candidates,
  - cross-encoder score query-doc pairs,
  - normalize score ve rank/similarity compatible voi `decide_mode`,
  - attach `reranker_score`, `retrieval_score`, `bm25_score`, `final_score`.
- Test:
  - fallback khi model loi,
  - deterministic order khi score bang nhau,
  - citation khong mat metadata,
  - timeout khong lam hong chat stream.

Model thu nghiem:

- First choice: `Qwen/Qwen3-Reranker-0.6B`.
- Fallback/baseline: `BAAI/bge-reranker-v2-m3`.

Acceptance criteria:

- Citation accuracy va MRR tang tren golden set.
- P95 retrieval+rerank latency nam trong budget.
- Khi reranker loi, app van tra loi bang BM25-lite.

Implementation baseline:

- Reranker mac dinh van la BM25-lite:

```dotenv
RAG_RERANKER_PROVIDER=bm25_lite
RAG_BM25_RERANKER_WEIGHT=0.15
```

- Bat neural reranker voi Qwen3:

```dotenv
RAG_RERANKER_PROVIDER=cross_encoder
RAG_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RAG_RERANKER_TOP_N=50
RAG_RERANKER_BATCH_SIZE=8
RAG_RERANKER_TIMEOUT_SECONDS=10
RAG_RERANKER_WEIGHT=0.85
RAG_RERANKER_MIN_SCORE=0.0
```

- Khi `cross_encoder` bat, retrieval lay candidate theo `max(top_k, RAG_RERANKER_TOP_N)`, rerank, roi cat ve `top_k`.
- Moi result giu nguyen metadata citation va them:
  - `retrieval_score`
  - `bm25_score` neu fallback/BM25
  - `reranker_score` neu neural
  - `reranker_provider`
  - `reranker_model`
  - `final_score`
- Neu load/predict/timeout loi, fallback ve BM25-lite.
- De dung Qwen3 reranker local, dependency can `transformers>=4.51.0,<5` va `sentence-transformers`.

### Phase 3: San xuat hoa Qdrant hybrid

Muc tieu: dung hybrid cho exact term, ma san pham, chinh sach, row/page query.

Viec can lam:

- Tao production profile trong docs/env:
  - `RAG_VECTOR_BACKEND=qdrant`
  - `RAG_QDRANT_HYBRID_ENABLED=true`
  - `RAG_QDRANT_HYBRID_PREFETCH_K=50`
  - collection rieng theo embedding model.
- Calibrate thresholds:
  - `RAG_QDRANT_HYBRID_MIN_SIMILARITY_THRESHOLD`
  - `RAG_QDRANT_HYBRID_THRESHOLD_LOW`
  - `RAG_QDRANT_HYBRID_THRESHOLD_GOOD`
- Them retrieval trace detail:
  - dense rank,
  - sparse rank,
  - RRF score,
  - reranker score.
- Them source diversification:
  - gioi han so chunk moi source truoc rerank hoac sau rerank,
  - uu tien chunk khac page/row khi query mo rong.

Acceptance criteria:

- Exact IDs va policy names dung hon dense-only.
- Khong tang hallucination citation.
- Co dashboard/log phan biet dense/sparse/rerank contribution.

Implementation baseline:

- Qdrant hybrid production profile nam trong `docs/vector-backends.md`.
- Khi bat hybrid, dung collection moi, vi schema dense-only va dense+sparse khac nhau.
- `retrieve()` ho tro source diversification opt-in:

```dotenv
RAG_RETRIEVAL_SOURCE_DIVERSIFICATION_ENABLED=true
RAG_RETRIEVAL_SOURCE_MAX_CHUNKS_PER_SOURCE=2
```

- Debug retrieval tra ve `retrieval_mode`, `qdrant_score`, `qdrant_query_mode`, `qdrant_fusion`, `qdrant_prefetch_k`, reranker score fields, va `source_diversified`.
- Eval snapshot luu hybrid settings va source diversification settings.
- Rollout dung thu tu: baseline current -> Qdrant dense-only -> Qdrant hybrid -> source diversification -> neural reranker.

### Phase 4: Corrective/agentic RAG nhe

Muc tieu: tang do ben khi retrieval lan dau kem ma khong can migrate LangGraph.

Viec can lam:

- Them document grader nho:
  - input: query + top retrieved snippets,
  - output structured: `relevant|partial|irrelevant`, reason, suggested_rewrite.
- Dung rule truoc LLM:
  - neu khong co result hoac top_score thap, rewrite voi LLM neu provider san sang,
  - retrieve lai toi da 1 lan.
- Them state vao stream start/done:
  - `retrieval_attempt_count`,
  - `query_rewritten`,
  - `correction_reason`.
- Record knowledge gap neu sau corrective loop van thap.
- Agent route eval:
  - dataset test expected route: RAG/tool/memory/clarify/fallback,
  - expected tool name/arguments cho single-step agent eval.

Acceptance criteria:

- Improve recall cho cau hoi follow-up/ambiguous.
- Khong lam tang latency dang ke cho cau hoi high-confidence.
- Khong goi tool/live-data neu chi can RAG.

Implementation baseline:

- Corrective RAG mac dinh tat:

```dotenv
RAG_CORRECTIVE_RAG_ENABLED=false
RAG_CORRECTIVE_RAG_MAX_ATTEMPTS=1
RAG_CORRECTIVE_RAG_MIN_SCORE=0.0
RAG_CORRECTIVE_RAG_MIN_RESULTS=1
RAG_CORRECTIVE_RAG_REWRITE_TIMEOUT_SECONDS=8
RAG_CORRECTIVE_RAG_REWRITE_MAX_TOKENS=128
```

- Khi bat, pipeline chi retry retrieval neu first attempt co `no_results`, `too_few_results`, hoac `low_top_score`.
- Rewrite query dung active LLM qua JSON `{ "query": "..." }`, khong tra loi nguoi dung o buoc rewrite.
- Retry toi da 1 lan; chi thay result neu retried top score tot hon hoac first attempt khong co result.
- SSE `start` va `done` tra `corrective_rag` metadata:
  - `attempt_count`
  - `query_rewritten`
  - `correction_reason`
  - `rewrite_error`
  - `rewritten_query`
  - `previous_top_score`
  - `corrected_top_score`
- Response cache scope co corrective settings de tranh tron cache giua pipeline cu va pipeline corrective.

### Phase 5: Visual document RAG track

Muc tieu: phuc vu PDF scan, bang bieu, hinh anh, tai lieu layout phuc tap.

Viec can lam:

- Them optional visual index:
  - convert PDF pages to images,
  - index page images bang ColPali/Byaldi hoac Jina v4,
  - luu visual hit metadata: file_id, page_num, image_path, score.
- Ket hop retrieval:
  - text retrieval hien tai,
  - visual page retrieval,
  - evidence assembly dua ca OCR text va page image preview/citation.
- Them eval visual:
  - cau hoi ve bang/hinh/layout,
  - expected page/source,
  - compare text-only vs visual+text.

Acceptance criteria:

- Co the tra citation dung page cho scanned/visual-heavy PDF.
- Visual track la optional dependency, core install van nhe.

### Phase 6: Deployment va cost/latency controls

Muc tieu: model moi khong lam app kho chay.

Viec can lam:

- Ho tro 3 mode:
  - local CPU: BGE-M3 hoac Qwen3-Embedding-0.6B neu latency chap nhan duoc,
  - local GPU: Qwen3 embedding/reranker,
  - service mode: Hugging Face TEI/vLLM hoac embedding/reranker HTTP endpoint.
- Them budget settings:
  - max rerank candidates,
  - max answer chunks,
  - timeout per provider,
  - disable reranker per request/admin toggle khi overload.
- Them monitoring:
  - embedding latency,
  - vector query latency,
  - reranker latency,
  - LLM latency,
  - cache hit rate,
  - golden eval regression.

Acceptance criteria:

- Co config local dev va production rieng.
- Khi reranker/embedding service down, RAG fallback van hoat dong.

Trang thai trien khai Phase 6:

- Da them `RAG_DEPLOYMENT_PROFILE=custom|local_cpu|local_gpu|service`.
- Da them runtime budget:
  - `RAG_RUNTIME_MAX_RERANK_CANDIDATES`,
  - `RAG_RUNTIME_MAX_ANSWER_CHUNKS`,
  - `RAG_RUNTIME_RETRIEVAL_LATENCY_BUDGET_MS`,
  - `RAG_RUNTIME_LLM_LATENCY_BUDGET_MS`.
- Da them overload toggles:
  - `RAG_RUNTIME_DISABLE_RERANKER`,
  - `RAG_RUNTIME_DISABLE_NEURAL_RERANKER`,
  - `RAG_RUNTIME_DISABLE_CORRECTIVE_RAG`,
  - per-request `/api/chat` flags `disable_reranker`, `disable_corrective_rag`.
- SSE `start` va `done` tra:
  - `runtime_budget`,
  - `latency_breakdown.embedding_ms`,
  - `latency_breakdown.vector_query_ms`,
  - `latency_breakdown.reranker_ms`,
  - `latency_breakdown.llm_ms`,
  - `latency_breakdown.cache_hit`.
- `/api/system` va golden eval snapshot luu runtime budget de so sanh benchmark/release.
- Chi tiet run-mode nam trong `docs/run-modes.md`.

## Thu tu uu tien de lam ngay

1. Benchmark baseline va bo golden eval lon hon.
2. Them reranker abstraction voi `Qwen/Qwen3-Reranker-0.6B`.
3. Thu `Qwen/Qwen3-Embedding-0.6B` va `BAAI/bge-m3` tren Qdrant collection rieng.
4. Calibrate Qdrant hybrid thresholds bang eval gate.
5. Them corrective loop nhe: grade/rewrite/retrieve lai 1 lan.
6. Chi sau do moi mo visual RAG track voi Jina v4/ColPali.

## Quyet dinh khuyen nghi

- Khong thay embedding mac dinh cua MVP ngay. Giu local-friendly default, them production profile.
- Khong migrate sang LangGraph ngay. Du an da co agent runtime, pending actions, va eval; chi nen hoc pattern grade/rewrite/trajectory eval.
- Khong dua Jina v4/ColPali vao core dependency. De optional vi nang va can visual-specific eval.
- Nen chon Qwen3-Reranker-0.6B lam nang cap dau tien vi current reranker la diem yeu ro nhat.
- Nen chon Qwen3-Embedding-0.6B vs BGE-M3 bang eval thuc te tren data cua du an, khong dua vao leaderboard don thuan.

## Definition of done cho sprint nang cap dau tien

- Co file config/env mau cho `qdrant + qwen3 embedding + qwen3 reranker`.
- Co reranker provider moi va fallback BM25-lite.
- Co it nhat 80 golden questions va eval before/after.
- Co bao cao:
  - recall_at_k,
  - MRR,
  - citation_accuracy,
  - answer score,
  - p50/p95 latency,
  - failure examples.
- Khong co regression trong tests hien tai cho RAG, KB access control, auth scope, va cache scope.
