---
description: Full RAG pipeline v2 — KB data format, 3-mode answer logic, dual thresholds
---

# RAG Pipeline Workflow v2

## A. Chuẩn hóa dữ liệu đầu vào (KB format)

Thay vì dạng Q/A đơn giản, dữ liệu nên lưu theo dạng **knowledge base by topic**.

### Cột gợi ý cho Excel/CSV

| Cột | Bắt buộc | Mô tả |
|-----|----------|--------|
| `title` | ✅ | Tiêu đề ngắn, gọn |
| `category` | ✅ | Nhóm: Shipping / Return / Payment / Support / Products |
| `content` | ✅ | Nội dung chính sách / điều khoản / hướng dẫn |
| `keywords` | ☐ | Từ khóa bổ trợ embedding (optional) |
| `source_url` | ☐ | Liên kết nguồn gốc (optional) |

### Ví dụ

```csv
title,category,content,keywords
"Miễn phí vận chuyển","Shipping","Miễn phí ship cho đơn từ 500.000đ. Nội thành 20.000đ, ngoại thành 30.000đ.","ship,phí,giao hàng"
"Chính sách đổi trả","Return","Đổi trả trong 7 ngày kể từ khi nhận hàng, còn nguyên tem mác.","đổi,trả,hoàn"
```

> Khách hỏi bằng bất kỳ cách nào đều match được nhờ embedding semantic search.

---

## B. Ingest Pipeline (Parse → Chunk → Embed → Store)

```
File upload
  │
  ▼
Parse (CSV/Excel/PDF/HTML)
  │  Mỗi dòng Excel/CSV = 1 "knowledge record"
  │  Gom text: title + category + content + keywords
  ▼
Chunk (nếu content dài > chunk_size)
  │
  ▼
Embed (SentenceTransformers — local hoặc offline)
  │
  ▼
Vector Store (numpy, thread-safe, atomic write)
  │  Lưu kèm metadata: filename, category, row, title, chunk_id
  ▼
Done → cập nhật status DB
```

---

## C. Chat Pipeline (Retrieve → Decide → Answer)

### Step 1: Retrieve

```python
query_emb = embed(query)
results   = vector_store.query(query_emb, top_k=10)  # top_k=8-12
top_score = results[0].similarity  # 0.0–1.0
```

### Step 2: Decide mode

```
THRESHOLD_GOOD = 0.60   # chắc chắn có dữ liệu
THRESHOLD_LOW  = 0.40   # hơi liên quan

score >= THRESHOLD_GOOD  → MODE: answer
THRESHOLD_LOW <= score < THRESHOLD_GOOD  → MODE: clarify
score < THRESHOLD_LOW    → MODE: fallback
```

### Step 3: Render theo mode

Xem mục D.

---

## D. 3 Chế độ trả lời

### 1. Answer mode (`score >= 0.60`)

- **Extractive** (không có LLM): trích top 3 chunk liên quan nhất, hiển thị có cấu trúc + citations.
- **Generative** (có LLM): prompt LLM với context để synthesize câu trả lời tự nhiên.
- Luôn kèm citations.

**Ví dụ** — Khách hỏi: *"Đơn bao nhiêu thì free ship?"*
> "Miễn phí vận chuyển cho đơn từ 500.000đ trở lên." [Nguồn: kb.csv dòng 1]

---

### 2. Clarify mode (`0.40 <= score < 0.60`)

Khi dữ liệu hơi liên quan nhưng chưa đủ chắc để tự trả lời.

Agent hỏi lại **1 câu** để cụ thể hoá, sau đó chạy lại retrieve với câu trả lời của khách.

**Ví dụ** — Khách hỏi: *"Ship bao nhiêu tiền?"*
> "Bạn ở khu vực nội thành hay ngoại thành (hoặc tỉnh nào) để mình báo phí giao hàng chính xác nhé?"

Flow sau clarify:
```
User trả lời → query mới = "ship [khu vực user trả lời]" → retrieve lại → answer mode
```

---

### 3. Fallback mode (`score < 0.40`)

KHÔNG bịa. Trả lời theo template cố định.

**Ví dụ** — Khách hỏi: *"Shop có giao hỏa tốc 2 giờ không?"* (KB không có)
> "Hiện trong tài liệu mình chưa thấy thông tin về giao hỏa tốc 2 giờ."
> "Bạn muốn mình ghi nhận yêu cầu và nhờ nhân viên CSKH liên hệ lại không?"

Frontend hiện nút **"Tạo yêu cầu hỗ trợ"** → ticket/form liên hệ.

---

## E. SSE Events

```
event: start      → { "query": "...", "mode": "answer|clarify|fallback", "score": 0.72 }
event: token      → { "text": "..." }        ← stream từng token (hoặc 1 lần cho extractive)
event: citations  → { "items": [...] }       ← chỉ có trong answer mode
event: done       → { "ok": true }
event: error      → { "message": "..." }     ← chỉ khi có exception
```

---

## F. Thay đổi code cần thiết

### `app/rag.py`

- Thêm `THRESHOLD_GOOD = 0.60`, `THRESHOLD_LOW = 0.40`
- Hàm `decide_mode(top_score) → "answer" | "clarify" | "fallback"`
- Hàm `_clarify_question(results) → str` — sinh câu hỏi làm rõ dựa trên top chunk
- Hàm `_fallback_response() → str` — template cố định
- Cập nhật `rag_stream()` để emit `event: start` với `mode` + `score`

### `app/config.py`

- Thêm `threshold_good: float = 0.60`
- Thêm `threshold_low: float = 0.40`

### `static/chat.html`

- Parse `event: start` để đọc `mode`
- Nếu `mode == "fallback"` → hiện nút **"Tạo yêu cầu hỗ trợ"**
- Nếu `mode == "clarify"` → hiện badge "Cần làm rõ" màu vàng

### `app/parsers.py` (optional)

- Nhận dạng `title`/`category`/`content`/`keywords` columns khi parse CSV/Excel
- Gom text theo format: `{title}: {content}` để embedding phong phú hơn

---

## G. Tham số cần điều chỉnh theo dữ liệu thực

| Tham số | Mặc định | Điều chỉnh khi |
|---------|----------|----------------|
| `THRESHOLD_GOOD` | 0.60 | Embedding model khác → calibrate lại |
| `THRESHOLD_LOW` | 0.40 | Quá nhiều clarify → hạ xuống 0.35 |
| `top_k` | 10 | Dataset nhỏ → 5–8 |
| `MAX_EXTRACTIVE_CHUNKS` | 3 | Nội dung dài → 2 |
| `chunk_size` | 1000 | Content field ngắn → 300–500 |
