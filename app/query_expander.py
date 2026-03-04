"""
Query expansion with VI/EN synonym dict, slang mapping, and no-diacritics normalization.
Returns at most 2 query variants (original + 1 expansion).
"""

from __future__ import annotations

import re
import unicodedata

# ── No-diacritics normalization ───────────────────────────────────────────────

def _remove_diacritics(text: str) -> str:
    """Convert 'học phí' → 'hoc phi', 'đăng ký' → 'dang ky'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D")


# ── Synonym & slang dictionary ────────────────────────────────────────────────
# Format: "no-diacritics lowercase" → canonical Vietnamese (+ optional English)

_SLANG_MAP: dict[str, str] = {
    # Shipping / delivery
    "ship": "giao hàng",
    "shipping": "giao hàng",
    "giao hang": "giao hàng",
    "phi giao hang": "phí giao hàng",
    "delivery": "giao hàng",
    # Fees / prices
    "fee": "phí",
    "fees": "phí",
    "price": "giá",
    "cost": "giá",
    "gia": "giá",
    "hoc phi": "học phí",
    "tuition": "học phí",
    # Scholarship
    "hoc bong": "học bổng",
    "scholarship": "học bổng",
    # Admission
    "tuyen sinh": "tuyển sinh",
    "admission": "tuyển sinh",
    "nhap hoc": "nhập học",
    "enroll": "đăng ký",
    "dang ky": "đăng ký",
    "register": "đăng ký",
    # Deadline
    "deadline": "hạn nộp",
    "han nop": "hạn nộp",
    # Return / exchange
    "doi tra": "đổi trả",
    "return": "đổi trả",
    "refund": "hoàn tiền",
    "hoan tien": "hoàn tiền",
    # Payment
    "thanh toan": "thanh toán",
    "payment": "thanh toán",
    "pay": "thanh toán",
    # Promotion
    "khuyen mai": "khuyến mãi",
    "promo": "khuyến mãi",
    "discount": "giảm giá",
    "giam gia": "giảm giá",
    # Contact / support
    "hotline": "số điện thoại",
    "so dien thoai": "số điện thoại",
    "lien he": "liên hệ",
    "contact": "liên hệ",
    # Exam / re-exam
    "thi lai": "thi lại",
    "retake": "thi lại",
    # Internship
    "thuc tap": "thực tập",
    "internship": "thực tập",
}

_WORD_BOUNDARY = re.compile(r"\b\w+\b", re.UNICODE)


def _normalize_query(text: str) -> str:
    """Lowercase + strip."""
    return " ".join(text.strip().lower().split())


def expand_query(query: str) -> list[str]:
    """
    Returns [original_query] or [original_query, expanded_variant].
    Expansion is triggered when:
    - Query is short (<= 5 words) OR
    - A slang/synonym key is matched
    Never returns more than 2 variants.
    """
    normalized = _normalize_query(query)
    no_diac = _normalize_query(_remove_diacritics(normalized))

    # Check direct match first (whole query)
    if no_diac in _SLANG_MAP:
        expanded = _SLANG_MAP[no_diac]
        if expanded.lower() != normalized:
            return [query, expanded]

    # Check token-level matches
    tokens = no_diac.split()
    expanded_tokens = list(tokens)
    matched = False

    for i, token in enumerate(tokens):
        if token in _SLANG_MAP:
            replacement = _SLANG_MAP[token]
            expanded_tokens[i] = replacement
            matched = True

    # Also try bigrams
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in _SLANG_MAP:
            replacement = _SLANG_MAP[bigram]
            expanded_tokens[i] = replacement
            expanded_tokens[i + 1] = ""
            matched = True

    expanded_str = " ".join(t for t in expanded_tokens if t).strip()

    if matched and expanded_str and expanded_str.lower() != normalized:
        return [query, expanded_str]

    # Short query heuristic: also search with no-diacritics form
    if len(tokens) <= 3 and no_diac != normalized:
        return [query, no_diac]

    return [query]
