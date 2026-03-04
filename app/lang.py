"""
Language detection utility.
Shared between parsers.py and rag.py to avoid circular imports.
"""

from __future__ import annotations

import re


_VI_DIACRITICS = re.compile(
    r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)

_VI_HINTS = [
    "bao nhiêu", "giao hàng", "đổi trả", "thanh toán", "sản phẩm",
    "khuyến mãi", "ở đâu", "học phí", "tuyển sinh", "học bổng",
    "đăng ký", "khai giảng", "nhập học",
]


def detect_language(text: str, explicit_lang: str | None = None) -> str:
    """
    Return 'vi' or 'en'.
    explicit_lang overrides auto-detection when set to a valid value.
    """
    if explicit_lang:
        explicit = explicit_lang.strip().lower()
        if explicit in {"vi", "en"}:
            return explicit

    lowered = text.lower()
    if _VI_DIACRITICS.search(lowered):
        return "vi"

    if any(hint in lowered for hint in _VI_HINTS):
        return "vi"

    return "en"
