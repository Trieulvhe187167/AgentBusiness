"""
Mock business API for local end-to-end demos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "mock_business_api.json"


class OrderRecord(BaseModel):
    order_code: str = Field(..., min_length=3, max_length=80)
    user_id: str = Field(..., min_length=1, max_length=120)
    status: str = Field(..., min_length=1, max_length=80)
    last_update: str | None = None
    tracking_code: str | None = None
    carrier: str | None = None


class AllianceRecord(BaseModel):
    alliance_id: str = Field(..., min_length=1, max_length=80)
    server_id: str | None = Field(default=None, max_length=80)
    online_count: int = Field(..., ge=0)
    observed_at: str = Field(..., min_length=1, max_length=80)


class MockBusinessStore:
    def __init__(self, path: Path):
        self.path = path
        self.orders: list[OrderRecord] = []
        self.alliances: list[AllianceRecord] = []
        self.reload()

    def reload(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_default_file()

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.orders = [OrderRecord.model_validate(item) for item in payload.get("orders") or []]
        self.alliances = [AllianceRecord.model_validate(item) for item in payload.get("alliances") or []]

    def save(self) -> None:
        payload = {
            "orders": [item.model_dump() for item in self.orders],
            "alliances": [item.model_dump() for item in self.alliances],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_default_file(self) -> None:
        default_payload = {
            "orders": [
                {
                    "order_code": "DH12345",
                    "user_id": "user-1",
                    "status": "dang_giao",
                    "last_update": "2026-03-15T14:20:00+07:00",
                    "tracking_code": "GHN-9988",
                    "carrier": "GHN",
                },
                {
                    "order_code": "DH12346",
                    "user_id": "user-1",
                    "status": "cho_xac_nhan",
                    "last_update": "2026-03-15T10:05:00+07:00",
                    "tracking_code": None,
                    "carrier": None,
                },
                {
                    "order_code": "DH54321",
                    "user_id": "user-2",
                    "status": "da_giao",
                    "last_update": "2026-03-14T16:30:00+07:00",
                    "tracking_code": "VN-1001",
                    "carrier": "VNPOST",
                },
            ],
            "alliances": [
                {
                    "alliance_id": "LM01",
                    "server_id": "S1",
                    "online_count": 128,
                    "observed_at": "2026-03-15T14:21:00+07:00",
                },
                {
                    "alliance_id": "LM02",
                    "server_id": "S2",
                    "online_count": 57,
                    "observed_at": "2026-03-15T14:22:00+07:00",
                },
            ],
        }
        self.path.write_text(json.dumps(default_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_order(self, order_code: str, user_id: str | None = None) -> OrderRecord:
        normalized = order_code.strip().upper()
        for order in self.orders:
            if order.order_code.upper() != normalized:
                continue
            if user_id and order.user_id != user_id:
                raise HTTPException(status_code=403, detail="Order does not belong to this user")
            return order
        raise HTTPException(status_code=404, detail="Order not found")

    def recent_orders(self, user_id: str, limit: int) -> list[OrderRecord]:
        rows = [item for item in self.orders if item.user_id == user_id]
        rows.sort(key=lambda item: item.last_update or "", reverse=True)
        return rows[:limit]

    def get_alliance(self, alliance_id: str, server_id: str | None = None) -> AllianceRecord:
        normalized = alliance_id.strip().upper()
        preferred_server = (server_id or "").strip().upper() or None
        exact_match = None
        first_match = None
        for item in self.alliances:
            if item.alliance_id.upper() != normalized:
                continue
            if first_match is None:
                first_match = item
            if preferred_server and (item.server_id or "").upper() == preferred_server:
                exact_match = item
                break
        if exact_match is not None:
            return exact_match
        if first_match is not None and preferred_server is None:
            return first_match
        raise HTTPException(status_code=404, detail="Alliance not found")

    def upsert_order(self, payload: OrderRecord) -> OrderRecord:
        normalized = payload.order_code.strip().upper()
        self.orders = [item for item in self.orders if item.order_code.upper() != normalized]
        saved = payload.model_copy(update={"order_code": normalized})
        self.orders.append(saved)
        self.save()
        return saved

    def upsert_alliance(self, payload: AllianceRecord) -> AllianceRecord:
        normalized_id = payload.alliance_id.strip().upper()
        normalized_server = (payload.server_id or "").strip().upper() or None
        self.alliances = [
            item
            for item in self.alliances
            if not (
                item.alliance_id.upper() == normalized_id and ((item.server_id or "").strip().upper() or None) == normalized_server
            )
        ]
        saved = payload.model_copy(update={"alliance_id": normalized_id, "server_id": normalized_server})
        self.alliances.append(saved)
        self.save()
        return saved

    def dump(self) -> dict[str, Any]:
        return {
            "orders": [item.model_dump() for item in self.orders],
            "alliances": [item.model_dump() for item in self.alliances],
            "data_path": str(self.path),
        }


store = MockBusinessStore(DATA_PATH)
app = FastAPI(title="Mock Business API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "orders": len(store.orders), "alliances": len(store.alliances)}


@app.get("/orders/status")
def get_order_status(
    order_code: str = Query(..., min_length=3),
    user_id: str | None = Query(default=None),
) -> dict[str, Any]:
    return store.get_order(order_code, user_id=user_id).model_dump()


@app.get("/orders/recent")
def get_recent_orders(
    user_id: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    return {
        "orders": [item.model_dump() for item in store.recent_orders(user_id, limit)],
    }


@app.get("/alliances/online")
def get_alliance_online(
    alliance_id: str = Query(..., min_length=1),
    server_id: str | None = Query(default=None),
) -> dict[str, Any]:
    return store.get_alliance(alliance_id, server_id=server_id).model_dump()


@app.get("/admin/dump")
def admin_dump() -> dict[str, Any]:
    return store.dump()


@app.post("/admin/orders/upsert")
def admin_upsert_order(payload: OrderRecord) -> dict[str, Any]:
    return store.upsert_order(payload).model_dump()


@app.post("/admin/alliances/upsert")
def admin_upsert_alliance(payload: AllianceRecord) -> dict[str, Any]:
    return store.upsert_alliance(payload).model_dump()


@app.post("/admin/reload")
def admin_reload() -> dict[str, Any]:
    store.reload()
    return {"ok": True, "orders": len(store.orders), "alliances": len(store.alliances)}
