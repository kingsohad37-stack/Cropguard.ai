"""Scan persistence with Supabase REST storage and a local SQLite fallback.

Persistence is deliberately non-blocking for inference: a Supabase auth/RLS/network
failure must never turn a valid model prediction into a failed /predict request.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import requests


class ScanStore:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.remote = bool(self.url and self.key)
        self.initialize()

    def initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as con:
            con.execute("""CREATE TABLE IF NOT EXISTS scans(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
                filename TEXT, label TEXT NOT NULL, confidence REAL NOT NULL,
                severity REAL NOT NULL, coverage REAL NOT NULL,
                image_hash TEXT NOT NULL DEFAULT ''
            )""")
            con.commit()

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

    def _insert_local(self, record: dict[str, Any]) -> int:
        with sqlite3.connect(self.sqlite_path) as con:
            cur = con.execute(
                "INSERT INTO scans(ts,filename,label,confidence,severity,coverage,image_hash) VALUES(?,?,?,?,?,?,?)",
                (
                    record["ts"],
                    record["filename"],
                    record["label"],
                    record["confidence"],
                    record["severity_score"],
                    record["heatmap_coverage_percent"],
                    record["image_sha256"],
                ),
            )
            con.commit()
            return int(cur.lastrowid)

    def insert(self, record: dict[str, Any]) -> int | str | None:
        # Severity/coverage are derived from genuine Grad-CAM. Never invent
        # fallback values merely to satisfy the persistence schema. A valid
        # classification must still be returned when Grad-CAM is unavailable.
        severity = record.get("severity_score")
        coverage = record.get("heatmap_coverage_percent")
        if severity is None or coverage is None:
            return None

        payload = {
            "filename": record["filename"],
            "label": record["label"],
            "confidence": record["confidence"],
            "severity_score": severity,
            "heatmap_coverage_percent": coverage,
            "image_sha256": record["image_sha256"],
        }
        if self.remote:
            try:
                response = requests.post(
                    f"{self.url}/rest/v1/scans",
                    headers={**self._headers(), "Prefer": "return=representation"},
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                rows = response.json()
                if rows and "id" in rows[0]:
                    return rows[0]["id"]
            except requests.RequestException as exc:
                print(f"[CropGuard] Supabase scan storage unavailable; keeping inference result: {exc}")

        return self._insert_local(record)

    def history(self) -> list[dict[str, Any]]:
        if self.remote:
            try:
                response = requests.get(
                    f"{self.url}/rest/v1/scans",
                    headers=self._headers(),
                    params={
                        "select": "id,created_at,filename,label,confidence,severity_score,heatmap_coverage_percent,image_sha256",
                        "order": "created_at.desc",
                    },
                    timeout=10,
                )
                response.raise_for_status()
                return [
                    {
                        "id": r["id"],
                        "timestamp": r["created_at"],
                        "filename": r["filename"],
                        "label": r["label"],
                        "confidence": r["confidence"],
                        "severity_score": r["severity_score"],
                        "heatmap_coverage_percent": r["heatmap_coverage_percent"],
                        "image_sha256": r["image_sha256"],
                    }
                    for r in response.json()
                ]
            except requests.RequestException as exc:
                print(f"[CropGuard] Supabase history unavailable; using local fallback: {exc}")

        with sqlite3.connect(self.sqlite_path) as con:
            rows = con.execute(
                "SELECT id,ts,filename,label,confidence,severity,coverage,image_hash FROM scans ORDER BY id DESC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "filename": r[2],
                "label": r[3],
                "confidence": r[4],
                "severity_score": r[5],
                "heatmap_coverage_percent": r[6],
                "image_sha256": r[7],
            }
            for r in rows
        ]
