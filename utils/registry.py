# utils/registry.py
import sqlite3
import time
import sys
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DB_PATH = config.REGISTRY_DB


def init_registry():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL DEFAULT 'camelyon17',
            model TEXT NOT NULL,
            tta_method TEXT NOT NULL,
            center INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            started_at REAL,
            completed_at REAL,
            result_path TEXT,
            error_msg TEXT,
            created_at REAL DEFAULT (strftime('%s','now'))
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON runs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON runs(dataset, model, tta_method, center, seed)")
        conn.commit()


def make_run_id(model, tta_method, center, seed, stage, dataset="camelyon17"):
    return f"{dataset}__{model}__{tta_method}__c{center}__s{seed}__{stage}"


def is_done(model, tta_method, center, seed, stage, dataset="camelyon17"):
    rid = make_run_id(model, tta_method, center, seed, stage, dataset)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT status FROM runs WHERE run_id=?", (rid,)).fetchone()
    if row is not None and row[0] == 'done':
        return True
    # Double-check disk: if result JSON exists and is non-empty, treat as done & update registry.
    # The filename MUST carry the ablation tag. An ablation shares
    # (model, method, center, seed) with the main sweep and differs only by the
    # stage suffix, which run_evaluation.py also appends to the result filename.
    # Without it this lookup finds the MAIN sweep's json and reports every
    # ablation as already done -- which silently skipped all 18 of them.
    tag = stage[len("tta_evaluation"):] if stage.startswith("tta_evaluation") else ""
    raw_path = (config.RESULTS_ROOT / "raw" /
                f"{dataset}__{model}__{tta_method}__d{center}__s{seed}{tag}.json")
    if raw_path.exists() and raw_path.stat().st_size > 0:
        try:
            mark_done(model, tta_method, center, seed, stage, result_path=str(raw_path), dataset=dataset)
        except Exception:
            pass
        return True
    return False


def mark_running(model, tta_method, center, seed, stage, dataset="camelyon17"):
    rid = make_run_id(model, tta_method, center, seed, stage, dataset)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        INSERT OR REPLACE INTO runs
        (run_id, dataset, model, tta_method, center, seed, stage, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)
        """, (rid, dataset, model, tta_method, center, seed, stage, time.time()))
        conn.commit()


def mark_done(model, tta_method, center, seed, stage, result_path=None, dataset="camelyon17"):
    rid = make_run_id(model, tta_method, center, seed, stage, dataset)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE runs SET status='done', completed_at=?, result_path=? WHERE run_id=?",
                     (time.time(), result_path, rid))
        conn.commit()


def mark_failed(model, tta_method, center, seed, stage, error_msg, dataset="camelyon17"):
    rid = make_run_id(model, tta_method, center, seed, stage, dataset)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE runs SET status='failed', completed_at=?, error_msg=? WHERE run_id=?",
                     (time.time(), str(error_msg)[:2000], rid))
        conn.commit()


def print_status():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status").fetchall()
        by_ds = conn.execute(
            "SELECT dataset, status, COUNT(*) FROM runs GROUP BY dataset, status").fetchall()
    total = sum(r[1] for r in rows)
    print(f"\n=== Run Registry Status (total: {total}) ===")
    for status, count in rows:
        print(f"  {status:12s}: {count}")
    print("  --- by dataset ---")
    for ds, status, count in by_ds:
        print(f"  {ds:12s} {status:10s}: {count}")
    print()
