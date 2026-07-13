import json
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from utils.config_loader import Configs


def _load_configs() -> Configs:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "resources" / "config.yaml"
    return Configs(str(config_path))


def fetch_job_by_id(job_id: int) -> dict[str, Any] | None:
    """
    Fetch a row from `job` by id using the DataPG connection from config.
    """
    cfg = _load_configs().DataPG

    conn = psycopg2.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.database,
        user=cfg.username,
        password=cfg.password,
        cursor_factory=RealDictCursor,
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM job WHERE id = %s", (job_id,))
                row = cur.fetchone()
                return dict(row) if row is not None else None
    finally:
        conn.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="agent job-get")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    row = fetch_job_by_id(args.id)
    if args.pretty:
        print(json.dumps(row, indent=2, default=str))
    else:
        print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()

