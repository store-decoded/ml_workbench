import argparse

from mcp.postgres_mcp_config import build_postgres_mcp_config
from mcp.job_queries import fetch_job_by_id
from utils.config_loader import Configs


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mcp-postgres-config", help="Print DataPG MCP docker config")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    j = sub.add_parser("job-get", help="Fetch a job row by id from DataPG")
    j.add_argument("--id", type=int, required=True, help="job.id")
    j.add_argument("--pretty", action="store_true", help="Pretty-print JSON")

    e = sub.add_parser("mcp-postgres-env", help="Print env vars for docker/compose (DataPG)")

    args = parser.parse_args()

    if args.command == "mcp-postgres-config":
        config = build_postgres_mcp_config()
        if args.pretty:
            import json

            print(json.dumps(config, indent=2))
        else:
            import json

            print(json.dumps(config))
        return

    if args.command == "job-get":
        import json

        row = fetch_job_by_id(args.id)
        if args.pretty:
            print(json.dumps(row, indent=2, default=str))
        else:
            print(json.dumps(row, default=str))
        return

    if args.command == "mcp-postgres-env":
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "resources" / "config.yaml"
        cfg = Configs(str(config_path)).DataPG

        # MCP server expects URL without embedded credentials.
        url = f"postgresql://{cfg.host}:{cfg.port}/{cfg.database}"

        print(f"DATA_PG_URL={url}")
        print(f"DATA_PG_USERNAME={cfg.username}")
        print(f"DATA_PG_PASSWORD={cfg.password}")
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()

