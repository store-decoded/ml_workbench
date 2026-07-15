import os
import re
import sys
import json
import decimal
import datetime
import psycopg2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("postgres-custom")

ALLOW_WRITE = os.getenv("PG_ALLOW_WRITE", "0") == "1"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _dsn() -> str:
    url = os.getenv("PG_URL", "")
    user = os.getenv("PG_USER", "")
    password = os.getenv("PG_PASSWORD", "")
    if user and password and "://" in url:
        proto, rest = url.split("://", 1)
        url = f"{proto}://{user}:{password}@{rest}"
    return url


def _serialize(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")


def _is_read_only(sql: str) -> bool:
    first = sql.strip().split()[0].upper() if sql.strip() else ""
    return first in {"SELECT", "SHOW", "EXPLAIN", "DESCRIBE", "WITH"}


def _quote_ident(ident: str) -> str:
    if not _IDENT_RE.match(ident or ""):
        raise ValueError(f"Invalid identifier: {ident!r}")
    return f'"{ident}"'


@mcp.tool()
def pg_query(sql: str) -> str:
    """Execute a SQL query and return results as JSON."""
    if not ALLOW_WRITE and not _is_read_only(sql):
        return json.dumps({"error": "Write queries require PG_ALLOW_WRITE=1."})
    try:
        with psycopg2.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description:
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
                    return json.dumps(rows, default=_serialize)
                conn.commit()
                return json.dumps({"rowcount": cur.rowcount})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def pg_list_tables(schema: str = "public") -> str:
    """List all tables in the given schema."""
    sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """
    try:
        with psycopg2.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (schema,))
                return json.dumps([r[0] for r in cur.fetchall()])
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def pg_create_table(
    table: str,
    columns: list[dict],
    schema: str = "public",
    if_not_exists: bool = True,
) -> str:
    """
    Create a new table. Requires PG_ALLOW_WRITE=1.

    Args:
        table: Table name.
        columns: List of dicts with 'name', 'type', and optional 'constraints'.
                 Example: [{"name": "id", "type": "SERIAL", "constraints": "PRIMARY KEY"}]
        schema: Target schema (default: "public").
        if_not_exists: Add IF NOT EXISTS clause (default True).
    """
    if not ALLOW_WRITE:
        return json.dumps({"error": "Table creation requires PG_ALLOW_WRITE=1."})
    if not columns:
        return json.dumps({"error": "columns must be a non-empty list."})
    try:
        ident_table = _quote_ident(table)
        ident_schema = _quote_ident(schema)
        col_defs = []
        for col in columns:
            name, col_type = col.get("name"), col.get("type")
            if not name or not col_type:
                return json.dumps({"error": f"Column missing 'name' or 'type': {col!r}"})
            constraints = col.get("constraints", "")
            col_defs.append(f"{_quote_ident(name)} {col_type} {constraints}".strip())
        ine = "IF NOT EXISTS " if if_not_exists else ""
        sql = (
            f"CREATE TABLE {ine}{ident_schema}.{ident_table} (\n  "
            + ",\n  ".join(col_defs)
            + "\n);"
        )
        with psycopg2.connect(_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
                return json.dumps({"status": "ok", "sql": sql})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
    # mcp.run(transport="sse", host="0.0.0.0", port=8080)
