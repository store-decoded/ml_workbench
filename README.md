## This Project is Under Developement! 

feel free To use the naked code inside available files but its just a messy stockpile for now!

### MCP Postgres (Docker)

This repo includes a minimal setup to run the upstream MCP server image `mochoa/mcp-postgres`
against the `DataPG` database configured in `resources/config.yaml`.

- **Print runtime env vars from config**:

```bash
uv run agent mcp-postgres-env
```

- **Run the MCP server via docker compose** (in your own shell):

```bash
set -a
eval "$(uv run agent mcp-postgres-env | sed 's/^/export /')"
set +a

docker compose -f docker-compose.mcp-postgres.yml up --build
```

Once the MCP server is running, your MCP client can call tools like `pg-query` and run:

`select * from job where id = <given id>`


'''docker run -i --rm -e PG_USER=enricher -e PG_PASSWORD=5up3r53CUR3D mochoa/mcp-postgres:latest postgresql://192.168.88.208:5432/data_validation
'''