FROM mochoa/mcp-postgres:latest

# This image runs the MCP Postgres server. Per upstream docs, you pass the
# PostgreSQL URL (without credentials) as the container argument and provide
# credentials via PG_USER / PG_PASSWORD environment variables.
#
# Example:
#   docker run -i --rm \
#     -e PG_USER=... -e PG_PASSWORD=... \
#     ml-workbench-mcp-postgres postgresql://enricher:5up3r53CUR3D@192.168.88.208:5432/data_validation
#
# The default command is a placeholder; override it at runtime.
CMD ["postgresql://192.168.88.208:5432/data_validation"]

