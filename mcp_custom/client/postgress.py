import asyncio, json, os
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pprint import pprint

client = OpenAI(base_url='https://api.avalai.ir/v1', api_key='aa-wou0Xw7k9VkZUyuLWd8ZCydGcIPk3mV9Pto1Z7cG4pHtYN0t')

def llm_chat(messages, tools=None):
    kwargs = {"model": "gpt-5.5-mini", "messages": messages}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)

def msg_to_dict(msg):
    """Serialize an assistant message (with possible tool_calls) back into API dict form."""
    d = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d

async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", "/home/v0id/code/playground/ml_workbench", "python3.12", "mcp_custom/server/postgress.py"],
        env={
            **os.environ,
            "PG_URL": "postgresql://192.168.88.208:5432/data_validation",
            "PG_USER": "enricher",
            "PG_PASSWORD": "5up3r53CUR3D",
            "PG_ALLOW_WRITE": "1",
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in mcp_tools.tools
            ]

            messages = [{
                "role": "user",
                "content": "tell me the schema of table job",
            }]

            MAX_TURNS = 15  # safety guard against infinite tool-call loops

            for turn in range(MAX_TURNS):
                response = llm_chat(messages, tools=tools)
                msg = response.choices[0].message

                if not msg.tool_calls:
                    print("Final answer:", msg.content)
                    break

                # record the assistant's tool-call turn
                messages.append(msg_to_dict(msg))

                # execute every tool call the model asked for, not just the first
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    print(f"Calling tool: {tc.function.name}({args})")
                    try:
                        result = await session.call_tool(tc.function.name, args)
                        tool_result = result.content[0].text if result.content else ""
                    except Exception as e:
                        tool_result = f"ERROR calling {tc.function.name}: {e}"

                    print("Tool result:", tool_result)  # truncate for readability

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                    pprint(messages)
            else:
                print("Stopped after reaching MAX_TURNS without a final answer.")

asyncio.run(main())
