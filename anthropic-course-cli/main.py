import asyncio
import sys

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage





llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0.3,
)


async def use_tools(ai_msg, tools_by_name, messages):

    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(tool_call)

        # print(type(tool_args), tool_args)

        if tool_name in tools_by_name:
            tool = tools_by_name[tool_name]
            try:
                result = await tool.ainvoke(tool_args)

                messages.append(ToolMessage( content=str(result),tool_call_id=tool_call_id,))

                print(result)
            except Exception as e:
                result = f"Error invoking tool '{tool_name}': {str(e)}"
                
                messages.append(ToolMessage( content=str(result),tool_call_id=tool_call_id,))
        else:
            print(f"Tool '{tool_name}' not found.")
            return tools_by_name
    
    return



async def main():

    client = MultiServerMCPClient(
        {
            "documents": {
                "command": "uv",
                "args": ["run", "mcp_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    
    tools_by_name = {tool.name: tool for tool in tools}
    
    for tool in tools_by_name.values():
        print(f"Tool: {tool.name}")

    llm_with_tools = llm.bind_tools(tools)

    async def chat(prompt, iterations=10, llm_with_tools=llm_with_tools, tools_by_name=tools_by_name):
        messages = [
            SystemMessage(content="You are a helpful assistant. Use tools when needed. You can call tools multiple times if needed."),
            HumanMessage(content=str(prompt))
        ]

        ai_msg = None
        
        for _ in range(iterations):
            ai_msg = await llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)
            # print(ai_msg)

            if not ai_msg.tool_calls:
                return ai_msg

            await use_tools(ai_msg, tools_by_name, messages)
    
        return ai_msg

    result = await chat("Summarize the document with id 'report.pdf'. Change the document to state that the concondenser tower is 30 meters tall.")
    print(result.content)





# from mcp_client import MCPClient

# async def main():
#     command = "uv"
#     args = ["run", "mcp_server.py"]

#     async with MCPClient(command=command, args=args) as client:
#         tools = await client.list_tools()

#         print("Available MCP tools:")
#         for tool in tools:
#             print("-", tool.name, ":", tool.description)

#         result = await client.call_tool(
#             "read_doc_contents",
#             {"doc_id": "report.pdf"},
#         )

#         tool_text = result.content[0].text

#         response = llm.invoke(
#             f"""
#             Summarize this document:

#             {tool_text}
#             """
#         )

#         print(response.content)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())