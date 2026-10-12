"""Call each Legal Sections MCP tool once and print a compact summary (no section text)."""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/mcp"
CALLS = [
    ("get_section", {"act": "bns", "section_number": 103}),
    ("get_section", {"act": "ipc", "section_number": 4}),
    ("get_section", {"act": "xyz", "section_number": 1}),
    ("list_sections", {"act": "bns", "limit": 3}),
    ("list_chapters", {"act": "bns"}),
    ("search_sections_by_heading", {"act": "bns", "query": "murder", "limit": 5}),
    ("get_sections_in_range", {"act": "ipc", "start": 1, "end": 10}),
]


def brief(d: dict) -> dict:
    out = {"status": d.get("status")}
    for key in ("items", "chapters"):
        if key in d:
            out[key] = len(d[key])
    if "total" in d:
        out["total"] = d["total"]
    if isinstance(d.get("section"), dict):
        s = d["section"]
        out["section"] = {"id": s["section_id"], "text_chars": len(s.get("text") or ""),
                          "text_available": s.get("text_available")}
    if d.get("items") and "section_id" in d["items"][0]:
        out["ids"] = [i["section_id"] for i in d["items"]][:12]
    if d.get("message"):
        out["message"] = d["message"][:80]
    return out


async def main() -> None:
    async with streamablehttp_client(URL) as (r, w, _), ClientSession(r, w) as s:
        await s.initialize()
        print("tools:", [t.name for t in (await s.list_tools()).tools])
        for name, args in CALLS:
            res = await s.call_tool(name, args)
            data = res.structuredContent or json.loads(res.content[0].text)
            data = data.get("result", data) if set(data) == {"result"} else data
            print(name, args, "->", json.dumps(brief(data)))


asyncio.run(main())
