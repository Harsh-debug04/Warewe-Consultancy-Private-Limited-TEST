import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from src.agent import graph

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/stream")
async def api_stream(request: Request):
    payload = await request.json()
    goal = payload.get("goal", "")
    hitl = payload.get("hitl", False)
    thread_id = payload.get("thread_id", "default")
    action = payload.get("action", "start")
    feedback = payload.get("feedback", "")

    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            async def process_stream(initial_or_none):
                for event in graph.stream(initial_or_none, config):
                    for node_name, state_update in event.items():
                        yield json.dumps({"type": "step", "node": node_name, "status": "active"}) + "\n"

                        if node_name == "planner":
                            queries = state_update.get('search_queries', [])
                            yield json.dumps({"type": "log", "log_type": "action", "message": f"Generated search queries: {queries}"}) + "\n"
                        elif node_name == "researcher":
                            yield json.dumps({"type": "log", "log_type": "tool", "message": f"SearchEngine retrieved results."}) + "\n"
                        elif node_name == "writer":
                            draft = state_update.get("draft", "")
                            yield json.dumps({"type": "log", "log_type": "action", "message": "Drafting newsletter content..."}) + "\n"
                            yield json.dumps({"type": "draft_update", "content": draft}) + "\n"
                        elif node_name == "reviewer":
                            approved = state_update.get("approved", False)
                            critique = state_update.get("critique", "")
                            if approved:
                                yield json.dumps({"type": "log", "log_type": "reflection", "message": "Draft passed review."}) + "\n"
                            else:
                                yield json.dumps({"type": "log", "log_type": "reflection", "message": f"Draft needs revision. Critique: {critique}"}) + "\n"
                        elif node_name == "sender":
                             yield json.dumps({"type": "log", "log_type": "action", "message": "Sending finalized newsletter."}) + "\n"
                             yield json.dumps({"type": "step", "node": node_name, "status": "done"}) + "\n"

                        await asyncio.sleep(0.1)
                        yield json.dumps({"type": "step", "node": node_name, "status": "done"}) + "\n"

                final_state = graph.get_state(config)
                if final_state.next:
                    if hitl:
                         yield json.dumps({"type": "paused"}) + "\n"
                    else:
                         yield json.dumps({"type": "log", "log_type": "system", "message": "Auto mode: Skipping human approval..."}) + "\n"
                         graph.update_state(config, {"approved": True}, as_node="human_approval")
                         # recursively process the rest of the graph
                         async for result in process_stream(None):
                             yield result
                else:
                     yield json.dumps({"type": "finished"}) + "\n"

            if action == "start":
                initial_state = {
                    "goal": goal,
                    "search_queries": [],
                    "search_results": "",
                    "draft": "",
                    "critique": "",
                    "approved": False,
                    "final_output": ""
                }
                yield json.dumps({"type": "log", "log_type": "system", "message": "Initializing agent sequence..."}) + "\n"
                async for result in process_stream(initial_state):
                    yield result

            elif action == "approve":
                yield json.dumps({"type": "log", "log_type": "system", "message": "Approval received. Continuing..."}) + "\n"
                graph.update_state(config, {"approved": True}, as_node="human_approval")
                async for result in process_stream(None):
                    yield result

            elif action == "reject":
                yield json.dumps({"type": "log", "log_type": "system", "message": "Rejection received. Rewriting..."}) + "\n"
                graph.update_state(config, {"approved": False, "critique": feedback}, as_node="human_approval")
                yield json.dumps({"type": "step", "node": "writer", "status": "pending"}) + "\n"
                yield json.dumps({"type": "step", "node": "reviewer", "status": "pending"}) + "\n"
                async for result in process_stream(None):
                    yield result

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
