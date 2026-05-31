from dotenv import load_dotenv
load_dotenv()
import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.agent import graph

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

async def stream_agent(websocket: WebSocket, payload: dict):
    goal = payload.get("goal", "")
    api_key = payload.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    hitl = payload.get("hitl", False)
    thread_id = payload.get("thread_id", "default")
    action = payload.get("action", "start")
    feedback = payload.get("feedback", "")

    config = {"configurable": {"thread_id": thread_id}}

    async def process_stream(initial_or_none):
        for event in graph.stream(initial_or_none, config):
            for node_name, state_update in event.items():
                await websocket.send_json({"type": "step", "node": node_name, "status": "active"})

                if node_name == "planner":
                    queries = state_update.get('search_queries', [])
                    await websocket.send_json({"type": "log", "log_type": "action", "message": f"Generated search queries: {queries}"})
                elif node_name == "researcher":
                    await websocket.send_json({"type": "log", "log_type": "tool", "message": f"SearchEngine retrieved results."})
                elif node_name == "writer":
                    draft = state_update.get("draft", "")
                    await websocket.send_json({"type": "log", "log_type": "action", "message": "Drafting newsletter content..."})
                    await websocket.send_json({"type": "draft_update", "content": draft})
                elif node_name == "reviewer":
                    approved = state_update.get("approved", False)
                    critique = state_update.get("critique", "")
                    if approved:
                        await websocket.send_json({"type": "log", "log_type": "reflection", "message": "Draft passed review."})
                    else:
                        await websocket.send_json({"type": "log", "log_type": "reflection", "message": f"Draft needs revision. Critique: {critique}"})
                elif node_name == "sender":
                     await websocket.send_json({"type": "log", "log_type": "action", "message": "Sending finalized newsletter."})
                     await websocket.send_json({"type": "step", "node": node_name, "status": "done"})

                await asyncio.sleep(0.5)
                await websocket.send_json({"type": "step", "node": node_name, "status": "done"})

        final_state = graph.get_state(config)
        if final_state.next:
            if hitl:
                 await websocket.send_json({"type": "paused"})
            else:
                 await websocket.send_json({"type": "log", "log_type": "system", "message": "Auto mode: Skipping human approval..."})
                 # If autonomous, we manually update state to approved so graph continues to sender
                 graph.update_state(config, {"approved": True}, as_node="human_approval")
                 await process_stream(None)
        else:
             await websocket.send_json({"type": "finished"})

    try:
        if action == "start":
            initial_state = {
                "goal": goal,
                "api_key": api_key,
                "search_queries": [],
                "search_results": "",
                "draft": "",
                "critique": "",
                "approved": False,
                "final_output": ""
            }
            await websocket.send_json({"type": "log", "log_type": "system", "message": "Initializing agent sequence..."})
            await process_stream(initial_state)

        elif action == "approve":
            await websocket.send_json({"type": "log", "log_type": "system", "message": "Approval received. Continuing..."})
            # Ensure state is set to approved so the conditional edge routes to sender
            graph.update_state(config, {"approved": True}, as_node="human_approval")
            await process_stream(None)

        elif action == "reject":
            await websocket.send_json({"type": "log", "log_type": "system", "message": "Rejection received. Rewriting..."})
            graph.update_state(config, {"approved": False, "critique": feedback}, as_node="human_approval")
            # Clear UI step for writer and reviewer so they show up actively again
            await websocket.send_json({"type": "step", "node": "writer", "status": "pending"})
            await websocket.send_json({"type": "step", "node": "reviewer", "status": "pending"})
            await process_stream(None)

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            await stream_agent(websocket, payload)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
         print(f"WS Error: {e}")
         try:
             await websocket.send_json({"type": "error", "message": str(e)})
         except:
             pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
