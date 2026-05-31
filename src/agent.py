from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    goal: str
    api_key: str  # Store API key in state to avoid global env vars
    search_queries: List[str]
    search_results: str
    draft: str
    critique: str
    approved: bool
    final_output: str

class SearchPlan(BaseModel):
    queries: List[str] = Field(description="List of search queries to research the goal")

class ReviewResult(BaseModel):
    critique: str = Field(description="Constructive critique of the draft. Empty if approved.")
    approved: bool = Field(description="True if the draft meets the goal and quality standards, False otherwise.")

def get_llm(state: AgentState, temperature: float = 0):
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=state["api_key"])

def planner(state: AgentState):
    llm = get_llm(state)
    system_msg = SystemMessage(content="You are a planning assistant. Break down the user's goal into 2-3 specific search queries to find the most relevant and recent news/articles.")
    human_msg = HumanMessage(content=f"Goal: {state['goal']}")

    structured_llm = llm.with_structured_output(SearchPlan)
    result = structured_llm.invoke([system_msg, human_msg])
    return {"search_queries": result.queries}

def researcher(state: AgentState):
    queries = state.get("search_queries", [])
    all_results = []

    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.text(q, max_results=3)
                for r in results:
                    all_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}\n")
            except Exception as e:
                all_results.append(f"Error searching for {q}: {e}")

    combined_results = "\n---\n".join(all_results)
    return {"search_results": combined_results}

def writer(state: AgentState):
    llm = get_llm(state, temperature=0.7)
    system_msg = SystemMessage(content="You are an expert newsletter writer. Create a clean, engaging newsletter in Markdown format based on the search results. Include a catchy title, a brief introduction, and summarize the top news items. Add a concluding remark.")

    prompt = f"Goal: {state['goal']}\n\nSearch Results:\n{state.get('search_results', '')}\n\nDraft the newsletter now."
    if state.get("critique"):
        prompt += f"\n\nPrevious Critique to address:\n{state['critique']}\n\nPrevious Draft:\n{state.get('draft', '')}"

    human_msg = HumanMessage(content=prompt)
    result = llm.invoke([system_msg, human_msg])
    return {"draft": result.content}

def reviewer(state: AgentState):
    llm = get_llm(state)
    system_msg = SystemMessage(content="You are an editor reviewing a newsletter draft. Check if it meets the user's goal, has a good tone, and accurately summarizes the news without hallucinations. If it needs work, provide a critique and set approved to false. If it is excellent, set approved to true.")

    prompt = f"Goal: {state['goal']}\n\nDraft:\n{state['draft']}"
    human_msg = HumanMessage(content=prompt)

    structured_llm = llm.with_structured_output(ReviewResult)
    result = structured_llm.invoke([system_msg, human_msg])
    return {"critique": result.critique, "approved": result.approved}

def sender(state: AgentState):
    filename = "newsletter_output.md"
    with open(filename, "w") as f:
        f.write(state["draft"])
    return {"final_output": f"Newsletter sent! Content saved to {filename}"}

def should_continue(state: AgentState):
    if state["approved"]:
        return "human_approval"
    return "writer"

def after_human(state: AgentState):
    # Route based on whether human approved or rejected
    if state["approved"]:
        return "sender"
    return "writer"

def human_approval(state: AgentState):
    return {}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("sender", sender)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "human_approval": "human_approval",
            "writer": "writer"
        }
    )

    workflow.add_conditional_edges(
        "human_approval",
        after_human,
        {
            "sender": "sender",
            "writer": "writer"
        }
    )

    workflow.add_edge("sender", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])

graph = build_graph()
