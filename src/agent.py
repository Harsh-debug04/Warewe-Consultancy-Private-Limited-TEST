import os
import json
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from duckduckgo_search import DDGS

# Define the State
class AgentState(TypedDict):
    goal: str
    search_queries: List[str]
    search_results: str
    draft: str
    critique: str
    approved: bool
    final_output: str

# Define schema for planner
class SearchPlan(BaseModel):
    queries: List[str] = Field(description="List of search queries to research the goal")

# Define schema for reviewer
class ReviewResult(BaseModel):
    critique: str = Field(description="Constructive critique of the draft. Empty if approved.")
    approved: bool = Field(description="True if the draft meets the goal and quality standards, False otherwise.")

# Node: Planner
def planner(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_msg = SystemMessage(content="You are a planning assistant. Break down the user's goal into 2-3 specific search queries to find the most relevant and recent news/articles.")
    human_msg = HumanMessage(content=f"Goal: {state['goal']}")

    # Use structured output for the queries
    structured_llm = llm.with_structured_output(SearchPlan)
    result = structured_llm.invoke([system_msg, human_msg])

    return {"search_queries": result.queries}

# Node: Researcher
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

# Node: Writer
def writer(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    system_msg = SystemMessage(content="You are an expert newsletter writer. Create a clean, engaging newsletter in Markdown format based on the search results. Include a catchy title, a brief introduction, and summarize the top news items. Add a concluding remark.")

    prompt = f"Goal: {state['goal']}\n\nSearch Results:\n{state.get('search_results', '')}\n\nDraft the newsletter now."
    if state.get("critique"):
        prompt += f"\n\nPrevious Critique to address:\n{state['critique']}\n\nPrevious Draft:\n{state.get('draft', '')}"

    human_msg = HumanMessage(content=prompt)
    result = llm.invoke([system_msg, human_msg])

    return {"draft": result.content}

# Node: Reviewer
def reviewer(state: AgentState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_msg = SystemMessage(content="You are an editor reviewing a newsletter draft. Check if it meets the user's goal, has a good tone, and accurately summarizes the news without hallucinations. If it needs work, provide a critique and set approved to false. If it is excellent, set approved to true.")

    prompt = f"Goal: {state['goal']}\n\nDraft:\n{state['draft']}"
    human_msg = HumanMessage(content=prompt)

    structured_llm = llm.with_structured_output(ReviewResult)
    result = structured_llm.invoke([system_msg, human_msg])

    return {"critique": result.critique, "approved": result.approved}

# Node: Sender
def sender(state: AgentState):
    # Simulate sending the email by saving it to a file
    filename = "newsletter_output.md"
    with open(filename, "w") as f:
        f.write(state["draft"])

    print(f"Newsletter successfully generated and saved to {filename}")
    return {"final_output": f"Newsletter sent! Content saved to {filename}"}

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Conditional edge logic
def should_continue(state: AgentState):
    if state["approved"]:
        return "human_approval"
    return "writer" # Reviser is just the writer addressing critique

# Node: Human Approval (HITL Breakpoint)
def human_approval(state: AgentState):
    # This node acts as a pass-through when autonomous,
    # but serves as an interrupt point for HITL.
    return {}

# Compile the Graph
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    workflow.add_node("reviewer", reviewer)
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("sender", sender)

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")

    # Conditional edge from reviewer
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "human_approval": "human_approval",
            "writer": "writer"
        }
    )

    workflow.add_edge("human_approval", "sender")
    workflow.add_edge("sender", END)

    # Compile with memory for interrupts
    memory = MemorySaver()
    # We will interrupt BEFORE human_approval if HITL is needed
    graph = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
    return graph

def run_agent(goal: str, hitl: bool = False, thread_id: str = "1", action: str = None):
    """
    Run the newsletter agent.
    If hitl is True, it will pause before sending and wait for approval.
    action can be 'approve' or 'reject' when continuing from a paused state.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # Check current state
    current_state = graph.get_state(config)

    if current_state and current_state.next:
        # Agent is paused at human_approval
        if hitl:
            if action == "approve":
                print("Human approved. Continuing to send...")
                # Continue execution
                for output in graph.stream(None, config, stream_mode="values"):
                    pass
                return graph.get_state(config).values
            elif action == "reject":
                print("Human rejected. Resetting to writer...")
                # Update state with human critique
                # (In a real app, we'd accept text feedback)
                graph.update_state(config, {"approved": False, "critique": "User rejected the draft. Please rewrite entirely."}, as_node="human_approval")
                # Now we need to manually route back to writer or planner...
                # For simplicity, if rejected, we will just return the state and let UI handle it
                return graph.get_state(config).values
            else:
                return current_state.values # Just return current state waiting for action
        else:
            # If HITL was turned off mid-way, just continue
            for output in graph.stream(None, config, stream_mode="values"):
                pass
            return graph.get_state(config).values

    else:
        # Start fresh
        initial_state = {
            "goal": goal,
            "search_queries": [],
            "search_results": "",
            "draft": "",
            "critique": "",
            "approved": False,
            "final_output": ""
        }

        # We process the graph until interruption or end
        # Notice we use stream to force execution, even if we just want the final result
        for output in graph.stream(initial_state, config, stream_mode="values"):
            pass

        final_state = graph.get_state(config)

        # If we are fully autonomous but hit the interrupt (because graph ALWAYS interrupts before human_approval)
        if not hitl and final_state.next:
            # Continue automatically
            for output in graph.stream(None, config, stream_mode="values"):
                pass
            final_state = graph.get_state(config)

        return final_state.values

if __name__ == "__main__":
    # Test execution
    pass
