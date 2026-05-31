import streamlit as st
import os
import uuid
from agent import build_graph

# Configure Streamlit page
st.set_page_config(page_title="AI Newsletter Agent", page_icon="📰", layout="wide")

# Initialize session state for thread_id if not exists
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False

st.title("📰 Autonomous AI Newsletter Agent")
st.markdown("Generate a weekly newsletter on any topic using multi-step reasoning, research, and writing.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    mode = st.radio("Operation Mode", ["Fully Autonomous", "Human-in-the-Loop (HITL)"])
    hitl_enabled = mode == "Human-in-the-Loop (HITL)"

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. **Planner**: Breaks down your goal into search queries.")
    st.markdown("2. **Researcher**: Searches DuckDuckGo for latest news.")
    st.markdown("3. **Writer**: Drafts the newsletter.")
    st.markdown("4. **Reviewer**: Critiques the draft (loops back to Writer if needed).")
    st.markdown("5. **Sender**: Saves the final newsletter.")

# Main content
goal = st.text_area("Newsletter Goal", value="Create a weekly newsletter on latest AI agent news and send it to our subscribers.", height=100)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Run Agent", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API Key in the sidebar.")
            st.stop()

        st.session_state.agent_running = True
        st.session_state.thread_id = str(uuid.uuid4()) # Reset thread for new run

if st.session_state.agent_running:
    # Build graph directly to interact with streams
    graph = build_graph()
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.subheader("Agent Progress")

    # Check if we are paused at human_approval
    current_state = graph.get_state(config)

    if current_state and current_state.next:
        # We are paused waiting for HITL approval
        st.warning("Agent is waiting for Human Approval.")
        st.markdown("### Current Draft")
        st.markdown(current_state.values.get("draft", "No draft available."))

        col_app, col_rej = st.columns(2)
        with col_app:
            if st.button("✅ Approve and Send", use_container_width=True):
                with st.spinner("Sending newsletter..."):
                    if not hitl_enabled:
                         # If mode switched mid-run
                         pass
                    for output in graph.stream(None, config, stream_mode="values"):
                        pass
                st.success("Newsletter Approved and Sent!")
                st.session_state.agent_running = False
                st.rerun()

        with col_rej:
            if st.button("❌ Reject and Rewrite", use_container_width=True):
                feedback = st.text_input("Feedback for Rewrite:")
                if feedback:
                    # Update state manually
                    graph.update_state(config, {"approved": False, "critique": feedback}, as_node="human_approval")
                    with st.spinner("Rewriting..."):
                        # We force the graph to continue. Since 'human_approval' currently routes to 'sender',
                        # we need to ensure the graph handles rewrite logic, or we just let it restart.
                        # For simplicity, if we update state with approved=False, we can restart from writer manually
                        pass
                st.warning("Rejection logic implementation is simplified. To rewrite, please clear and start over for now.")
                st.session_state.agent_running = False

    else:
        # Run graph from start
        initial_state = {
            "goal": goal,
            "search_queries": [],
            "search_results": "",
            "draft": "",
            "critique": "",
            "approved": False,
            "final_output": ""
        }

        status_container = st.empty()

        with st.spinner("Agent is working..."):
            for event in graph.stream(initial_state, config):
                # event is a dict with node name as key and state update as value
                for node_name, state_update in event.items():
                    with status_container.container():
                        st.info(f"🟢 **Completed Node:** {node_name.capitalize()}")

                        if node_name == "planner":
                            st.write("**Search Queries:**")
                            st.write(state_update.get("search_queries", []))
                        elif node_name == "researcher":
                            st.write("Found search results.")
                            with st.expander("View Results"):
                                st.write(state_update.get("search_results", "")[:1000] + "...")
                        elif node_name == "writer":
                            st.write("Drafting newsletter...")
                        elif node_name == "reviewer":
                            critique = state_update.get("critique", "")
                            approved = state_update.get("approved", False)
                            st.write(f"**Reviewer Approved:** {approved}")
                            if not approved:
                                st.warning(f"**Critique:** {critique}")
                            else:
                                st.success("Draft passed review!")

        final_state = graph.get_state(config)

        if hitl_enabled and final_state.next:
            st.rerun() # Refresh UI to show approval buttons

        elif not hitl_enabled and final_state.next:
            # Continue automatically
            with st.spinner("Continuing autonomously..."):
                for event in graph.stream(None, config):
                     for node_name, state_update in event.items():
                        st.info(f"🟢 **Completed Node:** {node_name.capitalize()}")
            final_state = graph.get_state(config)

        st.session_state.agent_running = False

        st.markdown("---")
        st.subheader("🎉 Final Output")
        final_vals = final_state.values
        st.markdown(final_vals.get("draft", ""))
        st.success(final_vals.get("final_output", ""))
