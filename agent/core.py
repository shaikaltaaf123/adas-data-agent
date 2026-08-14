from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from config.settings import settings
from rag.pipeline import query_knowledge_base, index_report
from rich.console import Console

console = Console()

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    data_path: str
    analysis_result: str
    rag_context: str
    report: str

# Initialize the LLM
def get_llm():
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1
    )

# pull relevant past knowledge from the RAG store
def retrieve_context(state: AgentState) -> AgentState:
    console.print("[bold cyan]Agent: Searching knowledge base...[/bold cyan]")

    query = state['messages'][-1].content
    rag_context = query_knowledge_base(query)

    return {
        "messages": state["messages"],
        "data_path": state["data_path"],
        "analysis_result": "",
        "rag_context": rag_context,
        "report": ""
    }

# run the actual analysis, using RAG context if we found any
def analyze_data(state: AgentState) -> AgentState:
    console.print("[bold blue]Agent: Analyzing ADAS data...[/bold blue]")

    llm = get_llm()

    messages = [
        SystemMessage(content="""You are an expert ADAS data analyst.
        Your job is to analyze driving datasets, identify edge cases,
        and provide detailed insights about the data quality and content.
        Use any relevant past context provided to enhance your analysis."""),
        HumanMessage(content=f"""Analyze this ADAS dataset:

{state['messages'][-1].content}

Relevant context from past analyses:
{state['rag_context']}

Identify:
1. Data patterns and statistics
2. Edge cases and anomalies
3. Safety critical scenarios
4. Data quality issues
5. Comparison with past reports if relevant""")
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "analysis_result": response.content,
        "data_path": state["data_path"],
        "rag_context": state["rag_context"],
        "report": ""
    }

# turn the analysis into a proper report
def generate_report(state: AgentState) -> AgentState:
    console.print("[bold green]Agent: Generating report...[/bold green]")

    llm = get_llm()

    messages = [
        SystemMessage(content="""You are a technical report writer specializing in
        autonomous driving systems. Write clear, structured, professional reports."""),
        HumanMessage(content=f"""Based on this analysis:
{state['analysis_result']}

Write a professional report with:
1. Executive Summary
2. Key Findings
3. Edge Cases Identified
4. Comparison with Historical Data (if available)
5. Recommendations for Model Improvement
6. Conclusion""")
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "analysis_result": state["analysis_result"],
        "data_path": state["data_path"],
        "rag_context": state["rag_context"],
        "report": response.content
    }

# stash the new report back into RAG so future runs can reference it
def index_to_rag(state: AgentState) -> AgentState:
    console.print("[bold magenta]Agent: Indexing report to knowledge base...[/bold magenta]")

    if state["report"]:
        from datetime import datetime
        report_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        index_report(
            report_content=state["report"],
            report_name=report_name,
            metadata={"data_path": state["data_path"]}
        )

    return state

# Build the agent graph
def build_agent():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("analyze", analyze_data)
    workflow.add_node("report", generate_report)
    workflow.add_node("index", index_to_rag)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", "index")
    workflow.add_edge("index", END)

    return workflow.compile()

# Run the agent
def run_agent(data_description: str, data_path: str = ""):
    console.print("[bold yellow]Starting ADAS Data Analysis Agent...[/bold yellow]")

    agent = build_agent()

    initial_state = {
        "messages": [HumanMessage(content=data_description)],
        "data_path": data_path,
        "analysis_result": "",
        "rag_context": "",
        "report": ""
    }

    result = agent.invoke(initial_state)

    return result
