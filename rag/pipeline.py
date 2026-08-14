from rag.embeddings import add_document_to_rag, search_rag
from agent.tools import save_report
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()


def index_report(report_content: str, report_name: str = None, metadata: dict = None):
    """Index a generated report into the RAG database"""
    if report_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"report_{timestamp}"

    if metadata is None:
        metadata = {}

    metadata.update({
        "type": "adas_report",
        "indexed_at": datetime.now().isoformat(),
        "report_name": report_name
    })

    add_document_to_rag(
        content=report_content,
        doc_id=report_name,
        metadata=metadata
    )

    console.print(f"[green]Report indexed in RAG: {report_name}[/green]")


def index_existing_reports():
    """Index all existing reports in the reports folder"""
    from config.settings import settings

    reports_dir = settings.reports_dir

    if not reports_dir.exists():
        console.print("[yellow]No reports folder found[/yellow]")
        return

    report_files = list(reports_dir.glob("*.md"))

    if not report_files:
        console.print("[yellow]No reports found to index[/yellow]")
        return

    console.print(
        f"[blue]Indexing {len(report_files)} existing reports...[/blue]")

    for report_file in report_files:
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()

        index_report(
            report_content=content,
            report_name=report_file.stem,
            metadata={"source": str(report_file)}
        )

    console.print(
        f"[bold green]Indexed {len(report_files)} reports successfully[/bold green]")


def query_knowledge_base(question: str) -> str:
    """Query the RAG knowledge base and return relevant context"""
    console.print(f"[blue]Searching knowledge base for: {question}[/blue]")

    results = search_rag(question, n_results=3)

    if not results:
        return "No relevant past reports found in knowledge base."

    context = "\n\n---\n\n".join(results)
    return f"Relevant context from past reports:\n\n{context}"
