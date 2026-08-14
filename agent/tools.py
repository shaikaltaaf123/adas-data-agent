import pandas as pd
from pathlib import Path
from config.settings import settings
from rich.console import Console
from datetime import datetime

console = Console()

def load_dataset(file_path: str) -> dict:
    """Load a CSV or JSON driving dataset and return basic info"""
    path = Path(file_path)

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif path.suffix == ".json":
            df = pd.read_json(file_path)
        else:
            return {"error": "Unsupported file format. Use CSV or JSON."}

        info = {
            "file_name": path.name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "sample_data": df.head(5).to_string()
        }

        console.print(f"[green]Dataset loaded: {path.name} ({len(df)} rows)[/green]")
        return info

    except Exception as e:
        return {"error": str(e)}


def detect_edge_cases(file_path: str) -> dict:
    """Detect edge cases and anomalies in the dataset"""
    path = Path(file_path)

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif path.suffix == ".json":
            df = pd.read_json(file_path)
        else:
            return {"error": "Unsupported file format"}

        edge_cases = {}

        # Check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
            if len(outliers) > 0:
                edge_cases[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": round(len(outliers)/len(df)*100, 2)
                }

        # Check for missing values
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].to_dict()

        # Basic statistics
        stats = df.describe().to_string() if len(numeric_cols) > 0 else "No numeric columns"

        return {
            "total_rows": len(df),
            "edge_cases_by_column": edge_cases,
            "missing_values": missing_cols,
            "statistics": stats
        }

    except Exception as e:
        return {"error": str(e)}


def save_report(report_content: str, file_name: str = None) -> str:
    """Save the generated report to the reports folder"""
    reports_dir = settings.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"adas_report_{timestamp}.md"

    report_path = reports_dir / file_name

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    console.print(f"[bold green]Report saved: {report_path}[/bold green]")
    return str(report_path)


def get_dataset_summary(file_path: str) -> str:
    """Get a human readable summary of the dataset for the LLM"""
    info = load_dataset(file_path)

    if "error" in info:
        return f"Error loading dataset: {info['error']}"

    edge_cases = detect_edge_cases(file_path)

    summary = f"""
DATASET SUMMARY
===============
File: {info['file_name']}
Total Rows: {info['total_rows']}
Total Columns: {info['total_columns']}
Columns: {', '.join(info['columns'])}

DATA TYPES:
{info['data_types']}

MISSING VALUES:
{info['missing_values']}

EDGE CASES DETECTED:
{edge_cases.get('edge_cases_by_column', 'None detected')}

STATISTICS:
{edge_cases.get('statistics', 'N/A')}

SAMPLE DATA (first 5 rows):
{info['sample_data']}
"""
    return summary
