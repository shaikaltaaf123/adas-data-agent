from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
from agent.core import run_agent
from agent.tools import get_dataset_summary, save_report
from config.settings import settings
from rich.console import Console

console = Console()

app = FastAPI(
    title="ADAS Data Analysis Agent",
    description="An AI agent that analyzes ADAS driving datasets and generates reports",
    version="1.0.0"
)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/ui")
def serve_ui():
    return FileResponse("frontend/index.html")

# Request model for text based analysis
class AnalysisRequest(BaseModel):
    data_description: str
    save_report_file: bool = True

# Health check endpoint
@app.get("/")
def root():
    return {"status": "running", "agent": "ADAS Data Analysis Agent v1.0"}

# Analyze uploaded CSV file
@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    upload_dir = settings.data_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    file_path = upload_dir / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        console.print(f"[blue]File uploaded: {safe_filename}[/blue]")

        # Get dataset summary
        summary = get_dataset_summary(str(file_path))

        # Run the agent
        result = run_agent(summary, str(file_path))

        # Save the report
        report_path = save_report(result['report'])

        return JSONResponse(content={
            "status": "success",
            "file": safe_filename,
            "report": result['report'],
            "report_saved_to": report_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analyze from text description
@app.post("/analyze/text")
async def analyze_text(request: AnalysisRequest):
    try:
        result = run_agent(request.data_description)

        report_path = None
        if request.save_report_file:
            report_path = save_report(result['report'])

        return JSONResponse(content={
            "status": "success",
            "report": result['report'],
            "report_saved_to": report_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get list of saved reports
@app.get("/reports")
def list_reports():
    reports_dir = settings.reports_dir

    if not reports_dir.exists():
        return {"reports": []}

    reports = [f.name for f in reports_dir.glob("*.md")]
    return {"reports": reports, "total": len(reports)}
