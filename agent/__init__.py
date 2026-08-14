from pydantic_settings import BaseSettings
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Data Configuration
    data_dir: Path = BASE_DIR / "data" / "samples"
    reports_dir: Path = BASE_DIR / "reports"

    # Vector Database Configuration
    chroma_dir: Path = BASE_DIR / "data" / "chroma_db"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"


# Global settings instance
settings = Settings()
