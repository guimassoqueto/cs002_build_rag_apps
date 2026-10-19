from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LESSONS_DIR = PROJECT_ROOT / "lessons"


def doc_path(script: str) -> Path:
    """Return the notes a lesson was written from: the .md file beside the script.

    lessons/module02/01_langchain_core_concepts.py
        -> lessons/module02/01_langchain_core_concepts.md

    Args:
        script: the calling script's __file__.
    """
    return Path(script).resolve().with_suffix(".md")


def output_path(script: str, filename: str) -> Path:
    """Return the path a lesson should write a generated file to.

    lessons/module02/01_langchain_core_concepts.py + "concepts.json"
        -> lessons/module02/output/01_langchain_core_concepts/concepts.json

    The parent directory is created if needed, so callers can write immediately.

    Args:
        script: the calling script's __file__.
        filename: name of the file to generate.
    """
    lesson = Path(script).resolve()
    path = lesson.parent / "output" / lesson.stem / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Hugging Face image captioning model used by the Gradio lessons.
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# Local RAG stack: Ollama and Chroma run in Docker (lessons/module02/03_qa_bot_with_local_qwen3/compose.yaml).
OLLAMA_URL = "http://localhost:11434"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Qwen3-8B Q8_0 with the chat template fixed so thinking can be turned off, built from the lesson's Modelfile.
QWEN3_RAG_MODEL = "qwen3-8b-rag"
QWEN3_EMBEDDING_MODEL = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
