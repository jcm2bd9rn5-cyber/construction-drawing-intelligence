"""Usage: python -m scripts.ingest PATH [PATH ...]"""
import argparse
import logging
from pathlib import Path
from app.core.database import SessionLocal
from app.core.config import get_settings
from app.services.ingestion import ingest_document

def main():
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--vision", action="store_true", help="Enable Vision explicitly (enabled by default when the API key exists)")
    parser.add_argument("--embed", action="store_true", help="Enable embeddings explicitly (enabled by default when the API key exists)")
    parser.add_argument("--no-vision", action="store_true", help="Disable automatic Vision fallback")
    parser.add_argument("--no-embed", action="store_true", help="Disable embedding generation")
    args = parser.parse_args()
    settings = get_settings()
    if (args.embed or args.vision) and not settings.gemini_api_key:
        parser.error("GEMINI_API_KEY is required for --embed or --vision")
    vision_enabled = bool(settings.gemini_api_key) and not args.no_vision
    embed_enabled = bool(settings.gemini_api_key) and not args.no_embed
    from app.ai.embeddings import GeminiEmbedder
    from app.ai.gemini import GeminiVision
    embedder = GeminiEmbedder() if embed_enabled else None
    vision = GeminiVision() if vision_enabled else None
    with SessionLocal() as session:
        for index, path in enumerate(args.paths):
            job = ingest_document(session, path, embedder=embedder, vision=vision)
            print(f"{path}: {job.status}" + (f" ({job.error})" if job.error else ""))
            print(
                f"  pages={job.pages_created} chunks={job.chunks_created} "
                f"failed_pages={job.pages_failed} "
                f"embeddings={job.embeddings_created} vision_pages={job.vision_pages} "
                f"text_pages={job.text_pages} searchable={job.searchable} "
                f"embedding_requests={job.embedding_api_requests} "
                f"embedding_batches={job.embedding_batches}"
            )
            if job.status == "DEFERRED" and job.error_type == "DAILY_QUOTA_EXHAUSTED":
                remaining = len(args.paths) - index - 1
                print(
                    "Embedding daily quota exhausted; stopping bulk ingestion "
                    f"with {remaining} file(s) remaining. "
                    f"retry_after_seconds={job.retry_after_seconds}"
                )
                break
            for warning in job.warnings_json or []:
                print(f"  WARNING: {warning}")

if __name__ == "__main__":
    main()
