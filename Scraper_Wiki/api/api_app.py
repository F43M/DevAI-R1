"""FastAPI server exposing dataset generation endpoints."""

from typing import List, Optional
from fastapi import FastAPI, Query as FastQuery, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from scraper_wiki.models import DatasetRecord
import scraper_wiki as sw
import os
import json
from datetime import datetime
import graphene
from utils.text import clean_text, extract_entities
from utils.compression import load_json_file
from task_queue import publish, clear as clear_queue
import asyncio
from search import indexer

app = FastAPI()

auth_scheme = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    token = os.environ.get("API_TOKEN")
    if token and (credentials is None or credentials.credentials != token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )


DATA_FILE = os.path.join(sw.Config.OUTPUT_DIR, "wikipedia_qa.json")
PROGRESS_FILE = os.path.join(sw.Config.LOG_DIR, "progress.json")
INFO_FILE = os.path.join(sw.Config.OUTPUT_DIR, "dataset_info.json")

JOBS: dict[str, dict] = {}


def load_dataset() -> List[dict]:
    """Return dataset records from ``DATA_FILE`` if present."""

    if not os.path.exists(DATA_FILE):
        return []
    return load_json_file(DATA_FILE)


def load_dataset_info() -> dict:
    """Return metadata saved alongside the dataset."""

    if not os.path.exists(INFO_FILE):
        return {}
    try:
        with open(INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_progress() -> dict:
    """Return scraper progress statistics."""

    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def filter_dataset(
    data: List[dict],
    langs: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[dict]:
    """Filter records by language, category and date range."""

    result = data
    if langs:
        result = [d for d in result if d.get("language") in langs]
    if categories:
        result = [d for d in result if d.get("category") in categories]
    if start_date:
        s_dt = datetime.fromisoformat(start_date)
        result = [
            d
            for d in result
            if "created_at" in d and datetime.fromisoformat(d["created_at"]) >= s_dt
        ]
    if end_date:
        e_dt = datetime.fromisoformat(end_date)
        result = [
            d
            for d in result
            if "created_at" in d and datetime.fromisoformat(d["created_at"]) <= e_dt
        ]
    return result


def enrich_record(record: dict) -> dict:
    """Return record with cleaned text and extracted entities."""
    text = clean_text(record.get("content", ""))
    record = {**record, "clean_content": text, "entities": extract_entities(text)}
    return record


class ScrapeParams(BaseModel):
    lang: Optional[List[str]] | Optional[str] = None
    category: Optional[List[str]] | Optional[str] = None
    format: str = "all"
    plugin: str = "wikipedia"  # e.g. "infobox_parser" or "table_parser"


class JobParams(ScrapeParams):
    start_page: Optional[List[str]] | Optional[str] = None
    depth: int = 1
    rate_limit_delay: Optional[float] = None
    revisions: bool = False
    rev_limit: int = 5


class QueueItem(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    lang: Optional[str] = None


@app.post("/scrape")
async def scrape(params: ScrapeParams, _=Depends(require_token)):
    langs = params.lang
    if isinstance(langs, str):
        langs = [langs]
    cats = params.category
    if isinstance(cats, str):
        cats = [cats]
    if cats is not None:
        cats = [sw.normalize_category(c) or c for c in cats]

    if params.plugin == "wikipedia":
        sw.main(langs, cats, params.format)
    else:
        from plugins import load_plugin, run_plugin

        plg = load_plugin(params.plugin)
        run_plugin(
            plg,
            langs or sw.Config.LANGUAGES,
            cats or list(sw.Config.CATEGORIES),
            params.format,
        )
    return {"status": "ok"}


@app.get("/records", response_model=List[DatasetRecord])
async def get_records(
    lang: List[str] | None = FastQuery(None),
    category: List[str] | None = FastQuery(None),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    _=Depends(require_token),
):
    langs = lang if isinstance(lang, list) else ([lang] if lang else None)
    cats = (
        category if isinstance(category, list) else ([category] if category else None)
    )
    data = load_dataset()
    filtered = filter_dataset(data, langs, cats, start_date, end_date)
    processed = await asyncio.gather(
        *(asyncio.to_thread(enrich_record, rec) for rec in filtered)
    )
    validated = [DatasetRecord.parse_obj(rec).dict() for rec in processed]
    return validated


@app.get("/stats")
async def get_stats(_=Depends(require_token)):
    return load_progress()


@app.get("/search")
async def search_endpoint(q: str, _=Depends(require_token)):
    """Return records matching ``q`` from Elasticsearch."""
    results = await asyncio.to_thread(indexer.query_index, q)
    return results


@app.get("/dataset/summary")
async def dataset_summary(_=Depends(require_token)):
    """Return metadata about the generated dataset."""
    return load_dataset_info()


@app.post("/queue/add")
async def queue_add(items: List[QueueItem], _=Depends(require_token)):
    """Add scraping tasks to the queue."""
    for item in items:
        publish("scrape_tasks", item.model_dump(exclude_none=True))
    return {"status": "ok", "queued": len(items)}


@app.post("/queue/clear")
async def queue_clear(_=Depends(require_token)):
    """Remove all pending scraping tasks and results."""
    clear_queue("scrape_tasks")
    clear_queue("scrape_results")
    return {"status": "cleared"}


@app.post("/jobs")
async def start_job(params: JobParams, _=Depends(require_token)):
    """Start a scraping job in the background and return its ID."""
    langs = params.lang
    if isinstance(langs, str):
        langs = [langs]
    cats = params.category
    if isinstance(cats, str):
        cats = [cats]
    if cats is not None:
        cats = [sw.normalize_category(c) or c for c in cats]
    if isinstance(params.start_page, str):
        start_pages = [params.start_page]
    else:
        start_pages = params.start_page

    job_id = os.urandom(8).hex()

    async def _run_job() -> None:
        try:
            await sw.main_async(
                langs,
                cats,
                params.format,
                params.rate_limit_delay,
                start_pages=start_pages,
                depth=params.depth,
                revisions=params.revisions,
                rev_limit=params.rev_limit,
            )
            JOBS[job_id]["status"] = "completed"
        except Exception as exc:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)

    task = asyncio.create_task(_run_job())
    JOBS[job_id] = {"task": task, "status": "running"}
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, _=Depends(require_token)):
    """Return status information for a running job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running" and job["task"].done():
        if job["task"].exception() is None:
            job["status"] = "completed"
        else:
            job["status"] = "failed"
            job["error"] = str(job["task"].exception())
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
    }


class RecordType(graphene.ObjectType):
    id = graphene.String()
    title = graphene.String()
    language = graphene.String()
    category = graphene.String()
    topic = graphene.String()
    subtopic = graphene.String()
    summary = graphene.String()
    created_at = graphene.String()
    quality_score = graphene.Float()


class Query(graphene.ObjectType):
    records = graphene.List(
        RecordType,
        lang=graphene.List(graphene.String),
        category=graphene.List(graphene.String),
        start_date=graphene.String(),
        end_date=graphene.String(),
    )

    def resolve_records(
        root, info, lang=None, category=None, start_date=None, end_date=None
    ):
        data = load_dataset()
        filtered = filter_dataset(data, lang, category, start_date, end_date)
        return [enrich_record(rec) for rec in filtered]


schema = graphene.Schema(query=Query)


@app.post("/graphql")
async def graphql_endpoint(request: Request, _=Depends(require_token)):
    body = await request.json()
    result = schema.execute(body.get("query"), variable_values=body.get("variables"))
    return JSONResponse(result.data)


@app.get("/health")
async def health_check():
    """Basic health check endpoint used by containers and load balancers."""
    return {"status": "ok"}
