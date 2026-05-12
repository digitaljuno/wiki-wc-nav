"""Wiki WC Nav — FastAPI web app with live progress streaming."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure wiki_wc_nav.py is importable from the same directory
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import httpx

import wiki_wc_nav as wn

app = FastAPI(title="Wiki WC Nav")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/analyze")
async def analyze(
    lang: str = Query("en,es"),
    no_clickstream: bool = Query(True),
    depth: int = Query(1, ge=1, le=3),
    top: int = Query(30, ge=5, le=200),
    month: str = Query(""),
    cache_dir: str = Query("~/.cache/wiki-wc-nav"),
):
    """Stream analysis progress and results as Server-Sent Events."""

    async def generate():
        def evt(type_: str, **kwargs) -> str:
            return f"data: {json.dumps({'type': type_, **kwargs})}\n\n"

        try:
            langs = [l.strip() for l in lang.split(",") if l.strip()]
            use_clickstream = not no_clickstream
            cache_path = Path(cache_dir).expanduser()

            categories = {l: wn.DEFAULT_CATEGORIES[l] for l in langs if l in wn.DEFAULT_CATEGORIES}

            yield evt("progress", msg="Starting analysis...")

            async with httpx.AsyncClient(timeout=120.0) as client:

                # A. Resolve month
                year, month_num = None, None
                if use_clickstream:
                    yield evt("progress", msg="Resolving latest clickstream month...")
                    try:
                        year, month_num = await wn.resolve_clickstream_month(
                            client, forced=month or None
                        )
                        yield evt("progress", msg=f"Using clickstream: {year:04d}-{month_num:02d}")
                    except Exception as e:
                        yield evt("error", msg=str(e))
                        return

                # B. Discover articles
                yield evt("progress", msg=f"Discovering articles ({', '.join(l.upper() for l in langs)} Wikipedia, depth={depth})...")
                article_map = await wn.discover_articles(client, langs, categories, max_depth=depth)
                for l, data in article_map.items():
                    yield evt("progress", msg=f"[{l.upper()}] {len(data['titles'])} articles found")

                total_articles = sum(len(d["titles"]) for d in article_map.values())
                yield evt("progress", msg=f"Total: {total_articles} articles")

                # C+D. Download + parse clickstream
                nav_data: dict = {l: {} for l in langs}
                if use_clickstream:
                    yield evt("progress", msg="Downloading clickstream files...")
                    try:
                        paths = await asyncio.gather(*[
                            wn.download_clickstream(client, l, year, month_num, cache_path)
                            for l in langs
                        ])
                    except Exception as e:
                        yield evt("error", msg=f"Download failed: {e}")
                        return

                    yield evt("progress", msg="Parsing navigation data (EN takes 2–3 min — please wait)...")
                    loop = asyncio.get_event_loop()
                    parsed = await asyncio.gather(*[
                        loop.run_in_executor(
                            None, wn.parse_clickstream, path, article_map[l]["underscored"]
                        )
                        for l, path in zip(langs, paths)
                    ])
                    for l, nav in zip(langs, parsed):
                        nav_data[l] = nav
                    yield evt("progress", msg="Clickstream parsed")

                # E. Pageviews
                yield evt("progress", msg="Fetching 30-day pageviews...")
                views_map = await wn.fetch_pageviews_all(client, article_map)
                yield evt("progress", msg=f"Pageviews fetched for {len(views_map)} articles")

                # Quality
                yield evt("progress", msg="Checking article quality and translation gaps...")
                quality_map = await wn.fetch_quality_all(client, article_map)
                yield evt("progress", msg=f"Quality checked for {len(quality_map)} articles")

            # Score + rank
            yield evt("progress", msg="Scoring and ranking...")
            records = wn.assemble_and_score(
                article_map, nav_data, views_map, quality_map, use_clickstream
            )

            yield evt("done", records=records, langs=langs, total=len(records))

        except Exception as e:
            yield evt("error", msg=f"Analysis failed: {e}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/article-paths")
async def article_paths(
    title: str = Query(..., min_length=1),
    lang: str = Query("en"),
    month: str = Query(""),
    cache_dir: str = Query("~/.cache/wiki-wc-nav"),
    limit: int = Query(25, ge=5, le=50),
):
    """Stream navigation paths for a single article as Server-Sent Events."""

    async def generate():
        def evt(type_: str, **kwargs) -> str:
            return f"data: {json.dumps({'type': type_, **kwargs})}\n\n"

        try:
            cache_path = Path(cache_dir).expanduser()
            title_underscored = title.strip().replace(" ", "_")
            lang_code = lang.strip().lower()

            yield evt("progress", msg=f"Looking up paths for '{title}' on {lang_code.upper()} Wikipedia...")

            async with httpx.AsyncClient(timeout=120.0) as client:
                # Resolve clickstream month
                yield evt("progress", msg="Checking latest available clickstream month...")
                try:
                    year, month_num = await wn.resolve_clickstream_month(
                        client, forced=month or None
                    )
                    yield evt("progress", msg=f"Using: {year:04d}-{month_num:02d}")
                except Exception as e:
                    yield evt("error", msg=str(e))
                    return

                # Download if not cached
                try:
                    path = await wn.download_clickstream(client, lang_code, year, month_num, cache_path)
                except Exception as e:
                    yield evt("error", msg=f"Download failed: {e}")
                    return

            # Scan (blocking — run in executor)
            yield evt("progress", msg=f"Scanning clickstream for '{title}' (may take 1–3 min for EN)...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, wn.lookup_article_nav, path, title_underscored, limit
            )

            if not result["incoming"] and not result["outgoing"] and not result["external"]:
                yield evt("error", msg=f"No navigation data found for '{title}'. Check the spelling or try a different article.")
                return

            yield evt("done", **result, lang=lang_code, month=f"{year:04d}-{month_num:02d}")

        except Exception as e:
            yield evt("error", msg=f"Lookup failed: {e}")

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/suggest")
async def suggest(q: str = Query(""), lang: str = Query("en")):
    """Wikipedia title autocomplete via MediaWiki search API."""
    if not q.strip():
        return {"results": []}
    api = wn.API_URLS.get(lang, wn.API_URLS["en"])
    params = {
        "action": "opensearch",
        "search": q,
        "limit": "8",
        "namespace": "0",
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(api, params=params, headers=wn.HEADERS)
            data = resp.json()
            return {"results": data[1] if len(data) > 1 else []}
    except Exception:
        return {"results": []}


@app.get("/api/export-csv")
async def export_csv_endpoint(data_json: str = Query(...)):
    """Download results as a CSV file."""
    try:
        records = json.loads(data_json)
    except Exception:
        records = []

    if not records:
        return StreamingResponse(iter([""]), media_type="text/csv")

    fieldnames = [
        "lang", "title", "views_30d", "in_link_n", "out_link_n",
        "search_arrivals", "centrality", "is_stub", "no_refs",
        "has_image", "has_translation", "priority_score", "wiki_url",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="wc-nav-results.csv"'},
    )


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), reload=False)
