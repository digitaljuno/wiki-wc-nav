"""Wikipedia FIFA World Cup navigation analyzer for Maremoto.

Identifies which Wikipedia articles to edit by combining Wikimedia Clickstream
navigation data, Pageviews API, and quality signals across EN and ES Wikipedia.

Usage:
    # Fast mode (no clickstream download, ~2 min):
    python wiki_wc_nav.py --no-clickstream --top 10

    # Full analysis with CSV export:
    python wiki_wc_nav.py --top 30 --output results.csv

    # Spanish only, fast mode:
    python wiki_wc_nav.py --lang es --no-clickstream

    # Force a specific cached month:
    python wiki_wc_nav.py --month 2026-03
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import math
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "wiki-wc-nav"

HEADERS = {
    "User-Agent": "WikiWcNav/1.0 (FIFA WC 2026 analysis; Maremoto program)",
}

API_URLS = {
    "en": "https://en.wikipedia.org/w/api.php",
    "es": "https://es.wikipedia.org/w/api.php",
}

PAGEVIEWS_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
CLICKSTREAM_BASE = "https://dumps.wikimedia.org/other/clickstream"

DEFAULT_CATEGORIES = {
    "en": ["2026_FIFA_World_Cup", "FIFA_World_Cup"],
    "es": ["Copa_Mundial_de_Fútbol_de_2026", "Copa_Mundial_de_Fútbol"],
}

PARTNER_LANG = {"en": "es", "es": "en"}

_RETRY_AFTER = 2.0  # seconds to wait on 429 before retrying

REF_TEMPLATES = (
    "refimprove", "unreferenced", "citation needed", "nofootnotes",
    "more citations", "sin referencias", "cita requerida", "más referencias",
    "wikificar", "no references",
)


# ---------------------------------------------------------------------------
# Phase A: Resolve clickstream month
# ---------------------------------------------------------------------------

async def resolve_clickstream_month(
    client: httpx.AsyncClient, forced: str | None = None
) -> tuple[int, int]:
    """Find the most recent available clickstream month via HEAD requests."""
    if forced:
        return int(forced[:4]), int(forced[5:7])

    today = date.today()
    year, month = today.year, today.month

    for _ in range(4):
        if month == 1:
            year, month = year - 1, 12
        else:
            month -= 1

        url = (
            f"{CLICKSTREAM_BASE}/{year:04d}-{month:02d}/"
            f"clickstream-enwiki-{year:04d}-{month:02d}.tsv.gz"
        )
        try:
            resp = await client.head(url, headers=HEADERS, follow_redirects=True)
            if resp.status_code == 200:
                return year, month
        except Exception:
            continue

    raise RuntimeError("No clickstream data found in last 4 months.")


# ---------------------------------------------------------------------------
# Phase B: Article discovery
# ---------------------------------------------------------------------------

async def discover_articles(
    client: httpx.AsyncClient,
    langs: list[str],
    categories: dict[str, list[str]],
    max_depth: int = 2,
) -> dict[str, dict]:
    """BFS category crawl for each language.

    Returns:
        {lang: {"titles": set[str], "underscored": set[str]}}
    """
    results = {}

    for lang in langs:
        lang_cats = categories.get(lang, DEFAULT_CATEGORIES.get(lang, []))
        api_url = API_URLS[lang]
        prefix = "Categoría:" if lang == "es" else "Category:"

        visited: set[str] = set()
        articles: set[str] = set()
        queue: list[tuple[str, int]] = []

        for cat in lang_cats:
            if not cat.startswith("Category:") and not cat.startswith("Categoría:"):
                cat = f"{prefix}{cat}"
            queue.append((cat, 0))

        print(
            f"  [{lang}] Crawling {len(lang_cats)} root categories (depth={max_depth})...",
            flush=True,
        )

        while queue:
            cat, depth = queue.pop(0)
            if cat in visited:
                continue
            visited.add(cat)
            await asyncio.sleep(0.15)

            cmcontinue = None
            while True:
                params = {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": cat,
                    "cmtype": "page|subcat",
                    "cmlimit": "50",
                    "format": "json",
                    "origin": "*",
                }
                if cmcontinue:
                    params["cmcontinue"] = cmcontinue

                try:
                    resp = await client.get(api_url, params=params, headers=HEADERS)
                    if resp.status_code == 429:
                        await asyncio.sleep(_RETRY_AFTER)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    print(f"  Warning: skipping {cat}: {e}", file=sys.stderr)
                    break

                for m in data.get("query", {}).get("categorymembers", []):
                    if m["ns"] == 0:
                        articles.add(m["title"])
                    elif m["ns"] == 14 and depth < max_depth:
                        queue.append((m["title"], depth + 1))

                cmcontinue = data.get("continue", {}).get("cmcontinue")
                if not cmcontinue:
                    break

                await asyncio.sleep(0.2)

        underscored = {t.replace(" ", "_") for t in articles}
        results[lang] = {"titles": articles, "underscored": underscored}
        print(f"  [{lang}] Found {len(articles)} articles", flush=True)

    return results


# ---------------------------------------------------------------------------
# Phase C: Clickstream download
# ---------------------------------------------------------------------------

async def download_clickstream(
    client: httpx.AsyncClient, lang: str, year: int, month: int, cache_dir: Path
) -> Path:
    """Download clickstream file if not cached. Returns local path."""
    filename = f"clickstream-{lang}wiki-{year:04d}-{month:02d}.tsv.gz"
    path = cache_dir / filename

    if path.exists():
        size_mb = path.stat().st_size // (1024 * 1024)
        print(f"  [{lang}] Using cached clickstream ({size_mb} MB): {filename}", flush=True)
        return path

    url = f"{CLICKSTREAM_BASE}/{year:04d}-{month:02d}/{filename}"
    print(f"  [{lang}] Downloading: {url}", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    with open(path, "wb") as f:
        async with client.stream("GET", url, headers=HEADERS, follow_redirects=True) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) == 0:
                    print(f"  [{lang}]   {downloaded // (1024 * 1024)} MB downloaded...", flush=True)

    print(f"  [{lang}] Download complete: {downloaded // (1024 * 1024)} MB", flush=True)
    return path


# ---------------------------------------------------------------------------
# Phase D: Clickstream parsing (synchronous — run in thread executor)
# ---------------------------------------------------------------------------

def lookup_article_nav(
    cache_path: Path, title_underscored: str, limit: int = 25
) -> dict:
    """Scan clickstream for a single article and return its navigation paths.

    Returns dicts of incoming links, outgoing links, and external arrivals,
    each sorted descending by count and capped at `limit`.
    """
    incoming: dict[str, int] = {}   # prev_article -> count (type=link, curr=title)
    outgoing: dict[str, int] = {}   # next_article -> count (type=link, prev=title)
    external: dict[str, int] = {}   # source tag -> count (type=external, curr=title)
    line_count = 0
    t0 = time.time()

    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 10_000_000 == 0:
                elapsed = time.time() - t0
                print(f"    {line_count // 1_000_000}M lines ({elapsed:.0f}s)...", flush=True)

            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            prev, curr, typ, n_str = parts
            try:
                n = int(n_str)
            except ValueError:
                continue

            if typ == "link":
                if curr == title_underscored:
                    incoming[prev] = incoming.get(prev, 0) + n
                elif prev == title_underscored:
                    outgoing[curr] = outgoing.get(curr, 0) + n
            elif typ == "external":
                if curr == title_underscored:
                    external[prev] = external.get(prev, 0) + n

    def top(d: dict, n: int) -> list[dict]:
        return [
            {"article": k.replace("_", " "), "count": v}
            for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
        ]

    return {
        "title": title_underscored.replace("_", " "),
        "incoming": top(incoming, limit),
        "outgoing": top(outgoing, limit),
        "external": top(external, limit),
        "total_incoming": sum(incoming.values()),
        "total_outgoing": sum(outgoing.values()),
        "total_external": sum(external.values()),
    }


def parse_clickstream(cache_path: Path, article_set: set[str]) -> dict[str, dict]:
    """Stream-parse a clickstream TSV.gz and return nav stats for articles in article_set.

    article_set must use underscores (matching the clickstream file format).
    This function is synchronous and safe to run in a thread executor.
    """
    nav: dict[str, dict] = defaultdict(lambda: {"in_link_n": 0, "out_link_n": 0, "search_n": 0})
    line_count = 0
    t0 = time.time()

    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if line_count % 10_000_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"    [{cache_path.name[:6]}] {line_count // 1_000_000}M lines "
                    f"({elapsed:.0f}s)...",
                    flush=True,
                )

            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue

            prev, curr, typ, n_str = parts
            try:
                n = int(n_str)
            except ValueError:
                continue

            prev_in = prev in article_set
            curr_in = curr in article_set

            if not (prev_in or curr_in):
                continue

            if typ == "link":
                if curr_in:
                    nav[curr]["in_link_n"] += n
                if prev_in:
                    nav[prev]["out_link_n"] += n
            elif typ == "external":
                if curr_in:
                    nav[curr]["search_n"] += n

    elapsed = time.time() - t0
    print(
        f"    [{cache_path.name[:6]}] Parsed {line_count // 1_000_000}M lines in {elapsed:.0f}s",
        flush=True,
    )
    return dict(nav)


# ---------------------------------------------------------------------------
# Phase E: Pageviews, quality, translation gap
# ---------------------------------------------------------------------------

async def _fetch_single_pageviews(
    client: httpx.AsyncClient, title: str, lang: str, start: str, end: str
) -> int:
    encoded = quote(title.replace(" ", "_"), safe="")
    url = (
        f"{PAGEVIEWS_BASE}/{lang}.wikipedia/all-access/user"
        f"/{encoded}/daily/{start.replace('-', '')}/{end.replace('-', '')}"
    )
    try:
        resp = await client.get(url, headers=HEADERS)
        if resp.status_code == 200:
            return sum(item.get("views", 0) for item in resp.json().get("items", []))
    except Exception:
        pass
    return 0


async def fetch_pageviews_all(
    client: httpx.AsyncClient, article_map: dict[str, dict]
) -> dict[str, int]:
    """Fetch 30-day pageviews for all articles. Returns {"lang:title": views}."""
    end = date.today()
    start = end - timedelta(days=29)
    start_str = start.isoformat()
    end_str = end.isoformat()

    pairs = [
        (lang, title)
        for lang, data in article_map.items()
        for title in data["titles"]
    ]
    views_map: dict[str, int] = {}
    total = len(pairs)

    for i in range(0, total, 50):
        batch = pairs[i : i + 50]
        tasks = [
            _fetch_single_pageviews(client, title, lang, start_str, end_str)
            for lang, title in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (lang, title), result in zip(batch, results):
            views_map[f"{lang}:{title}"] = result if isinstance(result, int) else 0

        if (i + 50) % 200 == 0:
            print(f"  Pageviews: {min(i + 50, total)}/{total}", flush=True)

    return views_map


async def _quality_batch(
    client: httpx.AsyncClient,
    api_url: str,
    titles: list[str],
    target_lang: str,
) -> dict[str, dict]:
    """Check quality + translation for up to 50 titles via MediaWiki API."""
    params = {
        "action": "query",
        "prop": "langlinks|categories|templates|pageimages",
        "titles": "|".join(titles),
        "lllang": target_lang,
        "lllimit": "500",
        "clshow": "hidden",
        "cllimit": "500",
        "tllimit": "500",
        "tlnamespace": "10",
        "pithumbsize": "100",
        "format": "json",
        "origin": "*",
    }
    for attempt in range(3):
        try:
            resp = await client.get(api_url, params=params, headers=HEADERS)
            if resp.status_code == 429:
                await asyncio.sleep(_RETRY_AFTER * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception:
            if attempt == 2:
                return {}
            await asyncio.sleep(1.0)
    else:
        return {}

    out: dict[str, dict] = {}
    for page_data in data.get("query", {}).get("pages", {}).values():
        title = page_data.get("title", "")
        if not title:
            continue

        cats = [c.get("title", "").lower() for c in page_data.get("categories", [])]
        is_stub = any("stub" in c or "esbozo" in c for c in cats)

        tpls = [t.get("title", "").lower() for t in page_data.get("templates", [])]
        no_refs = any(kw in t for t in tpls for kw in REF_TEMPLATES)

        has_image = (
            page_data.get("thumbnail") is not None
            or page_data.get("pageimage") is not None
        )

        has_translation = bool(page_data.get("langlinks"))

        out[title] = {
            "is_stub": is_stub,
            "no_refs": no_refs,
            "has_image": has_image,
            "has_translation": has_translation,
        }

    return out


async def fetch_quality_all(
    client: httpx.AsyncClient, article_map: dict[str, dict]
) -> dict[str, dict]:
    """Fetch quality + translation gap for all articles. Returns {"lang:title": quality}."""
    quality_map: dict[str, dict] = {}

    for lang, data in article_map.items():
        titles = list(data["titles"])
        api_url = API_URLS[lang]
        target_lang = PARTNER_LANG.get(lang, "es" if lang == "en" else "en")
        total = len(titles)

        for i in range(0, total, 50):
            batch = titles[i : i + 50]
            batch_result = await _quality_batch(client, api_url, batch, target_lang)
            for title, q in batch_result.items():
                quality_map[f"{lang}:{title}"] = q
            await asyncio.sleep(0.5)

            if (i + 50) % 200 == 0:
                print(f"  [{lang}] Quality: {min(i + 50, total)}/{total}", flush=True)

    return quality_map


# ---------------------------------------------------------------------------
# Phase F: Scoring, table rendering, CSV export
# ---------------------------------------------------------------------------

def assemble_and_score(
    article_map: dict,
    nav_data: dict,
    views_map: dict,
    quality_map: dict,
    use_clickstream: bool,
) -> list[dict]:
    """Assemble per-article records and compute priority scores. Returns sorted list."""
    records: list[dict] = []
    for lang, data in article_map.items():
        for title in data["titles"]:
            key = f"{lang}:{title}"
            underscore_title = title.replace(" ", "_")
            nav = nav_data[lang].get(underscore_title, {})
            q = quality_map.get(key, {})
            encoded = quote(title.replace(" ", "_"), safe="")
            records.append({
                "lang": lang,
                "title": title,
                "views_30d": views_map.get(key, 0),
                "in_link_n": nav.get("in_link_n", 0),
                "out_link_n": nav.get("out_link_n", 0),
                "search_arrivals": nav.get("search_n", 0),
                "centrality": (
                    nav.get("in_link_n", 0)
                    + nav.get("out_link_n", 0)
                    + nav.get("search_n", 0)
                ),
                "is_stub": q.get("is_stub", False),
                "no_refs": q.get("no_refs", False),
                "has_image": q.get("has_image", True),
                "has_translation": q.get("has_translation", True),
                "wiki_url": f"https://{lang}.wikipedia.org/wiki/{encoded}",
                "priority_score": 0.0,
            })

    max_views = max((r["views_30d"] for r in records), default=1)
    max_centrality = max((r["centrality"] for r in records), default=1)

    for r in records:
        r["priority_score"] = compute_priority(
            r["views_30d"],
            r["in_link_n"],
            r["out_link_n"],
            r["search_arrivals"],
            r["is_stub"],
            r["no_refs"],
            r["has_image"],
            r["has_translation"],
            max_views,
            max_centrality,
            use_clickstream,
        )

    records.sort(key=lambda r: r["priority_score"], reverse=True)
    return records

def compute_priority(
    views: int,
    in_link: int,
    out_link: int,
    search_n: int,
    is_stub: bool,
    no_refs: bool,
    has_image: bool,
    has_translation: bool,
    max_views: int,
    max_centrality: int,
    use_clickstream: bool,
) -> float:
    views_score = math.log10(max(views, 1)) / math.log10(max(max_views, 2))
    quality_score = (
        2.0 * is_stub
        + 1.0 * no_refs
        + 1.0 * (not has_image)
        + 1.5 * (not has_translation)
    ) / 5.5

    if use_clickstream:
        centrality_raw = in_link + out_link + search_n
        centrality_score = math.log10(max(centrality_raw, 1)) / math.log10(
            max(max_centrality, 2)
        )
        return round(
            0.40 * views_score + 0.30 * centrality_score + 0.30 * quality_score, 4
        )
    else:
        return round(0.57 * views_score + 0.43 * quality_score, 4)


def _human_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def render_table(records: list[dict], lang: str, top_n: int):
    lang_records = sorted(
        [r for r in records if r["lang"] == lang],
        key=lambda r: r["priority_score"],
        reverse=True,
    )[:top_n]

    width = 122
    lang_name = {"en": "English", "es": "Spanish"}.get(lang, lang.upper())

    print(f"\n{'─' * width}")
    print(f"  TOP ARTICLES TO EDIT — {lang_name} Wikipedia")
    print(f"{'─' * width}")
    print(
        f"  {'#':>4}  {'Article':<42}  {'Views/30d':>10}  "
        f"{'In-links':>9}  {'Out-links':>9}  {'S R I T':>7}  {'Priority':>8}"
    )
    print(f"{'─' * width}")

    for i, r in enumerate(lang_records, 1):
        flags = (
            ("S" if r["is_stub"] else "·")
            + " "
            + ("R" if r["no_refs"] else "·")
            + " "
            + ("I" if not r["has_image"] else "·")
            + " "
            + ("T" if not r["has_translation"] else "·")
        )
        title = r["title"]
        if len(title) > 43:
            title = title[:42] + "…"
        print(
            f"  {i:>4}  {title:<42}  {_human_num(r['views_30d']):>10}  "
            f"{_human_num(r['in_link_n']):>9}  {_human_num(r['out_link_n']):>9}  "
            f"{flags:>7}  {r['priority_score']:>8.4f}"
        )

    print(f"{'─' * width}")
    print(f"  Flags: S=Stub  R=Needs refs  I=No image  T=No translation to partner language")


def export_csv(records: list[dict], output_path: str):
    fieldnames = [
        "lang", "title", "views_30d", "in_link_n", "out_link_n", "search_arrivals",
        "centrality", "is_stub", "no_refs", "has_image", "has_translation",
        "priority_score", "wiki_url",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda r: r["priority_score"], reverse=True))
    print(f"\n  Exported {len(records)} rows → {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace):
    langs = [l.strip() for l in args.lang.split(",")]
    use_clickstream = not args.no_clickstream
    cache_dir = Path(args.cache_dir).expanduser()

    # Build category map from defaults + any --categories overrides
    categories: dict[str, list[str]] = {
        l: list(DEFAULT_CATEGORIES[l]) for l in langs if l in DEFAULT_CATEGORIES
    }
    if args.categories:
        for entry in args.categories.split(","):
            entry = entry.strip()
            if ":" in entry:
                lang_prefix, cat = entry.split(":", 1)
                categories.setdefault(lang_prefix, []).append(cat)
            else:
                for l in langs:
                    categories.setdefault(l, []).append(entry)

    t_start = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:

        # A. Resolve clickstream month
        year, month = None, None
        if use_clickstream:
            print("\nResolving latest clickstream month...", flush=True)
            year, month = await resolve_clickstream_month(client, forced=args.month)
            print(f"  Using: {year:04d}-{month:02d}", flush=True)

        # B. Discover articles
        print("\nDiscovering articles...", flush=True)
        article_map = await discover_articles(client, langs, categories, max_depth=args.depth)

        total_articles = sum(len(d["titles"]) for d in article_map.values())
        print(f"  Total: {total_articles} articles across {len(langs)} language(s)", flush=True)

        # C. Download clickstream files (concurrently for all langs)
        nav_data: dict[str, dict] = {lang: {} for lang in langs}
        if use_clickstream:
            print("\nDownloading clickstream files...", flush=True)
            paths = await asyncio.gather(*[
                download_clickstream(client, lang, year, month, cache_dir)
                for lang in langs
            ])

            # D. Parse clickstream files (concurrently in thread executor)
            print("\nParsing clickstream (EN may take 2–3 min)...", flush=True)
            loop = asyncio.get_event_loop()
            parsed_results = await asyncio.gather(*[
                loop.run_in_executor(
                    None, parse_clickstream, path, article_map[lang]["underscored"]
                )
                for lang, path in zip(langs, paths)
            ])
            for lang, nav in zip(langs, parsed_results):
                nav_data[lang] = nav

        # E. Pageviews + quality
        print("\nFetching pageviews (30 days)...", flush=True)
        views_map = await fetch_pageviews_all(client, article_map)
        print(f"  Done: {len(views_map)} articles", flush=True)

        print("Fetching article quality + translation gaps...", flush=True)
        quality_map = await fetch_quality_all(client, article_map)
        print(f"  Done: {len(quality_map)} articles", flush=True)

    # F. Output
    records = assemble_and_score(article_map, nav_data, views_map, quality_map, use_clickstream)
    for lang in langs:
        render_table(records, lang, args.top)

    if args.output:
        export_csv(records, args.output)

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.0f}s\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Wikipedia navigation patterns around FIFA World Cup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  Fast mode — views + quality only (~2 min, no download):
    python wiki_wc_nav.py --no-clickstream --top 10

  Full analysis with CSV export:
    python wiki_wc_nav.py --top 30 --output results.csv

  Spanish only, fast mode:
    python wiki_wc_nav.py --lang es --no-clickstream

  Force a specific cached month:
    python wiki_wc_nav.py --month 2026-03

  Custom categories (English only):
    python wiki_wc_nav.py --lang en --categories "en:FIFA_World_Cup_players,en:FIFA_World_Cup_managers"
        """,
    )
    parser.add_argument(
        "--lang", default="en,es",
        help="Comma-separated languages to analyze (default: en,es)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Top N articles to display per language (default: 20)",
    )
    parser.add_argument(
        "--output", metavar="FILE.csv",
        help="Export full ranked results to CSV",
    )
    parser.add_argument(
        "--no-clickstream", action="store_true",
        help="Skip clickstream download; use views + quality only (fast mode)",
    )
    parser.add_argument(
        "--depth", type=int, default=2, choices=[1, 2, 3],
        help="Subcategory crawl depth (default: 2)",
    )
    parser.add_argument(
        "--month", metavar="YYYY-MM",
        help="Force a specific clickstream month (default: auto-detect latest)",
    )
    parser.add_argument(
        "--cache-dir", default="~/.cache/wiki-wc-nav",
        help="Cache directory for clickstream files (default: ~/.cache/wiki-wc-nav)",
    )
    parser.add_argument(
        "--categories", metavar="CAT,...",
        help=(
            "Custom categories, comma-separated. Prefix with lang: to target a "
            "language, e.g. 'en:FIFA_World_Cup_hosts,es:Copa_del_Mundo'"
        ),
    )

    args = parser.parse_args()
    print("Wikipedia FIFA World Cup Navigation Analyzer")
    print("=" * 50)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
