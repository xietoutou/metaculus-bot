"""Free research via Google News RSS (no AskNews / search-API key needed).

Flow: a small LLM turns the question into a few search queries, we pull recent
headlines for each from Google News RSS, then a stronger LLM writes the research
brief from those headlines. The headline list is appended so the forecaster can
see the raw evidence (and dates) itself.
"""

import asyncio
import email.utils
import html
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote_plus

import requests
from forecasting_tools import GeneralLlm, MetaculusQuestion

logger = logging.getLogger(__name__)

RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
MAX_PER_QUERY = 8
MAX_HEADLINES = 24


def _fetch_rss(query: str) -> list[dict]:
    url = RSS_URL.format(q=quote_plus(f"{query} when:60d"))
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:  # network / parse problems shouldn't kill the forecast
        logger.warning(f"Google News fetch failed for {query!r}: {e}")
        return []
    items = []
    for it in list(root.iter("item"))[:MAX_PER_QUERY]:
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        pub = it.findtext("pubDate") or ""
        try:
            dt = email.utils.parsedate_to_datetime(pub).astimezone(timezone.utc)
        except Exception:
            dt = None
        desc = re.sub(r"<[^>]+>", " ", html.unescape(it.findtext("description") or ""))
        desc = re.sub(r"\s+", " ", desc).strip()
        items.append({"title": title, "source": source, "dt": dt, "desc": desc})
    return items


async def _make_queries(question: MetaculusQuestion, llm: GeneralLlm) -> list[str]:
    prompt = (
        "Write 3 short Google News search queries (2-6 words each) that would find the most "
        "recent news relevant to forecasting this question. One query per line, no numbering, "
        "no quotes, nothing else.\n\n"
        f"Question: {question.question_text}"
    )
    try:
        out = await llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
        out = ""
    queries = [re.sub(r'^[\s\-\d.*"]+|["\s]+$', "", line) for line in out.splitlines()]
    queries = [q for q in queries if 2 <= len(q) <= 80][:3]
    if not queries:  # fall back to the question title itself
        queries = [question.question_text[:80]]
    return queries


async def google_news_research(
    question: MetaculusQuestion, query_llm: GeneralLlm, writer_llm: GeneralLlm
) -> str:
    queries = await _make_queries(question, query_llm)
    batches = await asyncio.gather(*(asyncio.to_thread(_fetch_rss, q) for q in queries))

    seen, headlines = set(), []
    for batch in batches:
        for h in batch:
            key = h["title"].lower()[:90]
            if key and key not in seen:
                seen.add(key)
                headlines.append(h)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    headlines.sort(key=lambda h: h["dt"] or epoch, reverse=True)
    headlines = headlines[:MAX_HEADLINES]

    if not headlines:
        news_block = "(No recent news found on Google News for these queries.)"
    else:
        news_block = "\n".join(
            f"- [{h['dt'].strftime('%Y-%m-%d') if h['dt'] else 'undated'}] {h['title']}"
            + (f" ({h['source']})" if h["source"] and h["source"] not in h["title"] else "")
            for h in headlines
        )

    prompt = (
        "You are an assistant to a superforecaster. Using ONLY the news headlines below plus "
        "general background knowledge you are confident about, write a concise but detailed "
        "rundown of the most relevant facts for the question, noting dates, and whether the "
        "question currently looks likely to resolve Yes or No based on current information. "
        "Flag clearly when the headlines are thin or do not address the question. "
        "Do not produce a forecast yourself.\n\n"
        f"Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.\n\n"
        f"Question:\n{question.question_text}\n\n"
        f"Resolution criteria:\n{question.resolution_criteria}\n\n{question.fine_print}\n\n"
        f"Search queries used: {', '.join(queries)}\n\n"
        f"Recent headlines (newest first):\n{news_block}"
    )
    try:
        brief = await writer_llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"Research summary failed: {e}")
        brief = "(Research summary unavailable.)"
    return f"{brief}\n\nHeadlines used:\n{news_block}"
