"""Download MedlinePlus Health Topics (National Library of Medicine) into the knowledge library.

MedlinePlus is NLM's consumer health service; its topic summaries are public domain. The bulk XML
lists every English topic with a plain-language summary, alternate names, and MeSH headings.

    uv run python tools/fetch_medlineplus.py
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import sourcelib as sl  # noqa: E402

INDEX_URL = "https://medlineplus.gov/xml.html"
SOURCE = sl.ALLOWED_HOSTS["medlineplus.gov"]


def latest_xml_url() -> str:
    page = sl.fetch(INDEX_URL).decode("utf-8", "replace")
    dated = sorted(set(re.findall(r"https://medlineplus\.gov/xml/mplus_topics_(\d{4}-\d{2}-\d{2})\.xml", page)))
    if not dated:
        raise RuntimeError("couldn't find the current MedlinePlus topics file")
    return f"https://medlineplus.gov/xml/mplus_topics_{dated[-1]}.xml"


def main() -> None:
    url = latest_xml_url()
    print("downloading", url)
    raw = sl.fetch_cached(url, url.rsplit("/", 1)[1])
    root = ET.fromstring(raw)
    generated = root.get("date-generated", "")
    retrieved = sl.today()
    entries: list[str] = []
    topics = 0
    for topic in root.iter("health-topic"):
        if topic.get("language") != "English":
            continue
        title = topic.get("title", "").strip()
        page_url = topic.get("url", "")
        summary = topic.findtext("full-summary") or ""
        if not title or not summary.strip() or not page_url.startswith("https://medlineplus.gov/"):
            continue
        text = sl.html_to_text(summary)
        if len(text) < 80:
            continue
        keywords = [title, *(e.text or "" for e in topic.findall("also-called"))]
        keywords += [(m.findtext("descriptor") or "") for m in topic.findall("mesh-heading")]
        pieces = sl.passages(text)
        for number, piece in enumerate(pieces, start=1):
            part = f" (part {number} of {len(pieces)})" if len(pieces) > 1 else ""
            entries.append(
                sl.entry(f"{title}{part}", f"{SOURCE}, {title}", page_url, retrieved, keywords, piece)
            )
        topics += 1
    header = (
        "MedlinePlus Health Topics, National Library of Medicine. Public domain. "
        f"Source file generated {generated}. Text is verbatim from NLM apart from HTML removal."
    )
    path = sl.write_topic_file("medlineplus.md", header, entries)
    sl.update_manifest(
        "medlineplus",
        {
            "authority": "National Library of Medicine, NIH",
            "url": url,
            "retrieved": retrieved,
            "source_generated": generated,
            "sha256_of_download": sl.sha256(raw),
            "topics": topics,
            "passages": len(entries),
        },
    )
    print(f"{topics} topics, {len(entries)} passages -> {path}")


if __name__ == "__main__":
    main()
