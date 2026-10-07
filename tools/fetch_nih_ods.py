"""Download NIH Office of Dietary Supplements consumer fact sheets (vitamins, minerals, common
supplements) into the knowledge library. NIH content is public domain.

    uv run python tools/fetch_nih_ods.py
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sourcelib as sl  # noqa: E402

API = "https://ods.od.nih.gov/api/?resourcename={name}&readinglevel=Consumer&outputformat=HTML"
SOURCE = sl.ALLOWED_HOSTS["ods.od.nih.gov"]
RESOURCES = {
    "VitaminD": "vitamin D", "VitaminC": "vitamin C", "VitaminB12": "vitamin B12", "VitaminB6": "vitamin B6",
    "VitaminA": "vitamin A", "VitaminE": "vitamin E", "VitaminK": "vitamin K", "Folate": "folate folic acid",
    "Calcium": "calcium", "Iron": "iron", "Magnesium": "magnesium", "Zinc": "zinc", "Potassium": "potassium",
    "Selenium": "selenium", "Iodine": "iodine", "Biotin": "biotin", "Niacin": "niacin", "Thiamin": "thiamin",
    "Riboflavin": "riboflavin", "Choline": "choline", "Omega3FattyAcids": "omega-3 fish oil",
    "Probiotics": "probiotics", "MVMS": "multivitamin mineral supplements", "Ashwagandha": "ashwagandha",
    "Echinacea": "echinacea", "Ginkgo": "ginkgo", "Turmeric": "turmeric curcumin", "Melatonin": "melatonin",
    "Creatine": "creatine", "Chromium": "chromium", "Copper": "copper", "Phosphorus": "phosphorus",
    "Sodium": "sodium salt", "Fiber": "fiber", "Coenzyme Q10": "coq10 ubiquinone", "Glucosamine": "glucosamine",
    "Elderberry": "elderberry", "Cranberry": "cranberry", "Garlic": "garlic", "Ginger": "ginger",
    "Kava": "kava", "StJohnsWort": "st john's wort", "Valerian": "valerian", "Saw Palmetto": "saw palmetto",
}


def main() -> None:
    entries: list[str] = []
    done, missing = [], []
    for name, label in RESOURCES.items():
        url = API.format(name=urllib.parse.quote(name))
        try:
            page = sl.fetch(url, pause=0.3).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            missing.append(name)
            print(f"{name}: failed ({exc})")
            continue
        if "<ErrorOccurred>true" in page or "factsheet-toc" not in page and "<h1>" not in page:
            missing.append(name)
            print(f"{name}: no fact sheet")
            continue
        title = re.search(r"<h1>(.*?)</h1>", page, re.S)
        title = sl.html_to_text(title.group(1)) if title else name
        canonical = re.search(r'rel="canonical" href="([^"]+)"', page)
        page_url = (canonical.group(1) if canonical else f"https://ods.od.nih.gov/factsheets/{name}-Consumer/").replace(":443", "")
        body = page.split('id="fact-sheet"', 1)[-1]
        body = re.sub(r"<nav.*?</nav>", "", body, flags=re.S)
        body = re.sub(r"<(script|style)\b.*?</\1>", "", body, flags=re.S)
        sections = re.split(r"(?=<h2\b)", body)
        base = re.sub(r":? Fact Sheet for Consumers", "", title).strip()
        count = 0
        for section in sections:
            head = re.match(r"<h2[^>]*>(.*?)</h2>", section, re.S)
            if not head:
                continue
            heading = sl.html_to_text(head.group(1)).strip()
            text = sl.html_to_text(section[head.end():])
            if len(text) < 80 or heading.lower().startswith(("table of contents", "disclaimer", "acknowledg", "for more information", "references")):
                continue
            pieces = sl.passages(text)
            for number, piece in enumerate(pieces, start=1):
                part = f" (part {number} of {len(pieces)})" if len(pieces) > 1 else ""
                entries.append(
                    sl.entry(f"{base}: {heading}{part}", f"{SOURCE}, {base} fact sheet", page_url, sl.today(),
                             [base, label, heading], piece)
                )
                count += 1
        done.append(name)
        print(f"{name}: {count} passages")
    header = "NIH Office of Dietary Supplements consumer fact sheets. Public domain. Text verbatim apart from HTML removal."
    path = sl.write_topic_file("nih_ods.md", header, entries)
    sl.update_manifest("nih_ods", {"authority": "NIH Office of Dietary Supplements", "retrieved": sl.today(),
                                   "fact_sheets": done, "no_fact_sheet": missing, "passages": len(entries)})
    print(f"{len(done)} fact sheets, {len(entries)} passages -> {path}; missing: {missing}")


if __name__ == "__main__":
    main()
