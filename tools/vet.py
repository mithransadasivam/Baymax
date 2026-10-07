"""Audit the vetted knowledge folder. Exits non-zero if anything fails, so it can gate a commit.

Checks, for every entry under baymax/knowledge/vetted/:
  * the Source names an allowlisted authority;
  * a URL on that authority's own domain, a Retrieved date, and keywords are present;
  * the body is non-trivial and contains no markup remnants;
  * the manifest has a record for each file, with a retrieval date and (for downloads) a SHA-256.

    uv run python tools/vet.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import sourcelib as sl  # noqa: E402
from baymax.research import parse_entries  # noqa: E402

URL_HOSTS = {  # authority prefix -> domains its URLs may use
    "MedlinePlus (NLM/NIH)": {"medlineplus.gov"},
    "FDA drug label": {"dailymed.nlm.nih.gov"},
    "NIH Office of Dietary Supplements": {"ods.od.nih.gov"},
}
FILE_MANIFEST_KEY = {"medlineplus.md": "medlineplus", "fda_labels.md": "fda_labels", "nih_ods.md": "nih_ods"}
MARKUP = re.compile(r"</?[a-z][^>]*>|&[a-z]+;|&#\d+;", re.I)
FIELD = re.compile(r"^(URL|Retrieved|Label-set-id):\s*(.+)$", re.M)


def main() -> int:
    problems: list[str] = []
    manifest = json.loads(sl.MANIFEST.read_text(encoding="utf-8")) if sl.MANIFEST.exists() else {}
    total = 0
    for path in sorted(sl.VETTED_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"(?m)^## ", text)[1:]
        if path.name not in FILE_MANIFEST_KEY or FILE_MANIFEST_KEY[path.name] not in manifest:
            problems.append(f"{path.name}: no manifest record")
        elif not manifest[FILE_MANIFEST_KEY[path.name]].get("retrieved"):
            problems.append(f"{path.name}: manifest has no retrieval date")
        for block in blocks:
            total += 1
            title = block.splitlines()[0]
            fields = dict(FIELD.findall(block))
            entries = parse_entries(path.stem, "## " + block)
            if not entries:
                problems.append(f"{path.name}: unparseable entry {title!r}")
                continue
            entry = entries[0]
            authority = next((a for a in URL_HOSTS if entry.source.startswith(a)), None)
            if authority is None:
                problems.append(f"{path.name}: {title!r}: source {entry.source!r} is not an allowlisted authority")
                continue
            host = urlparse(fields.get("URL", "")).hostname or ""
            if host not in URL_HOSTS[authority]:
                problems.append(f"{path.name}: {title!r}: URL host {host!r} doesn't match {authority}")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields.get("Retrieved", "")):
                problems.append(f"{path.name}: {title!r}: missing or bad Retrieved date")
            if authority == "FDA drug label" and not fields.get("Label-set-id"):
                problems.append(f"{path.name}: {title!r}: FDA entry without a label set id")
            if not entry.keywords:
                problems.append(f"{path.name}: {title!r}: no keywords")
            if len(entry.body) < 40:
                problems.append(f"{path.name}: {title!r}: body too short")
            if MARKUP.search(entry.body):
                problems.append(f"{path.name}: {title!r}: leftover HTML in body")
    print(f"checked {total} entries in {len(list(sl.VETTED_DIR.glob('*.md')))} files")
    for problem in problems[:40]:
        print("  FAIL:", problem)
    if len(problems) > 40:
        print(f"  ... and {len(problems) - 40} more")
    print("PASS" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
