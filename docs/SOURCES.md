# Where Baymax's knowledge comes from, and what "vetted" means here

## The policy

Baymax answers from retrieved passages, and each passage has a tier:

| Tier | What it is | How it's treated |
|---|---|---|
| **Official** | Text published by a US government health authority, copied verbatim | Preferred in ranking, cited by name ("according to the FDA label") |
| **Baymax summary** | Short summaries of published trials, written in Baymax's words | Labelled `Baymax summary of published research` |
| **Baymax notes** | Reference notes written from memory of guidelines | Labelled `Baymax notes, not clinically reviewed`, ranked below official sources |
| **Yours** | Files you put in `~/.baymax/library/` | Cited by file name and page |

## Official sources (the allowlist)

| Source | Authority | What we take | Licence |
|---|---|---|---|
| [MedlinePlus Health Topics](https://medlineplus.gov/xml.html) | National Library of Medicine, NIH | ~1,000 English topic summaries on conditions, tests, and procedures | Public domain |
| [FDA drug labels](https://open.fda.gov/apis/drug/label/) via DailyMed | US Food and Drug Administration | Uses, dosing, contraindications, boxed warnings, interactions, warnings, adverse reactions, pregnancy, overdose for ~220 common medicines | Public domain |
| [NIH Office of Dietary Supplements](https://ods.od.nih.gov/api/) | NIH | Consumer fact sheets for 27 vitamins, minerals, and supplements | Public domain |

The fetch code refuses any other host (`tools/sourcelib.py`, `ALLOWED_HOSTS`). Adding a source means editing that list on purpose.

## What is enforced, not just promised

- **Verbatim.** Official text is not paraphrased or summarised. The only changes are HTML removal and whitespace. A passage can't drift from what the authority published.
- **Traceable.** Every passage records its authority, a URL on that authority's own domain, and the date retrieved. FDA passages also record the exact label set id, manufacturer, and effective date. `baymax/knowledge/vetted/MANIFEST.json` records a SHA-256 of each downloaded file.
- **Audited.** `uv run python tools/vet.py` re-checks every entry: allowlisted source, matching URL host, retrieval date, no leftover markup, a manifest record. It exits non-zero on any failure.
- **Refreshable.** Re-run the fetchers (`tools/fetch_medlineplus.py`, `fetch_fda_labels.py`, `fetch_nih_ods.py`) to pull newer text; labels and guidance change.

## What "vetted" does not mean

- **No clinician has reviewed this library.** The sources are institutionally vetted, meaning they're published by the agencies responsible for them, but nobody has gone through each passage for accuracy or fitness for a particular use.
- **Labels are written for regulators and clinicians.** An FDA label is authoritative about what's approved, but it is dense, can describe rare risks at length, and is not personalised advice. The persona is told to explain it plainly and without alarm.
- **Labels are a snapshot.** One current label per drug is used (the most complete single-ingredient one from openFDA). Different manufacturers' labels for the same drug can differ slightly.
- **openFDA states that its data is unvalidated** and not to be used alone for care decisions. Baymax is for general information, not diagnosis or treatment.
- **Coverage is partial.** ~220 medicines, ~1,000 topics, 27 supplements. Anything outside returns nothing from the library, and Baymax is told to say so instead of guessing.

## Not yet included

CDC and USPSTF guideline text, ACC/AHA and ADA guidelines (copyrighted, not freely reusable), NICE, UpToDate, and PubMed abstracts. Guideline PDFs you obtain legitimately can be dropped into `~/.baymax/library/`.
