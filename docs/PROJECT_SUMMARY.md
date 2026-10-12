# Baymax: project summary (for résumé use)

> Paste this file into Claude and ask it to write résumé bullets. Everything below is something that was actually built and measured. The "Honest caveats" section is there so claims stay accurate in an interview.

## One-line description

Baymax is a fully local, offline personal healthcare companion you can talk or type to, inspired by the character from *Big Hero 6*. It answers health questions from a retrieval-augmented library of official US government sources, and it escalates real emergencies with deterministic rules instead of trusting the language model.

## Why it exists

Search engines and chatbots tend to jump to the worst-case diagnosis, and cloud tools send health questions off the user's machine. The goals were:

1. Keep all health data on the user's own hardware (no cloud APIs, no accounts, no usage limits).
2. Answer calmly and proportionately, leading with the most likely ordinary cause instead of the scariest one.
3. Never leave a real emergency to a small model's judgment.

## What it does

- **Voice and text.** Push-to-talk, hands-free ("Hey Baymax"), or typed chat, in the terminal, a desktop window, or a phone.
- **Symptom guidance.** Asks one focused follow-up at a time, explains likely causes, safe home care, and the specific signs that mean seeing a doctor.
- **Medicine information.** Doses, daily maximums, side effects, and interactions, drawn from FDA drug labels and checked against the allergies and medicines the user has mentioned.
- **Lab and term explainer, first aid, supplements.**
- **Health profile memory.** Remembers allergies, medicines, and conditions between sessions in a plain-text file the user can read and edit.
- **Emergency escalation.** Chest pain, stroke signs, trouble breathing, anaphylaxis, severe bleeding, overdose, seizure, and suicidal thoughts get a fixed, vetted script (call 911, call or text 988, Poison Control) that never passes through the model.
- **Phone access.** A tap-to-talk web page for a phone, reachable on the home network or from anywhere over Tailscale with a real HTTPS certificate.

## Technical design

**Runs entirely on a consumer GPU.** The model is Llama 3.1 8B served by Ollama on an 8 GB AMD Radeon RX 580 through Vulkan. Speech recognition is faster-whisper, speech output is Piper, and wake-word detection and voice activity detection use openWakeWord and Silero. No data leaves the machine.

**Retrieval-augmented generation (RAG)**
- Sources: ~1,017 MedlinePlus health topics (National Library of Medicine), FDA drug labels for ~220 common medicines (via openFDA / DailyMed), and 27 NIH Office of Dietary Supplements fact sheets. About 6,000 passages in total, plus the user's own PDFs and text files.
- Hybrid search: dense embeddings (`nomic-embed-text` via Ollama) combined with BM25 keyword search, fused with weighted reciprocal rank fusion. BM25 catches exact drug names and lab values that embeddings blur.
- Targeted routing: when a question names two medicines, the system pulls each drug's label passages that mention the other drug or its class, which general similarity search missed.
- Relevance gating, so small talk retrieves nothing. Source-tier ranking, so official passages edge out the assistant's own notes. Follow-up questions are searched together with the question before them.
- The model is instructed to ground answers in the passages, cite the authority ("according to the FDA label"), and say so when the passages don't cover the question.

**Provenance and vetting (enforced in code, not just documented)**
- The fetcher refuses any host outside an allowlist of three US government authorities.
- Official text is stored verbatim (HTML and whitespace removed only), never paraphrased.
- Every passage records its authority, a URL on that authority's own domain, and the retrieval date. FDA passages also record the label set id, manufacturer, and effective date. A manifest records a SHA-256 of each download.
- A vetting script (`tools/vet.py`) audits every entry and exits non-zero on any failure. It caught a real parser bug during development, where body text beginning with "Source:" overwrote the citation.
- The assistant's own notes are labelled "not clinically reviewed" and ranked below official sources.

**Safety architecture.** Emergency detection is rule-based on purpose: a regex gate with negation handling ("I don't have chest pain"), past-tense and general-question suppression ("what causes a stroke?"), and no suppression for suicidal language. The gate runs inside the conversation class, so the desktop, terminal, and phone all get identical behavior, and in hands-free mode an emergency is acted on even without the wake phrase.

**Evaluation.** A retrieval benchmark (`tools/eval_retrieval.py`) of 35 medical questions with expected sources plus 7 non-medical "chat" negatives measures hit rate, mean reciprocal rank, and false retrievals. It was used to compare dense-only against hybrid search and to tune the keyword weight, and it exposed a data-coverage gap (long drug-interaction sections had been truncated) that led to re-ingesting labels with higher limits.

**Application and packaging.** Desktop window (pywebview, WebView2) with a custom-drawn animated face that blinks, glances while thinking, and pulses while speaking, and a red alert state for emergencies. A phone web page served by Bottle over HTTP and HTTPS with a generated certificate, and an automatic Tailscale HTTPS certificate when available. Windows launcher and desktop shortcut with a generated multi-size `.ico`. The logo is an SVG, with a script that renders the PNG and ICO versions.

## Stack

Python 3.12 · Ollama (Llama 3.1 8B, nomic-embed-text) · faster-whisper · Piper TTS · openWakeWord · NumPy (BM25 and vector search) · pywebview · Bottle · cryptography (TLS certificates) · Tailscale · openFDA, MedlinePlus, and NIH ODS APIs · pytest · uv · Git

## Numbers you can state

| Fact | Value |
|---|---|
| Automated tests | 234 passing |
| Knowledge passages indexed | ~6,000 (6,167 vetted entries plus the assistant's own notes) |
| FDA drug labels | ~220 medicines |
| MedlinePlus topics | 1,017 |
| NIH supplement fact sheets | 27 |
| Retrieval benchmark (35 questions) | 33 of 35 correct in the top 4, 0 of 7 false retrievals on small talk |
| Emergency categories covered | 8 (heart attack, stroke, breathing/choking, anaphylaxis, severe bleeding, overdose/poisoning, seizure/unresponsive, suicide/self-harm) |
| Cost to run | $0 (no cloud APIs) |

## Honest caveats (keep these true on a résumé)

- **No clinical review.** The sources are official, but no clinician has reviewed this system. It is a general-information tool, not a diagnostic or medical device, and the README says so.
- **Small benchmark.** The 35-question set was written by the developer, and the keyword weight was tuned on that same set, so the figures show relative improvement rather than a general accuracy claim. The 33/35 result was measured before the final FDA re-ingest; the re-run on the final data did not complete.
- **Small model.** An 8B local model reasons about medicine far less reliably than a frontier model. Retrieval and the emergency rules compensate but don't eliminate that.
- **US-only emergency numbers** (911, 988, Poison Control).
- **Lineage.** Baymax is built on the developer's earlier project, Legion, a local voice assistant. The speech pipeline, window shell, and phone/Tailscale server originate there. The medical layer is new: the emergency gate, persona, vetted knowledge pipeline, hybrid retrieval, evaluation, and the Baymax interface and logo. It was built with AI-assisted development (Claude Code).

## Suggested résumé bullets (edit to fit)

- Built **Baymax**, a fully offline voice and text healthcare assistant running Llama 3.1 8B on a consumer 8 GB GPU, with no cloud dependency, so health data never leaves the user's machine.
- Designed a **hybrid RAG pipeline** (dense embeddings + BM25 with weighted reciprocal rank fusion, relevance gating, and drug-pair routing) over ~6,000 passages from MedlinePlus, FDA drug labels, and NIH fact sheets; benchmarked at 33/35 top-4 retrieval accuracy with zero false retrievals on non-medical chat.
- Engineered a **provenance and vetting pipeline**: host allowlist, verbatim storage, per-passage source URL, retrieval date, and label set id, SHA-256 manifests, and an audit script that gates ingestion. It caught a citation-corruption bug before release.
- Implemented **deterministic emergency detection** (regex with negation, tense, and question-form handling) that bypasses the LLM and returns vetted 911/988/Poison Control scripts across desktop, terminal, and phone; covered by unit tests.
- Shipped a **cross-device product**: animated desktop UI, a mobile tap-to-talk web app with auto-provisioned HTTPS over Tailscale, a Windows launcher, and a generated logo and icon set; 234 automated tests.
