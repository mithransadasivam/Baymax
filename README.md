# Baymax

A personal healthcare companion that runs entirely on your own machine. Talk to it or type to it. No cloud APIs, no accounts, and no usage limits once the models are downloaded.

Built on the [Legion](../legion) voice stack: faster-whisper (speech to text), Ollama (the local model), and Piper (text to speech).

> **Baymax is not a doctor.** It gives general health information, not a diagnosis, and it can be wrong. For anything urgent call 911. For suicidal thoughts call or text 988.

## What it does

- **Symptom guidance.** Describe how you feel. It asks one focused follow-up at a time, then explains likely causes, what is safe to do at home, and the specific signs that mean you should see a doctor.
- **Medicine help.** Doses, daily maximums, side effects, and dangerous interactions, checked against the allergies and medicines you've told it about.
- **Lab and term explainer.** What a CBC, metabolic panel, A1c, TSH, or cholesterol result means and what to ask your doctor.
- **First aid.** Burns, bleeding, choking, CPR, snake bites, heat illness, head injuries.
- **Remembers your health profile.** Tell it "I'm allergic to penicillin" or "I take metformin" once. Notes live in plain text at `~/.baymax/memory.txt`, so you can read or edit them.
- **Your own medical library.** Drop PDFs, `.txt`, or `.md` files (guidelines, textbooks, your own records) into `~/.baymax/library/` and it learns from them, searching by meaning with local embeddings.

## Emergencies are never left to the model

A small local model can't be trusted to notice that "my left arm is numb and my chest is tight" means call an ambulance. [`baymax/emergency.py`](baymax/emergency.py) checks every message with plain text rules, and a match gets a fixed, vetted script (call 911, what to do while waiting) instead of anything the model improvises. It covers heart attack, stroke, can't breathe or choking, anaphylaxis, severe bleeding, overdose and poisoning, seizure or unresponsive, and suicidal thoughts (988). It leans toward escalating, understands negation ("I don't have chest pain"), and ignores general questions ("what causes a stroke?"). In the GUI's hands-free mode, an emergency is acted on even if you didn't say "Hey Baymax" first.

## Run it

Start Ollama (see the Legion notes), then:

```powershell
uv run baymax --text          # typed chat in the terminal
uv run baymax                 # push-to-talk voice (Enter to start and stop)
uv run baymax --gui           # Baymax window: say "Hey Baymax" or type
uv run baymax --gui --text    # window, typing only
uv run baymax --no-search     # fully offline, no web lookups
```

Sensible default model is `llama3.1:8b`. Change with `--model`. Everything can also be set with `BAYMAX_*` environment variables.

## Knowledge

[`baymax/knowledge/clinical.md`](baymax/knowledge/clinical.md) holds about 45 short reference notes (drug dosing and safety, interactions, lab reference ranges, red-flag symptoms, first aid, screening, vaccines), each written in Baymax's own words with its guideline source named. [`medicine.md`](baymax/knowledge/medicine.md) adds summaries of landmark clinical trials. Add more entries in the same format: a `## Title` heading, a `Source:` line, a `Keywords:` line, then a paragraph.

These notes are a starting point and have not been clinically reviewed. Reference ranges and guidelines change; check anything that matters against a current source or a pharmacist.

## Limits worth knowing

- A local 8B model reasons about medicine far less reliably than a frontier model. Retrieval and the emergency rules compensate, but treat answers as a well-informed friend's, not a clinician's.
- It can't examine you, order tests, call anyone, or book appointments.
- Emergency numbers are for the United States.
