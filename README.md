# OTCSafe — OTC Self-Medication Risk Agent

A conversational AI agent that detects harmful over-the-counter self-medication behaviors and explains associated risks in plain language.

Type anything — "I take NyQuil and Tylenol every night with wine" — and the agent identifies dangerous drug combinations, duplicate ingredient exposure, dose limit exceedances, and drug-supplement interactions. Every finding is cited to its source.

**US medications only.** Built on RxNorm and OpenFDA, which cover drugs approved and sold in the United States.

---

## What it does

- Resolves brand names to active ingredients (NyQuil → acetaminophen + dextromethorphan + ...)
- Detects pairwise interactions via OpenFDA label search
- Reasons pharmacologically using Claude to catch interactions databases miss — serotonin syndrome, CYP450 enzyme interactions, stimulant stacking, CNS depression combinations, duplicate therapeutic classes
- Accumulates ingredient doses across multiple products and flags exceedances of safe daily limits
- Writes plain-language risk reports with severity ratings and actionable advice
- Cites every finding to a named source with a URL

---

## How to run it

### Option 1 — Terminal (always available, no coordination needed)

```bash
git clone https://github.com/yourusername/OTCAgent.git
cd OTCAgent

pip install anthropic requests flask flask-cors
export ANTHROPIC_API_KEY=your_key_here

# Interactive CLI — type anything, get a report
PYTHONPATH=src python src/run_agent.py
```

The CLI prints a full risk report for each query and saves a downloadable PDF-ready JSON to the project folder. No browser needed.

To get an API key: sign up at [console.anthropic.com](https://console.anthropic.com) — new accounts get $5 free credits, no credit card required.

### Option 2 — Browser UI (requires the server to be running)

The web interface requires the Flask server to be active. If you want to try the browser version, **message me to coordinate a time** and I'll spin it up for you.

```bash
PYTHONPATH=src python app.py
# Then open http://localhost:8080
```

---

## Project structure

```
OTCAgent/
├── src/
│   ├── data_layer.py         3 tools: ingredient resolution, interactions, dose accumulation
│   ├── extraction_layer.py   Turns free text into structured product lists
│   └── synthesis_layer.py    LLM reasoning + plain-language report generation
├── app.py                    Flask API server (port 8080)
├── index.html                Browser UI
├── tests/
│   └── test_data_layer_v2.py Integration tests for data layer tools
└── src/run_agent.py          Interactive CLI
```

---

## Data sources

| Source | What it provides | Cost |
|---|---|---|
| RxNorm (NLM) | Brand name → ingredient mapping | Free |
| OpenFDA | Drug labels, active ingredients, interaction text | Free |
| Claude API (Anthropic) | Extraction, pharmacological reasoning, synthesis | Pay-per-use |

---

## Example queries

```
I take Advil three times a day and Aleve twice a day for back pain
I take Theraflu twice a day, Tylenol 3 times a day, and drink wine every night
I took a Benadryl this morning, using Dramamine for a boat trip, ZzzQuil tonight
I'm on fluoxetine and started taking St. John's Wort for my mood
```

---

## Limitations

- US drug databases only — non-US brand names (Nurofen, Calpol, Panadol) may not resolve
- Supplement interaction data is sparse in OpenFDA — the LLM reasoning step partially compensates
- Phase 1 prototype — no session memory, no user accounts, no persistent history
- Not a substitute for professional medical advice

---

## Contact

Message me to coordinate access to the live browser demo, or to discuss the project.
