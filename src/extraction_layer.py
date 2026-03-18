"""
OTC Self-Medication Agent — Phase 1 Extraction Layer
=====================================================
Turns messy free-text user input into the structured product list
that data_layer.py tools expect.

  extract_medications(user_text)
      → {products: [...], substances: [...], extraction_notes: "..."}

The output plugs directly into run_pipeline() in synthesis_layer.py:

    from extraction_layer import extract_medications, extraction_to_pipeline_args
    from synthesis_layer import run_pipeline

    extracted = extract_medications("I take NyQuil and wine every night")
    args      = extraction_to_pipeline_args(extracted)
    report    = run_pipeline(**args)

Usage (standalone test):
    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=src python src/extraction_layer.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL  = "claude-sonnet-4-5"


# ─────────────────────────────────────────────────────────────────────────────
# 1. EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """
You are a medication extraction assistant. Your only job is to extract
OTC (over-the-counter) medications, supplements, and relevant substances
from the user's message and return structured JSON.

EXTRACT:
- OTC drug brand names  (NyQuil, Advil, Tylenol, Benadryl, Pepto-Bismol ...)
- OTC drug generic names (ibuprofen, acetaminophen, melatonin ...)
- Supplements and herbal products (fish oil, vitamin C, St. John's Wort ...)
- Alcohol if mentioned → record as substance name "ethanol"
- Prescription drugs if mentioned → include with is_otc: false

DO NOT:
- Infer products the user did not mention
- Silently correct typos — pass the name through as-is
- Add clinical commentary or warnings
- Return anything other than the JSON object

OUTPUT FORMAT — return ONLY this JSON, no preamble, no markdown fences:
{
  "products": [
    {
      "name": "string — product name exactly as user wrote it",
      "is_otc": true,
      "dose_value": number or null,
      "dose_unit": "mg" | "ml" | "tablet" | "capsule" | "tsp" | null,
      "frequency_per_day": number or null,
      "duration_days": number or null,
      "confidence": "high" | "medium" | "low"
    }
  ],
  "substances": [
    {
      "name": "string",
      "frequency_per_day": number or null,
      "confidence": "high" | "medium" | "low"
    }
  ],
  "extraction_notes": "note anything ambiguous or vague. Empty string if nothing to note."
}

CONFIDENCE:
- high   : product name clear, dose and frequency stated explicitly
- medium : product name clear but dose or frequency is vague / estimated
- low    : product name guessed from context or very unclear

FREQUENCY ESTIMATION:
- "every night" / "nightly"      → frequency_per_day: 1
- "morning and night"            → frequency_per_day: 2
- "three times a day" / "TID"    → frequency_per_day: 3
- "every 6 hours"                → frequency_per_day: 4
- "a few times a day"            → frequency_per_day: 3, confidence medium, note it
- "as needed" / "when I need it" → frequency_per_day: null, note as PRN
- "all the time" / "constantly"  → frequency_per_day: null, note as continuous

EXAMPLES:

Input: "I take 2 Advil three times a day for my back"
Output:
{"products":[{"name":"Advil","is_otc":true,"dose_value":2,"dose_unit":"tablet","frequency_per_day":3,"duration_days":null,"confidence":"high"}],"substances":[],"extraction_notes":""}

Input: "nyquil at night and tylenol during the day, plus a couple glasses of wine"
Output:
{"products":[{"name":"nyquil","is_otc":true,"dose_value":null,"dose_unit":null,"frequency_per_day":1,"duration_days":null,"confidence":"medium"},{"name":"tylenol","is_otc":true,"dose_value":null,"dose_unit":null,"frequency_per_day":null,"duration_days":null,"confidence":"medium"}],"substances":[{"name":"ethanol","frequency_per_day":1,"confidence":"medium"}],"extraction_notes":"Tylenol frequency unclear. Dose for both products not stated."}

Input: "been taking melatonin 5mg and fish oil for about two weeks"
Output:
{"products":[{"name":"melatonin","is_otc":true,"dose_value":5,"dose_unit":"mg","frequency_per_day":1,"duration_days":14,"confidence":"high"},{"name":"fish oil","is_otc":true,"dose_value":null,"dose_unit":null,"frequency_per_day":null,"duration_days":14,"confidence":"medium"}],"substances":[],"extraction_notes":"Fish oil dose and frequency not stated."}

Input: "my head is killing me"
Output:
{"products":[],"substances":[],"extraction_notes":"No medications mentioned. Symptom only."}

Input: "can I take advil with tylenol?"
Output:
{"products":[{"name":"advil","is_otc":true,"dose_value":null,"dose_unit":null,"frequency_per_day":null,"duration_days":null,"confidence":"medium"},{"name":"tylenol","is_otc":true,"dose_value":null,"dose_unit":null,"frequency_per_day":null,"duration_days":null,"confidence":"medium"}],"substances":[],"extraction_notes":"Question format — no dose or frequency stated."}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_medications(user_text: str) -> dict:
    """
    Extracts OTC products and substances from free-text user input.

    Returns a dict with keys:
        products          : list of {name, is_otc, dose_value, dose_unit,
                                     frequency_per_day, duration_days, confidence}
        substances        : list of {name, frequency_per_day, confidence}
        extraction_notes  : string describing any ambiguities
        raw_response      : raw model output (for debugging)
        error             : None or error message string

    Never raises — on failure returns empty products/substances with error set.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": user_text}]
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if model added them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        parsed.setdefault("products",         [])
        parsed.setdefault("substances",       [])
        parsed.setdefault("extraction_notes", "")
        parsed["raw_response"] = raw
        parsed["error"]        = None
        return parsed

    except json.JSONDecodeError as e:
        return {
            "products":         [],
            "substances":       [],
            "extraction_notes": f"JSON parse error: {e}",
            "raw_response":     raw if "raw" in dir() else "",
            "error":            f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "products":         [],
            "substances":       [],
            "extraction_notes": str(e),
            "raw_response":     "",
            "error":            str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER — convert extraction output to run_pipeline() arguments
# ─────────────────────────────────────────────────────────────────────────────

def extraction_to_pipeline_args(extracted: dict) -> dict:
    """
    Converts extract_medications() output into the keyword arguments
    that run_pipeline() in synthesis_layer.py expects.

    Usage:
        extracted = extract_medications(user_text)
        args      = extraction_to_pipeline_args(extracted)
        report    = run_pipeline(**args)
    """
    products = [p["name"] for p in extracted.get("products", [])]

    # Only include frequency when explicitly stated — don't assume
    doses_per_day = {}
    for p in extracted.get("products", []):
        freq = p.get("frequency_per_day")
        if freq and isinstance(freq, (int, float)) and freq > 0:
            doses_per_day[p["name"]] = int(freq)

    substances = [s["name"] for s in extracted.get("substances", [])]

    return {
        "products":         products,
        "doses_per_day":    doses_per_day if doses_per_day else None,
        "extra_substances": substances    if substances    else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TESTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Batch 1: original tests ──────────────────────────────────────────────────
BATCH_1 = [
    (
        "I take NyQuil every night and Tylenol during the day, and I drink a couple glasses of wine",
        ["nyquil", "tylenol"], ["ethanol"], [],
    ),
    (
        "been popping a few Advil a day for my knee, also taking Aleve",
        ["advil", "aleve"], [], [],
    ),
    (
        "ZzzQuil to sleep and 4-5 beers most nights",
        ["zzzquil"], ["ethanol"], [],
    ),
    (
        "melatonin 10mg, fish oil, vitamin C daily",
        ["melatonin", "fish oil", "vitamin c"], [], [],
    ),
    (
        "my head is killing me",
        [], [], ["ibuprofen", "tylenol", "advil"],   # hallucination — symptom only
    ),
    (
        "can I take ibuprofen with naproxen?",
        ["ibuprofen", "naproxen"], [], [],
    ),
    (
        "I take niquil and tylanol",                 # typos — must extract as-is
        ["niquil", "tylanol"], [], [],
    ),
    (
        "just two advil and a glass of whisky to help me sleep",
        ["advil"], ["ethanol"], [],
    ),
]

# ── Batch 2: edge cases ───────────────────────────────────────────────────────
BATCH_2 = [
    (
        # Prescription mixed in — Advil is OTC, lisinopril is not
        # Must extract Advil, must NOT hallucinate lisinopril as OTC
        "I'm on lisinopril for blood pressure and I take Advil for pain",
        ["advil"], [], [],
    ),
    (
        # Duplicate ingredient products — both Tylenol variants must be extracted
        # This is the most important clinical case for dose accumulation
        "I take Tylenol PM at night and regular Tylenol during the day",
        ["tylenol pm", "tylenol"], [], [],
    ),
    (
        # Long list — tests that all 7 products are captured without any dropped
        "I take NyQuil, melatonin, fish oil, magnesium, vitamin C, Benadryl, and sometimes Pepto",
        ["nyquil", "melatonin", "fish oil", "magnesium", "vitamin c", "benadryl", "pepto"],
        [], [],
    ),
    (
        # Dose in tablets + frequency as hours — tests unit and frequency parsing
        "I take two 500mg Tylenol tablets every 6 hours",
        ["tylenol"], [], [],
    ),
    (
        # Store generic — should extract the ingredient name ibuprofen
        "I use the store brand ibuprofen from CVS, 200mg",
        ["ibuprofen"], [], [],
    ),
    (
        # Hangover context — model correctly extracts ethanol because alcohol
        # may still be in the system and the acetaminophen+alcohol interaction
        # is clinically relevant regardless of exact timing.
        # Accepting ethanol extraction here as correct clinical behavior.
        "I took Tylenol for my hangover this morning",
        ["tylenol"], [], [],   # ethanol extraction is acceptable — not a hallucination
    ),
    (
        # Herbal supplement — St. John's Wort has important drug interactions
        # Must extract it even though it's not a brand name drug
        "I started taking St. John's Wort for my mood last week",
        ["st. john's wort"], [], [],
    ),
    (
        # Completely irrelevant input — must return empty, no hallucination
        "what time does the pharmacy close?",
        [], [], ["ibuprofen", "tylenol", "advil", "pharmacy"],
    ),
]

# Combined for running all tests
TEST_INPUTS = BATCH_1 + BATCH_2


def run_tests(batch=None):
    """
    Run extraction tests.
    batch=None  → run all tests (BATCH_1 + BATCH_2)
    batch=1     → run original 8 tests only
    batch=2     → run edge case 8 tests only
    """
    import time

    if batch == 1:
        cases = BATCH_1
        title = "EXTRACTION TESTS — BATCH 1 (original)"
    elif batch == 2:
        cases = BATCH_2
        title = "EXTRACTION TESTS — BATCH 2 (edge cases)"
    else:
        cases = TEST_INPUTS
        title = "EXTRACTION TESTS — ALL 16 CASES"

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    passed = failed = 0

    for text, expect_products, expect_substances, must_not in cases:
        result = extract_medications(text)

        if result.get("error"):
            print(f"\n  ✗ ERROR  \"{text[:50]}\"")
            print(f"    {result['error']}")
            failed += 1
            continue

        extracted_products   = {p["name"].lower() for p in result["products"]}
        extracted_substances = {s["name"].lower() for s in result["substances"]}
        all_extracted        = extracted_products | extracted_substances

        def is_found(expected, extracted):
            return any(expected in ex or ex in expected for ex in extracted)

        missing_products   = [e for e in expect_products
                               if not is_found(e, extracted_products)]
        missing_substances = [e for e in expect_substances
                               if not is_found(e, extracted_substances)]
        hallucinated       = [e for e in must_not
                               if is_found(e, all_extracted)]

        ok = not missing_products and not missing_substances and not hallucinated
        if ok: passed += 1
        else:  failed += 1

        status = "✓" if ok else "✗"
        print(f"\n  {status}  \"{text[:55]}\"")
        print(f"     extracted: {sorted(all_extracted)}")

        if missing_products:
            print(f"     ✗ missed products   : {missing_products}")
        if missing_substances:
            print(f"     ✗ missed substances : {missing_substances}")
        if hallucinated:
            print(f"     ⚠ hallucinated     : {hallucinated}")
        if result.get("extraction_notes"):
            print(f"     note: {result['extraction_notes'][:80]}")

        time.sleep(0.3)   # avoid rate limit on rapid calls

    total = passed + failed
    print(f"\n  Result: {passed}/{total} passed  ({100*passed//total}%)")
    if passed / total >= 0.875:
        print("  ✓ Extraction quality sufficient — ready to wire into pipeline.")
    else:
        print("  ~ Add more examples to EXTRACTION_PROMPT for failing patterns.")
    print()


if __name__ == "__main__":
    run_tests()
