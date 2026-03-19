"""
OTC Self-Medication Agent — Phase 1 Synthesis Layer
=====================================================
This module contains:

  1. SYNTHESIS_PROMPT  — system prompt that turns tool output into a
                         structured plain-language risk report

  2. synthesize_risk_report()  — calls Claude API with tool outputs,
                                 returns structured report dict

  3. run_pipeline()  — full end-to-end: takes a product list, runs all
                       three data layer tools, calls synthesis, returns
                       the final report ready to show a user

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python synthesis_layer.py

    Or import and call directly:
    from synthesis_layer import run_pipeline
    report = run_pipeline(["NyQuil", "Tylenol"], extra_substances=["ethanol"])
    print(report["output"]["risk_summary"])
"""

import os
import json
import anthropic
from data_layer import (
    resolve_brand_to_ingredients,
    get_interactions,
    check_dose_accumulation,
    INGREDIENT_ALIASES,
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL  = "claude-sonnet-4-5"


# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHESIS PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """
You are a pharmacist assistant that explains OTC medication risks to the
general public in plain, clear language. You are given structured data from
three analysis tools and must write a risk report the user can act on.

You ALWAYS output a JSON object with exactly these five keys:

{
  "risk_level": "high" | "moderate" | "low" | "none",
  "risk_summary": "1-2 sentence plain-language headline. State the most
                   important risk first. If no risks, say so clearly.",
  "details": [
    {
      "finding": "one specific risk or concern",
      "explanation": "plain-language explanation of why this matters,
                      what could happen, and how serious it is",
      "source": "source name — e.g. OpenFDA label, DrugBank, FDA monograph"
    }
  ],
  "what_to_do": "Concrete, actionable advice. What should the person
                 actually do differently? Be specific — name products,
                 suggest alternatives, state maximum doses where relevant.
                 Do not be vague. Do not use bullet points — write prose.",
  "see_pharmacist": true | false
}

RULES:

1. risk_level is the HIGHEST severity across all findings:
   - high     = any interaction severity:high OR any ingredient exceeds safe daily limit
   - moderate = any interaction severity:moderate and no dose exceedance
   - low      = only low/unknown severity interactions
   - none     = no interactions, no dose issues found

2. see_pharmacist = true when risk_level is "high" OR any dose exceeds its limit.
   Always true when ethanol (alcohol) is present with any other finding.

3. Every item in details[] must name its source. Do not invent sources.
   Use exactly the source names provided in the tool data.

4. Write for a general audience. No medical jargon without explanation.
   "Hepatotoxicity" → "liver damage". "CNS depression" → "extreme drowsiness
   and slowed breathing". "NSAID" → "anti-inflammatory painkiller".

5. If the same ingredient appears in multiple products, always name both
   products. For example: "Both NyQuil and Tylenol contain acetaminophen."

6. If no risks are found, risk_level is "none", details is [], and
   what_to_do reassures the user while reminding them of general safe use.

7. Output ONLY the JSON object. No preamble, no markdown fences, no commentary.

TONE:
- Calm and informative, not alarming
- Honest about severity without catastrophising
- Respectful — assume the user didn't know about these risks
- Never judgemental about the user's choices
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHESIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_risk_report(
    resolved_products: list[dict],
    interactions:      dict,
    dose_check:        dict,
    original_products: list[str],
    extra_substances:  list[str] | None = None,
) -> dict:
    """
    Takes the output of all three data layer tools and calls Claude to
    produce a structured risk report.

    Args:
        resolved_products : list of results from resolve_brand_to_ingredients
        interactions      : result from get_interactions
        dose_check        : result from check_dose_accumulation
        original_products : the product names the user mentioned (for context)
        extra_substances  : e.g. ["ethanol"] if user mentioned alcohol

    Returns:
        {
          "output": {risk_level, risk_summary, details, what_to_do, see_pharmacist},
          "sources": consolidated list of all sources cited,
          "raw_response": raw Claude output for debugging,
          "error": None or error message string
        }
    """
    extra_substances = extra_substances or []

    # ── Build the tool summary to pass to the LLM ────────────────────────────
    tool_summary = _build_tool_summary(
        resolved_products, interactions, dose_check,
        original_products, extra_substances
    )

    # ── Collect all sources for citation ─────────────────────────────────────
    all_sources = []
    for r in resolved_products:
        all_sources.extend(r.get("sources", []))
    all_sources.extend(interactions.get("sources", []))
    all_sources.extend(dose_check.get("sources", []))
    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for s in all_sources:
        url = s.get("url", s.get("name", ""))
        if url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(s)

    # ── Call Claude ───────────────────────────────────────────────────────────
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYNTHESIS_PROMPT,
            messages=[{"role": "user", "content": tool_summary}]
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        output = json.loads(raw)

        # Validate required keys
        for key in ["risk_level", "risk_summary", "details",
                    "what_to_do", "see_pharmacist"]:
            if key not in output:
                raise ValueError(f"Missing required key: {key}")

        return {
            "output":       output,
            "sources":      unique_sources,
            "raw_response": raw,
            "error":        None,
        }

    except json.JSONDecodeError as e:
        return {
            "output":       None,
            "sources":      unique_sources,
            "raw_response": raw if "raw" in dir() else "",
            "error":        f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "output":       None,
            "sources":      unique_sources,
            "raw_response": "",
            "error":        str(e),
        }


def _build_tool_summary(
    resolved_products: list[dict],
    interactions:      dict,
    dose_check:        dict,
    original_products: list[str],
    extra_substances:  list[str],
) -> str:
    """
    Formats tool outputs into a clear, structured prompt for the LLM.
    This is what the LLM reads to write the risk report.
    """
    lines = []
    lines.append(f"Products the user mentioned: {', '.join(original_products)}")
    if extra_substances:
        lines.append(f"Other substances mentioned: {', '.join(extra_substances)}")

    lines.append("\n--- TOOL 1: RESOLVED INGREDIENTS ---")
    for r in resolved_products:
        if r.get("data"):
            ings = [
                f"{i['ingredient']} {i['amount_per_dose_mg']}mg"
                if i.get("amount_per_dose_mg")
                else i["ingredient"]
                for i in r["data"]
                if " / " not in i.get("ingredient", "")   # skip combo entries
            ]
            if ings:
                lines.append(f"{r['product']}: {', '.join(ings)}")
        if r.get("warning"):
            lines.append(f"  WARNING: {r['warning']}")

    lines.append("\n--- TOOL 2: INTERACTIONS ---")
    if interactions.get("data"):
        for ix in interactions["data"]:
            lines.append(
                f"[{ix['severity'].upper()}] {ix['ingredient_a']} + {ix['ingredient_b']}: "
                f"{ix['description'][:300]}"
            )
            lines.append(f"  Management: {ix['management']}")
            sources = [s['name'] for s in interactions.get('sources', [])
                       if ix['ingredient_a'] in s['name'].lower()
                       or ix['ingredient_b'] in s['name'].lower()]
            if sources:
                lines.append(f"  Source: {sources[0]}")
    else:
        lines.append("No interactions found between the listed ingredients.")
    if interactions.get("warning"):
        lines.append(f"NOTE: {interactions['warning']}")

    lines.append("\n--- TOOL 3: DOSE ACCUMULATION ---")
    flagged = [i for i in dose_check.get("data", []) if i["exceeds_limit"]]
    safe    = [i for i in dose_check.get("data", [])
               if not i["exceeds_limit"] and i.get("safe_limit_mg")]

    if flagged:
        for item in flagged:
            lines.append(
                f"EXCEEDS LIMIT: {item['ingredient']} — "
                f"{item['total_mg_per_day']:.0f}mg/day "
                f"(safe limit: {item['safe_limit_mg']}mg/day) "
                f"from: {', '.join(item['contributing_products'])}"
            )
    if safe:
        for item in safe:
            lines.append(
                f"Within limit: {item['ingredient']} — "
                f"{item['total_mg_per_day']:.0f}mg/day "
                f"(limit: {item['safe_limit_mg']}mg/day)"
            )
    if not flagged and not safe:
        lines.append("No dose limit exceedances detected.")
    if dose_check.get("warning"):
        lines.append(f"NOTE: {dose_check['warning']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM INTERACTION REASONING
# ─────────────────────────────────────────────────────────────────────────────

LLM_INTERACTION_PROMPT = """
You are a clinical pharmacologist. Given a list of drug ingredients and
substances, identify any clinically significant interactions that a patient
should know about — including ones that may not appear in standard drug
label databases.

Focus especially on:
- Pharmacokinetic interactions (CYP450 enzyme inhibition/induction)
- Serotonin syndrome risk (SSRIs, SNRIs, tramadol, dextromethorphan,
  St. John's Wort, triptans, linezolid, MAOIs)
- CNS depression stacking (opioids, benzodiazepines, antihistamines,
  alcohol, sleep aids, muscle relaxants)
- Stimulant overload (pseudoephedrine, caffeine, ephedrine, synephrine,
  yohimbine, DMAA)
- Duplicate therapeutic class (two NSAIDs, two antihistamines,
  two decongestants, two acetaminophen sources)
- QT prolongation risk
- Bleeding risk amplification
- Hypertensive crisis risk (MAOIs + tyramine, stimulants + SSRIs)

Return ONLY a JSON array. Each item must have:
{
  "ingredient_a": "string",
  "ingredient_b": "string",
  "severity": "high" | "moderate" | "low",
  "mechanism": "one sentence — what is happening pharmacologically",
  "description": "2-3 sentences plain English — what could happen to the patient",
  "management": "one concrete sentence — what should they do",
  "source": "LLM pharmacological reasoning"
}

Return an empty array [] if there are no clinically significant interactions.
Return ONLY the JSON array, no preamble, no markdown fences.
""".strip()


def llm_interaction_reasoning(ingredient_list: list[str]) -> list[dict]:
    """
    Uses Claude to reason pharmacologically about a list of ingredients
    and identify interactions that OpenFDA label searches may miss.

    This catches:
    - Serotonin syndrome (SSRI + dextromethorphan, St. John's Wort + SSRI)
    - CYP450 interactions (St. John's Wort + many drugs)
    - Stimulant stacking (pseudoephedrine + caffeine)
    - Duplicate class (two antihistamines, two NSAIDs)
    - CNS depression combinations

    Returns a list of interaction dicts in the same format as
    _check_hardcoded_interactions() so they merge cleanly.
    """
    if len(ingredient_list) < 2:
        return []

    user_msg = (
        f"Ingredients and substances to check:\n"
        f"{', '.join(ingredient_list)}\n\n"
        f"Identify all clinically significant interactions between these."
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=LLM_INTERACTION_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        # Validate and normalise each interaction
        validated = []
        for ix in parsed:
            if not all(k in ix for k in
                       ["ingredient_a", "ingredient_b", "severity", "description"]):
                continue
            # Ensure management key exists
            ix.setdefault("management",
                          "Consult a pharmacist before combining these.")
            ix.setdefault("source", "LLM pharmacological reasoning")
            ix.setdefault("mechanism", "")
            validated.append(ix)

        return validated

    except Exception as e:
        print(f"  [LLM reasoning] Error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    products:          list[str],
    doses_per_day:     dict | None = None,
    extra_substances:  list[str] | None = None,
) -> dict:
    """
    Full end-to-end pipeline. Takes product names, runs all three tools,
    calls synthesis, returns the complete report.

    Args:
        products         : list of OTC product names e.g. ["NyQuil", "Tylenol"]
        doses_per_day    : optional {product: n_doses} e.g. {"NyQuil": 2}
        extra_substances : substances to include in interaction check
                           e.g. ["ethanol"] if user mentioned alcohol

    Returns:
        {
          "products": original product list,
          "output": {risk_level, risk_summary, details, what_to_do, see_pharmacist},
          "sources": [{name, url}, ...],
          "tool_data": {resolved, interactions, dose_check},
          "error": None or error string
        }
    """
    extra_substances = extra_substances or []
    doses_per_day    = doses_per_day or {}

    # ── Tool 1: resolve all products ─────────────────────────────────────────
    print(f"  [1/3] Resolving ingredients for: {products}")
    resolved = [resolve_brand_to_ingredients(p) for p in products]

    # ── Tool 2: check interactions ───────────────────────────────────────────
    print(f"  [2/3] Checking interactions...")
    all_ingredients = []
    for r in resolved:
        for ing in r.get("data", []):
            name = ing.get("ingredient", "")
            if " / " not in name:
                # Normalize salt forms to base ingredient for interaction lookup
                normalized = INGREDIENT_ALIASES.get(name, name)
                all_ingredients.append(normalized)
    all_ingredients = list(dict.fromkeys(all_ingredients))   # deduplicate
    all_ingredients.extend(extra_substances)

    interactions = get_interactions(all_ingredients)

    # ── LLM interaction reasoning ─────────────────────────────────────────────
    # Runs Claude pharmacological reasoning across all ingredients to catch
    # interactions that OpenFDA label searches miss — serotonin syndrome,
    # CYP450 induction, stimulant stacking, duplicate classes, etc.
    print(f"  [3/4] LLM pharmacological reasoning...")
    llm_interactions = llm_interaction_reasoning(all_ingredients)

    # Merge LLM findings into interactions — deduplicate by ingredient pair
    existing_pairs = {
        (ix["ingredient_a"], ix["ingredient_b"])
        for ix in interactions.get("data", [])
    }
    new_findings = []
    for ix in llm_interactions:
        pair     = (ix["ingredient_a"], ix["ingredient_b"])
        pair_rev = (ix["ingredient_b"], ix["ingredient_a"])
        if pair not in existing_pairs and pair_rev not in existing_pairs:
            new_findings.append(ix)
            existing_pairs.add(pair)

    # Add LLM findings to interactions data and sources
    if new_findings:
        interactions["data"].extend(new_findings)
        interactions["sources"].append({
            "name": "LLM pharmacological reasoning (Claude)",
            "url":  "https://www.anthropic.com"
        })

    # ── Tool 3: dose accumulation ─────────────────────────────────────────────
    print(f"  [4/5] Checking dose accumulation...")
    dose_check = check_dose_accumulation(resolved, doses_per_day)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    print(f"  [5/5] Synthesizing risk report...")
    synthesis = synthesize_risk_report(
        resolved_products=resolved,
        interactions=interactions,
        dose_check=dose_check,
        original_products=products,
        extra_substances=extra_substances,
    )

    return {
        "products":  products,
        "output":    synthesis.get("output"),
        "sources":   synthesis.get("sources"),
        "tool_data": {
            "resolved":          resolved,
            "interactions":      interactions,
            "llm_interactions":  llm_interactions,
            "dose_check":        dose_check,
        },
        "error": synthesis.get("error"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEXT ENTRY POINT — accepts raw user text via extraction layer
# ─────────────────────────────────────────────────────────────────────────────

def run_from_text(user_text: str) -> dict:
    """
    Convenience entry point for the full pipeline starting from raw user text.
    Runs extraction first, then the full pipeline.

    This is the function your eventual UI will call.

    Usage:
        from synthesis_layer import run_from_text, print_report
        report = run_from_text("I take NyQuil and Tylenol every night with wine")
        print_report(report)

    Args:
        user_text : anything the user typed — messy, informal, typos ok

    Returns:
        Same structure as run_pipeline() plus:
          "extracted": the extraction layer output (for debugging)
    """
    from extraction_layer import extract_medications, extraction_to_pipeline_args

    print(f"  [0/4] Extracting medications from text...")
    extracted = extract_medications(user_text)

    if extracted.get("error"):
        return {
            "products":  [],
            "output":    None,
            "sources":   [],
            "tool_data": {},
            "extracted": extracted,
            "error":     f"Extraction failed: {extracted['error']}",
        }

    if not extracted.get("products"):
        return {
            "products":  [],
            "output":    {
                "risk_level":    "none",
                "risk_summary":  "No medications were detected in your message.",
                "details":       [],
                "what_to_do":    "If you meant to ask about a specific medication, "
                                 "try naming it directly — for example: "
                                 "'I take Advil and Tylenol together.'",
                "see_pharmacist": False,
            },
            "sources":   [],
            "tool_data": {},
            "extracted": extracted,
            "error":     None,
        }

    args   = extraction_to_pipeline_args(extracted)
    report = run_pipeline(**args)
    report["extracted"] = extracted

    # Surface extraction notes as a pipeline warning if present
    notes = extracted.get("extraction_notes", "").strip()
    if notes and report.get("output"):
        report["extraction_notes"] = notes

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: dict) -> None:
    """Print a pipeline result in readable format."""
    if report.get("error"):
        print(f"\n❌ Error: {report['error']}")
        return

    out = report.get("output")
    if not out:
        print("\n❌ No output generated.")
        return

    LEVEL_ICON = {"high": "🔴", "moderate": "🟡", "low": "🟢", "none": "✅"}

    print("\n" + "═" * 60)
    print(f"OTC RISK REPORT — {', '.join(report['products'])}")
    print("═" * 60)

    icon  = LEVEL_ICON.get(out["risk_level"], "⚪")
    level = out["risk_level"].upper()
    print(f"\n{icon}  Risk level: {level}")
    print(f"\n{out['risk_summary']}")

    if out.get("details"):
        print("\n── DETAILS ──────────────────────────────────────────")
        for i, d in enumerate(out["details"], 1):
            print(f"\n{i}. {d['finding']}")
            print(f"   {d['explanation']}")
            print(f"   Source: {d['source']}")

    print("\n── WHAT TO DO ───────────────────────────────────────")
    print(f"\n{out['what_to_do']}")

    if out.get("see_pharmacist"):
        print("\n⚕  Speak with a pharmacist or doctor before continuing "
              "this combination.")

    if report.get("sources"):
        print("\n── SOURCES ──────────────────────────────────────────")
        for s in report["sources"]:
            print(f"   • {s['name']}")
            if s.get("url"):
                print(f"     {s['url']}")

    print("\n" + "═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MANUAL TESTS — run directly to verify
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Tests use run_from_text() — the full pipeline from raw user input
    TEST_CASES = [
        {
            "label": "Duplicate acetaminophen + alcohol",
            "text":  "I take Theraflu twice a day and Tylenol 3 times a day, and I drink wine every night",
        },
        {
            "label": "Dual NSAID",
            "text":  "I take Advil three times a day and Aleve twice a day for my back pain",
        },
        {
            "label": "Sleep aid + alcohol",
            "text":  "I take Benadryl to sleep and have 4-5 beers most nights",
        },
        {
            "label": "No risk — single safe dose",
            "text":  "I take one Tylenol in the morning for a headache",
        },
    ]

    for tc in TEST_CASES:
        print(f"\n{'─'*60}")
        print(f"TEST: {tc['label']}")
        print(f"Input: \"{tc['text']}\"")
        print(f"{'─'*60}")

        report = run_from_text(tc["text"])

        # Show what the extraction layer found
        extracted = report.get("extracted", {})
        if extracted.get("products"):
            products   = [p["name"] for p in extracted["products"]]
            substances = [s["name"] for s in extracted.get("substances", [])]
            print(f"  Extracted: products={products}  substances={substances}")
        if report.get("extraction_notes"):
            print(f"  Extraction notes: {report['extraction_notes']}")

        print_report(report)

        fname = f"synthesis_{tc['label'][:25].replace(' ','_')}.json"
        with open(fname, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Saved → {fname}")
