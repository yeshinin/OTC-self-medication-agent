"""
OTC Self-Medication Agent — Phase 1 Data Layer
===============================================
Three tools the agent will call:
  1. resolve_brand_to_ingredients(product_name)  → via RxNorm + OpenFDA
  2. get_interactions(ingredient_list)            → via OpenFDA
  3. check_dose_accumulation(resolved_products)   → local logic + OpenFDA label data

All functions return a dict with:
  - data      : the actual result
  - sources   : list of {name, url} for citation in the final output
  - warning   : human-readable string if something went wrong / low confidence

No API keys required for RxNorm or OpenFDA (both are free & open).
Rate limits: RxNorm = 20 req/sec, OpenFDA = 240 req/min (no key) or 1000/min (with key).
We add small sleeps to stay safe.
"""

import time
import requests
from itertools import combinations

# ── Base URLs ────────────────────────────────────────────────────────────────

RXNORM_BASE  = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug"

# ── Known OTC safe daily limits (mg) ─────────────────────────────────────────
# Source: FDA OTC monographs + product labeling
# Extend this dict as you add more ingredients.
SAFE_DAILY_LIMITS = {
    "acetaminophen":       3000,   # 2000 mg/day for regular alcohol users
    "ibuprofen":           1200,   # OTC limit; Rx goes higher
    "naproxen sodium":      660,   # OTC limit (440 mg naproxen base)
    "aspirin":             4000,
    "diphenhydramine":      300,
    "dextromethorphan":     120,
    "guaifenesin":         2400,
    "pseudoephedrine":      240,
    "phenylephrine":        60,
    "loratadine":           10,
    "cetirizine":           10,
    "famotidine":           40,    # OTC dose
    "omeprazole":           20,    # OTC dose
    "loperamide":           16,
    "bismuth subsalicylate": 4200, # in mg; watch salicylate load
    "melatonin":            10,    # conservative; no formal FDA monograph
    #adding more to the list based on common OTC ingredients and supplements
    "Diclofenac":           150,  # OTC topical dose; oral Rx doses can be higher
    "Fish oil":             3000,   # typical supplemental dose; no formal FDA limit
    "Vitamin C":            2000,   # upper limit for adults
    "Vitamin D":            4000,   # upper limit for adults
    "Zinc":                 40,          # upper limit for adults
    "Echinacea":            0,      # no established safe limit; use with caution
    "Diphenhydramine":      300, # typical OTC max dose; higher doses can be dangerous
}

# ── Ingredient aliases (handle common name variants) ─────────────────────────
INGREDIENT_ALIASES = {
    "apap": "acetaminophen",
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",          # sometimes people say ingredient=brand
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen sodium",
    "naproxen": "naproxen sodium",
    "benadryl": "diphenhydramine",
    "dph": "diphenhydramine",
    "dxm": "dextromethorphan",
    "pse": "pseudoephedrine",
}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 · resolve_brand_to_ingredients
# ─────────────────────────────────────────────────────────────────────────────

def resolve_brand_to_ingredients(product_name: str) -> dict:
    """
    Given a brand name or informal name (e.g. "NyQuil", "Advil", "fish oil"),
    return the list of active ingredients with their per-dose amounts.

    Strategy:
      Step 1 → RxNorm: find RxCUI for the product name
      Step 2 → RxNorm: walk the graph to ingredient-level RxCUIs (term type IN)
      Step 3 → OpenFDA label: pull active_ingredient section for dose details
      Fallback → OpenFDA label search by brand name directly if RxNorm misses it

    Returns:
      {
        data: [
          {
            ingredient: "acetaminophen",
            amount_per_dose_mg: 325,
            rxcui: "161",
          }, ...
        ],
        sources: [{name, url}, ...],
        warning: str or None
      }
    """
    sources = []
    warning = None
    ingredients = []

    # ── Step 1: RxNorm name → RxCUI ──────────────────────────────────────────
    rxcui = _rxnorm_name_to_rxcui(product_name)

    if rxcui:
        sources.append({
            "name": f"RxNorm: '{product_name}' (RxCUI {rxcui})",
            "url":  f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allrelated.json"
        })

        # ── Step 2: RxCUI → ingredient RxCUIs ────────────────────────────────
        ingredient_rxcuis = _rxnorm_rxcui_to_ingredients(rxcui)
        for ing in ingredient_rxcuis:
            normalized = INGREDIENT_ALIASES.get(ing["name"].lower(), ing["name"].lower())
            ingredients.append({
                "ingredient": normalized,
                "amount_per_dose_mg": None,   # filled in Step 3
                "rxcui": ing["rxcui"],
            })

    # ── Step 3: OpenFDA label → dose amounts (and fallback ingredient list) ──
    fda_label = _openfda_label_by_brand(product_name)

    if fda_label:
        label_url = (
            f"https://api.fda.gov/drug/label.json"
            f"?search=openfda.brand_name:\"{product_name}\"&limit=1"
        )
        sources.append({
            "name": f"OpenFDA label: '{product_name}'",
            "url":  label_url
        })

        fda_ingredients = _parse_fda_active_ingredients(fda_label)

        if not ingredients:
            # RxNorm returned nothing — use FDA as primary source
            ingredients = fda_ingredients
            if not ingredients:
                warning = (
                    f"Could not resolve '{product_name}' to ingredients. "
                    "It may be a supplement, herbal, or misspelled product name. "
                    "Manual review recommended."
                )
        else:
            # Merge FDA dose amounts into RxNorm ingredient list
            fda_by_name = {i["ingredient"]: i for i in fda_ingredients}
            for ing in ingredients:
                match = fda_by_name.get(ing["ingredient"])
                if match:
                    ing["amount_per_dose_mg"] = match.get("amount_per_dose_mg")

    elif not ingredients:
        warning = (
            f"No data found for '{product_name}' in RxNorm or OpenFDA. "
            "Check spelling or try the generic ingredient name."
        )

    time.sleep(0.1)   # gentle rate-limit buffer

    return {
        "product":    product_name,
        "data":       ingredients,
        "sources":    sources,
        "warning":    warning,
    }


def _rxnorm_name_to_rxcui(name: str) -> str | None:
    """Calls RxNorm findRxcuiByString. Returns best-match RxCUI or None."""
    url = f"{RXNORM_BASE}/rxcui.json"
    try:
        resp = requests.get(url, params={"name": name, "search": 2}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        rxcui = (
            data.get("idGroup", {})
                .get("rxnormId", [None])[0]
        )
        return rxcui
    except Exception:
        return None


def _rxnorm_rxcui_to_ingredients(rxcui: str) -> list[dict]:
    """
    Walks the RxNorm graph from a product RxCUI to its ingredient (IN) concepts.
    Returns list of {rxcui, name}.
    Note: RxNorm interaction API was discontinued Jan 2024 — we only use it
    here for brand→ingredient resolution, which still works fine.
    """
    url = f"{RXNORM_BASE}/rxcui/{rxcui}/allrelated.json"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        concept_groups = (
            resp.json()
                .get("allRelatedGroup", {})
                .get("conceptGroup", [])
        )
        ingredients = []
        for group in concept_groups:
            # IN = ingredient, PIN = precise ingredient, MIN = multiple ingredients
            if group.get("tty") in ("IN", "PIN", "MIN"):
                for prop in group.get("conceptProperties", []):
                    ingredients.append({
                        "rxcui": prop["rxcui"],
                        "name":  prop["name"].lower(),
                    })
        return ingredients
    except Exception:
        return []


def _openfda_label_by_brand(brand_name: str) -> dict | None:
    """Fetches the first matching OTC drug label from OpenFDA."""
    url = f"{OPENFDA_BASE}/label.json"
    try:
        # Try OTC product_type filter first for precision
        resp = requests.get(url, params={
            "search": (
                f'openfda.brand_name:"{brand_name}"'
                f'+AND+openfda.product_type:"HUMAN+OTC+DRUG"'
            ),
            "limit": 1
        }, timeout=8)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return results[0]

        # Broader search if OTC filter returns nothing
        resp = requests.get(url, params={
            "search": f'openfda.brand_name:"{brand_name}"',
            "limit": 1
        }, timeout=8)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None

    except Exception:
        return None


def _parse_fda_active_ingredients(label: dict) -> list[dict]:
    """
    Extracts active ingredients + amounts from an OpenFDA label record.
    The 'active_ingredient' field is a list of free-text strings like:
      ["Active ingredient (in each tablet)\nAcetaminophen 500 mg\nPurpose\nPain reliever"]
    We extract ingredient names and numeric mg values with basic parsing.
    For Phase 1 this is good enough; a more robust NLP parse can be added later.
    """
    import re
    results = []
    raw_sections = label.get("active_ingredient", [])

    for section in raw_sections:
        # Find lines that look like "IngredientName NNN mg"
        lines = section.replace("\n", " ").split("  ")
        for line in lines:
            # Match patterns like "Acetaminophen 500 mg" or "ibuprofen 200mg"
            match = re.search(
                r"([a-zA-Z][a-zA-Z\s\-]+?)\s+([\d.]+)\s*mg",
                line,
                re.IGNORECASE
            )
            if match:
                name = match.group(1).strip().lower()
                # Remove noise words
                for noise in ["active ingredient", "purpose", "each", "tablet",
                              "capsule", "softgel", "liquid"]:
                    name = name.replace(noise, "").strip()
                name = INGREDIENT_ALIASES.get(name, name)
                if len(name) > 2:
                    results.append({
                        "ingredient": name,
                        "amount_per_dose_mg": float(match.group(2)),
                        "rxcui": None,
                    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 · get_interactions
# ─────────────────────────────────────────────────────────────────────────────

def get_interactions(ingredient_list: list[str]) -> dict:
    """
    Given a flat list of ingredient names, check all pairwise combinations
    for known interactions.

    Sources used:
      - OpenFDA drug label: warnings + drug_interactions sections
        (most reliable for OTC-specific language)

    Note: DrugBank interaction API requires a paid key for production use.
    For Phase 1 we use OpenFDA label text, which covers the clinically
    important OTC interactions well. DrugBank can be wired in Phase 2.

    Returns:
      {
        data: [
          {
            ingredient_a: str,
            ingredient_b: str,
            severity: "high" | "moderate" | "low" | "unknown",
            description: str,
            management: str,
          }, ...
        ],
        sources: [{name, url}, ...],
        warning: str or None
      }
    """
    sources = []
    interactions = []
    warnings_list = []

    # Normalise input
    normalized = [INGREDIENT_ALIASES.get(i.lower(), i.lower())
                  for i in ingredient_list]
    unique = list(dict.fromkeys(normalized))   # deduplicate, preserve order

    if len(unique) < 2:
        return {
            "data": [],
            "sources": [],
            "warning": "Need at least two ingredients to check interactions."
        }

    # Check each pair
    for ing_a, ing_b in combinations(unique, 2):
        result = _openfda_check_interaction_pair(ing_a, ing_b)
        if result:
            interactions.append(result)
            sources.append({
                "name": (
                    f"OpenFDA label drug interactions: "
                    f"{ing_a} + {ing_b}"
                ),
                "url": (
                    f"https://api.fda.gov/drug/label.json"
                    f"?search=drug_interactions:\"{ing_b}\""
                    f"+AND+active_ingredient:\"{ing_a}\"&limit=1"
                )
            })
        time.sleep(0.07)   # stay under 240 req/min

    # Flag known high-risk OTC combos not always in label text
    hardcoded = _check_hardcoded_interactions(unique)
    for hc in hardcoded:
        # Add only if not already found via FDA query
        already = any(
            (i["ingredient_a"] == hc["ingredient_a"] and
             i["ingredient_b"] == hc["ingredient_b"])
            for i in interactions
        )
        if not already:
            interactions.append(hc)
            sources.append({
                "name": (
                    f"FDA OTC monograph / clinical literature: "
                    f"{hc['ingredient_a']} + {hc['ingredient_b']}"
                ),
                "url": "https://www.fda.gov/drugs/drug-interactions-labeling"
            })

    if not interactions:
        warnings_list.append(
            "No documented interactions found between these ingredients. "
            "Absence of data does not confirm safety, especially for supplements."
        )

    return {
        "data":    interactions,
        "sources": sources,
        "warning": " | ".join(warnings_list) if warnings_list else None,
    }


def _openfda_check_interaction_pair(ing_a: str, ing_b: str) -> dict | None:
    """
    Searches OpenFDA label drug_interactions section for ing_b mentioned
    in a label where ing_a is the active ingredient.
    Returns a structured interaction dict or None.
    """
    url = f"{OPENFDA_BASE}/label.json"
    try:
        resp = requests.get(url, params={
            "search": (
                f'active_ingredient:"{ing_a}"'
                f'+AND+drug_interactions:"{ing_b}"'
            ),
            "limit": 1
        }, timeout=8)
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None

        label = results[0]
        interaction_text = " ".join(label.get("drug_interactions", []))
        severity = _infer_severity(interaction_text)

        return {
            "ingredient_a":  ing_a,
            "ingredient_b":  ing_b,
            "severity":      severity,
            "description":   interaction_text[:500] if interaction_text else
                             f"Interaction between {ing_a} and {ing_b} noted in product labeling.",
            "management":    _extract_management_advice(interaction_text),
        }

    except Exception:
        return None


def _check_hardcoded_interactions(ingredients: list[str]) -> list[dict]:
    """
    High-priority OTC interaction pairs that are clinically important but
    may not always surface cleanly from label text searches.
    Curated from FDA monographs and clinical pharmacology references.
    """
    known = [
        {
            "a": "acetaminophen", "b": "ethanol",
            "severity": "high",
            "description": (
                "Chronic alcohol use (3+ drinks/day) combined with acetaminophen "
                "significantly increases risk of hepatotoxicity. FDA requires "
                "alcohol warning on all acetaminophen OTC labels."
            ),
            "management": "Avoid acetaminophen or limit to <2g/day if drinking regularly."
        },
        {
            "a": "ibuprofen", "b": "aspirin",
            "severity": "moderate",
            "description": (
                "Ibuprofen can interfere with aspirin's antiplatelet effect when "
                "taken within 30 minutes before or 8 hours after aspirin. "
                "Both increase GI bleeding risk."
            ),
            "management": "Take aspirin at least 30 min before ibuprofen. Consider alternatives."
        },
        {
            "a": "ibuprofen", "b": "naproxen sodium",
            "severity": "high",
            "description": (
                "Two NSAIDs taken together do not provide additional pain relief "
                "but significantly increase risk of GI ulcers, bleeding, and "
                "kidney damage."
            ),
            "management": "Never combine two NSAIDs. Use one or the other."
        },
        {
            "a": "diphenhydramine", "b": "ethanol",
            "severity": "high",
            "description": (
                "Diphenhydramine (found in Benadryl, NyQuil, ZzzQuil, many sleep aids) "
                "combined with alcohol causes additive CNS depression: extreme drowsiness, "
                "impaired coordination, risk of respiratory depression."
            ),
            "management": "Avoid alcohol entirely when taking diphenhydramine."
        },
        {
            "a": "aspirin", "b": "bismuth subsalicylate",
            "severity": "moderate",
            "description": (
                "Bismuth subsalicylate (Pepto-Bismol) contains salicylate. "
                "Combined with aspirin, total salicylate load can cause toxicity, "
                "especially in children (Reye's syndrome risk)."
            ),
            "management": "Avoid combining. Do not give either to children with viral illness."
        },
        {
            "a": "pseudoephedrine", "b": "phenylephrine",
            "severity": "moderate",
            "description": (
                "Two decongestants combined increase cardiovascular risk: elevated "
                "blood pressure, heart rate, and risk of arrhythmia."
            ),
            "management": "Use only one decongestant at a time."
        },
    ]

    results = []
    ing_set = set(ingredients)

    for pair in known:
        a_present = pair["a"] in ing_set
        b_present = pair["b"] in ing_set

        # Also check if alcohol/ethanol mentioned by user context
        if pair["b"] == "ethanol":
            b_present = b_present or "alcohol" in ing_set

        if a_present and b_present:
            results.append({
                "ingredient_a": pair["a"],
                "ingredient_b": pair["b"],
                "severity":     pair["severity"],
                "description":  pair["description"],
                "management":   pair["management"],
            })

    return results


def _infer_severity(text: str) -> str:
    """Heuristic severity from label text keywords."""
    text_lower = text.lower()
    if any(w in text_lower for w in
           ["fatal", "life-threatening", "death", "serious", "severe",
            "hepatotoxicity", "hemorrhage", "overdose"]):
        return "high"
    elif any(w in text_lower for w in
             ["avoid", "caution", "monitor", "increased risk", "may increase"]):
        return "moderate"
    elif text_lower.strip():
        return "low"
    return "unknown"


def _extract_management_advice(text: str) -> str:
    """Pull the most actionable sentence from interaction text."""
    if not text:
        return "Consult a pharmacist before combining these medications."
    sentences = text.replace("\n", " ").split(".")
    # Prefer sentences with action words
    action_words = ["avoid", "do not", "consult", "stop", "reduce",
                    "limit", "take", "use", "monitor"]
    for sent in sentences:
        if any(w in sent.lower() for w in action_words):
            return sent.strip() + "."
    return sentences[0].strip() + "." if sentences else (
        "Consult a pharmacist before combining these medications."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 · check_dose_accumulation
# ─────────────────────────────────────────────────────────────────────────────

def check_dose_accumulation(resolved_products: list[dict],
                             doses_per_day: dict | None = None) -> dict:
    """
    Given a list of resolved products (output of resolve_brand_to_ingredients),
    sum ingredient totals across all products and flag any that exceed safe
    daily limits.

    Args:
      resolved_products : list of dicts from resolve_brand_to_ingredients
      doses_per_day     : optional {product_name: int} override from user input
                          e.g. {"NyQuil": 2, "Tylenol": 3}
                          Default assumes label-recommended doses (1 dose unit).

    Returns:
      {
        data: [
          {
            ingredient: str,
            total_mg_per_day: float,
            safe_limit_mg: float | None,
            exceeds_limit: bool,
            contributing_products: [str, ...]
          }, ...
        ],
        sources: [{name, url}],
        warning: str or None
      }
    """
    doses_per_day = doses_per_day or {}
    accumulation: dict[str, dict] = {}

    for product in resolved_products:
        product_name = product.get("product", "unknown product")
        n_doses = doses_per_day.get(product_name, 1)

        for ing in product.get("data", []):
            name   = ing["ingredient"]
            amount = ing.get("amount_per_dose_mg") or 0

            if name not in accumulation:
                accumulation[name] = {
                    "ingredient":            name,
                    "total_mg_per_day":      0.0,
                    "contributing_products": [],
                }
            accumulation[name]["total_mg_per_day"]      += amount * n_doses
            accumulation[name]["contributing_products"].append(product_name)

    # Attach safe limits and flag exceedances
    results  = []
    warnings = []

    for name, acc in accumulation.items():
        limit = SAFE_DAILY_LIMITS.get(name)
        exceeds = (limit is not None) and (acc["total_mg_per_day"] > limit)

        if exceeds:
            warnings.append(
                f"{name.title()}: estimated {acc['total_mg_per_day']:.0f} mg/day "
                f"exceeds safe OTC limit of {limit} mg/day."
            )

        results.append({
            "ingredient":             name,
            "total_mg_per_day":       acc["total_mg_per_day"],
            "safe_limit_mg":          limit,
            "exceeds_limit":          exceeds,
            "contributing_products":  list(set(acc["contributing_products"])),
        })

    # Sort: flagged items first, then alphabetical
    results.sort(key=lambda x: (not x["exceeds_limit"], x["ingredient"]))

    # Note if any ingredient has no dose data (amount = 0)
    no_dose_data = [r["ingredient"] for r in results
                    if r["total_mg_per_day"] == 0]
    if no_dose_data:
        warnings.append(
            f"Could not determine per-dose amounts for: "
            f"{', '.join(no_dose_data)}. Manual dose check recommended."
        )

    return {
        "data": results,
        "sources": [{
            "name": "FDA OTC safe daily dose limits (monograph + label guidance)",
            "url":  "https://www.fda.gov/drugs/drug-safety-and-availability/"
                    "questions-and-answers-information-acetaminophen-prescription-"
                    "drug-products-required-liver-warning"
        }],
        "warning": " | ".join(warnings) if warnings else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test — run this file directly to verify all three tools work
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    TEST_PRODUCTS = ["NyQuil", "Tylenol Extra Strength", "Advil"]
    TEST_PRODUCTS_1 = ["Tylenol", "Aleve", "Motrin", "Benadryl", "ZzzQuil", "Pepto-Bismol", "fish oil", "Voltaren"]  # for more edge cases
    TEST_PRODUCTS_2 = ["Tagamet HB", "Herbal Supp", "Grapefruit Juice", "Mylanta", "Bengay", "Sudafed"]
    print("=" * 60)
    print("TOOL 1 — resolve_brand_to_ingredients")
    print("=" * 60)

    resolved = []
    for p in TEST_PRODUCTS_2:
        result = resolve_brand_to_ingredients(p)
        resolved.append(result)
        print(f"\n{p}:")
        print(f"  ingredients : {[i['ingredient'] for i in result['data']]}")
        print(f"  sources     : {[s['name'] for s in result['sources']]}")
        if result["warning"]:
            print(f"  ⚠ warning  : {result['warning']}")

    print("\n" + "=" * 60)
    print("TOOL 2 — get_interactions")
    print("=" * 60)

    all_ingredients = [
        i["ingredient"]
        for r in resolved
        for i in r["data"]
    ]
    # Add alcohol to test the hardcoded interaction
    all_ingredients.append("ethanol")

    interactions = get_interactions(all_ingredients)
    print(f"\nIngredients checked: {all_ingredients}")
    print(f"Interactions found : {len(interactions['data'])}")
    for ix in interactions["data"]:
        print(f"\n  [{ix['severity'].upper()}] "
              f"{ix['ingredient_a']} + {ix['ingredient_b']}")
        print(f"  {ix['description'][:120]}...")
    if interactions["warning"]:
        print(f"\n⚠ {interactions['warning']}")

    print("\n" + "=" * 60)
    print("TOOL 3 — check_dose_accumulation")
    print("=" * 60)

    dose_check = check_dose_accumulation(
        resolved,
        doses_per_day={"NyQuil": 2, "Tylenol Extra Strength": 3, "Advil": 3}
    )
    print()
    for item in dose_check["data"]:
        flag = "⚠ EXCEEDS LIMIT" if item["exceeds_limit"] else "ok"
        print(f"  {item['ingredient']:30s} "
              f"{item['total_mg_per_day']:6.0f} mg/day  "
              f"(limit: {item['safe_limit_mg'] or 'n/a'})  {flag}")
        print(f"    from: {', '.join(item['contributing_products'])}")

    if dose_check["warning"]:
        print(f"\n⚠ {dose_check['warning']}")

    print("\n✓ Smoke test complete.")
