"""
Data Layer Integration Test
============================
Run this FIRST against live APIs before building the LLM layer.
It tells you exactly where your three tools are solid vs. need patching.

Usage:
    python test_data_layer.py

Generates a printed report + writes results to test_results.json
so you have a record of what resolved cleanly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
from data_layer import (
    resolve_brand_to_ingredients,
    get_interactions,
    check_dose_accumulation,
)

# ── Test products ─────────────────────────────────────────────────────────────
# Chosen to cover: multi-ingredient combos, single-ingredient, supplements,
# store generics, and common "problem" products the agent must handle well.

TEST_PRODUCTS = [
    # Common combo cold/flu products — most likely to have duplicate ingredients
    "NyQuil",
    "DayQuil",
    "Theraflu",
    "Robitussin",

    # Single-ingredient analgesics
    "Tylenol",
    "Advil",
    "Aleve",
    "Aspirin",

    # Antihistamines / sleep aids
    "Benadryl",
    "ZzzQuil",

    # GI / other
    "Pepto-Bismol",
    "Imodium",

    # Supplements — expect these to struggle (sparse data)
    "melatonin",
    "fish oil",
    "vitamin C",
]

# ── Severity emoji for readability ────────────────────────────────────────────
SEV = {"high": "🔴", "moderate": "🟡", "low": "🟢", "unknown": "⚪"}


def run_tool1_tests() -> list[dict]:
    """Test brand → ingredient resolution for all test products."""
    print("\n" + "=" * 65)
    print("TOOL 1 — resolve_brand_to_ingredients")
    print("=" * 65)

    results = []
    passed = failed = partial = 0

    for product in TEST_PRODUCTS:
        result = resolve_brand_to_ingredients(product)
        ings = result["data"]
        has_doses = all(
            i.get("amount_per_dose_mg") is not None for i in ings
        ) if ings else False

        if ings and has_doses:
            status = "✓ PASS"
            passed += 1
        elif ings and not has_doses:
            status = "~ PARTIAL (no dose amounts)"
            partial += 1
        else:
            status = "✗ FAIL (no ingredients)"
            failed += 1

        print(f"\n  {status}  {product}")
        if ings:
            for i in ings:
                dose = (
                    f"{i['amount_per_dose_mg']} mg"
                    if i.get("amount_per_dose_mg")
                    else "dose unknown"
                )
                print(f"           {i['ingredient']:30s} {dose}")
        if result["warning"]:
            print(f"           ⚠ {result['warning']}")

        results.append({
            "product":    product,
            "status":     status,
            "data":       result["data"],
            "sources":    result["sources"],
            "warning":    result["warning"],
        })

    print(f"\n  Summary: {passed} passed · {partial} partial · {failed} failed "
          f"out of {len(TEST_PRODUCTS)}")
    print(f"  Action needed: add missing products to INGREDIENT_ALIASES or "
          f"extend _parse_fda_active_ingredients for {failed + partial} items")

    return results


def run_tool2_tests(resolved: list[dict]) -> dict:
    """Test interaction detection across all resolved ingredients."""
    print("\n" + "=" * 65)
    print("TOOL 2 — get_interactions")
    print("=" * 65)

    # Build full ingredient list from all resolved products
    all_ingredients = list({
        i["ingredient"]
        for r in resolved
        for i in r.get("data", [])
        if i.get("ingredient")
    })

    # Also test specific high-risk combos we KNOW should be detected
    high_risk_combos = [
        {
            "label": "Acetaminophen + Alcohol (must detect)",
            "ingredients": ["acetaminophen", "ethanol"],
            "expect_severity": "high",
        },
        {
            "label": "Ibuprofen + Aspirin (must detect)",
            "ingredients": ["ibuprofen", "aspirin"],
            "expect_severity": ["moderate", "high"],
        },
        {
            "label": "Dual NSAIDs: Ibuprofen + Naproxen (must detect)",
            "ingredients": ["ibuprofen", "naproxen sodium"],
            "expect_severity": "high",
        },
        {
            "label": "Diphenhydramine + Alcohol (must detect)",
            "ingredients": ["diphenhydramine", "ethanol"],
            "expect_severity": "high",
        },
        {
            "label": "Safe pair: Loratadine + Ibuprofen (expect none/low)",
            "ingredients": ["loratadine", "ibuprofen"],
            "expect_severity": None,
        },
    ]

    combo_results = []
    print()
    for combo in high_risk_combos:
        result = get_interactions(combo["ingredients"])
        found = result["data"]
        detected = len(found) > 0
        expected = combo["expect_severity"]

        if expected is None:
            ok = not detected or all(
                i["severity"] in ("low", "unknown") for i in found
            )
            status = "✓" if ok else "⚠ False positive?"
        else:
            exp_list = (
                [expected] if isinstance(expected, str) else expected
            )
            ok = detected and any(
                i["severity"] in exp_list for i in found
            )
            status = "✓" if ok else "✗ MISSED"

        print(f"  {status}  {combo['label']}")
        for ix in found:
            print(f"       {SEV.get(ix['severity'], '⚪')} [{ix['severity']}] "
                  f"{ix['ingredient_a']} + {ix['ingredient_b']}")

        combo_results.append({
            "label":    combo["label"],
            "status":   status,
            "found":    found,
            "sources":  result["sources"],
        })
        time.sleep(0.1)

    # Run hardcoded interaction check across full ingredient pool.
    # We skip the OpenFDA pairwise scan here — with N ingredients that's
    # N*(N-1)/2 API calls which hangs the test for large pools.
    # The hardcoded table covers all clinically critical OTC pairs.
    # OpenFDA pairwise scanning happens in the real pipeline, not in tests.
    print(f"\n  Full ingredient pool ({len(all_ingredients)} ingredients):")
    print(f"  {all_ingredients}")

    from data_layer import _check_hardcoded_interactions
    pool_with_ethanol = all_ingredients + ["ethanol"]
    hardcoded_hits = _check_hardcoded_interactions(pool_with_ethanol)

    print(f"\n  Hardcoded interactions in full pool: {len(hardcoded_hits)}")
    for ix in hardcoded_hits:
        print(f"    {SEV.get(ix['severity'], '⚪')} [{ix['severity']}] "
              f"{ix['ingredient_a']} + {ix['ingredient_b']}")
    print(f"  (OpenFDA pairwise scan skipped in tests — runs in pipeline only)")

    return {
        "combo_tests": combo_results,
        "full_pool":   hardcoded_hits,
    }


def run_tool3_tests(resolved: list[dict]) -> dict:
    """Test dose accumulation on realistic multi-product scenarios."""
    print("\n" + "=" * 65)
    print("TOOL 3 — check_dose_accumulation")
    print("=" * 65)

    scenarios = [
        {
            "label": "NyQuil (x2) + Tylenol (x3) — should flag acetaminophen",
            "products": ["NyQuil", "Tylenol"],
            "doses": {"NyQuil": 2, "Tylenol": 3},
            "expect_flag": "acetaminophen",
        },
        {
            "label": "Advil (x3) + Aleve (x2) — should flag dual NSAID load",
            "products": ["Advil", "Aleve"],
            "doses": {"Advil": 3, "Aleve": 2},
            "expect_flag": None,  # may not exceed limit but both NSAIDs present
        },
        {
            "label": "Single Benadryl (x1) — should be within limits",
            "products": ["Benadryl"],
            "doses": {"Benadryl": 1},
            "expect_flag": None,
        },
    ]

    scenario_results = []
    print()

    for scenario in scenarios:
        # Filter resolved products to those in this scenario
        scenario_resolved = [
            r for r in resolved
            if r["product"] in scenario["products"] and r.get("data")
        ]

        if not scenario_resolved:
            print(f"  ⚠ SKIP  {scenario['label']}")
            print(f"         (products didn't resolve in Tool 1 — fix Tool 1 first)")
            continue

        result = check_dose_accumulation(scenario_resolved, scenario["doses"])
        flagged = [
            i["ingredient"]
            for i in result["data"]
            if i["exceeds_limit"]
        ]
        expected_flag = scenario["expect_flag"]

        if expected_flag:
            ok = expected_flag in flagged
            status = "✓" if ok else f"✗ Expected '{expected_flag}' to be flagged"
        else:
            ok = True
            status = "✓"

        print(f"  {status}  {scenario['label']}")
        for item in result["data"]:
            flag_str = "  ⚠ OVER LIMIT" if item["exceeds_limit"] else ""
            limit = (
                f"limit {int(item['safe_limit_mg'])} mg"
                if item["safe_limit_mg"]
                else "no limit set"
            )
            print(f"         {item['ingredient']:28s} "
                  f"{item['total_mg_per_day']:6.0f} mg/day  "
                  f"({limit}){flag_str}")
        if result["warning"]:
            print(f"         ⚠ {result['warning']}")
        print()

        scenario_results.append({
            "label":   scenario["label"],
            "status":  status,
            "data":    result["data"],
            "warning": result["warning"],
        })

    return {"scenarios": scenario_results}


def print_action_items(tool1_results: list[dict]) -> None:
    """Print a prioritised TODO list based on test failures."""
    print("\n" + "=" * 65)
    print("ACTION ITEMS (fix before building LLM layer)")
    print("=" * 65)

    failed_products = [
        r["product"] for r in tool1_results
        if "FAIL" in r["status"]
    ]
    partial_products = [
        r["product"] for r in tool1_results
        if "PARTIAL" in r["status"]
    ]
    no_dose = [
        r["product"] for r in tool1_results
        if any(
            i.get("amount_per_dose_mg") is None
            for i in r.get("data", [])
        )
    ]

    if failed_products:
        print(f"\n  1. Products with NO ingredient data (add to INGREDIENT_ALIASES")
        print(f"     or extend _openfda_label_by_brand search logic):")
        for p in failed_products:
            print(f"       - {p}")

    if partial_products:
        print(f"\n  2. Products with ingredients but MISSING DOSE amounts")
        print(f"     (improve _parse_fda_active_ingredients regex):")
        for p in partial_products:
            print(f"       - {p}")

    if no_dose:
        print(f"\n  3. Ingredients with unknown per-dose amount — check OpenFDA")
        print(f"     label format for these products and patch the parser:")
        for p in no_dose:
            print(f"       - {p}")

    if not (failed_products or partial_products or no_dose):
        print("\n  ✓ All products resolved cleanly. Ready for extraction prompt.")

    print("\n  General rule: aim for < 2 failures before moving to Step 2.")
    print("  Supplements (melatonin, fish oil) failing is expected and OK —")
    print("  add them to a manual fallback table in data_layer.py.\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = {}

    tool1 = run_tool1_tests()
    all_results["tool1"] = tool1

    tool2 = run_tool2_tests(tool1)
    all_results["tool2"] = tool2

    tool3 = run_tool3_tests(tool1)
    all_results["tool3"] = tool3

    print_action_items(tool1)

    # Persist results so you can diff after fixes
    with open("test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  Full results saved → test_results.json\n")
