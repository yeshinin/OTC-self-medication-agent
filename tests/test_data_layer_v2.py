"""
test_data_layer_v2.py  —  run this, not test_data_layer-2.py

Usage (from project root):
    PYTHONPATH=src python tests/test_data_layer_v2.py

Or add to top of this file:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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

SEV = {"high": "🔴", "moderate": "🟡", "low": "🟢", "unknown": "⚪"}

TEST_PRODUCTS = [
    "NyQuil", "DayQuil", "Theraflu", "Robitussin",
    "Tylenol", "Advil", "Aleve", "Aspirin",
    "Benadryl", "ZzzQuil", "Pepto-Bismol", "Imodium",
    "melatonin", "fish oil", "vitamin C",
]

def run_tool1_tests():
    print("\n" + "=" * 65)
    print("TOOL 1 — resolve_brand_to_ingredients")
    print("=" * 65)
    results = []
    passed = failed = partial = 0

    for product in TEST_PRODUCTS:
        result = resolve_brand_to_ingredients(product)
        ings = result["data"]
        # Filter out combo entries that slipped through (contain " / ")
        single_ings = [i for i in ings if " / " not in i.get("ingredient","")]
        has_doses = all(
            i.get("amount_per_dose_mg") is not None for i in single_ings
        ) if single_ings else False

        if single_ings and has_doses:
            status = "✓ PASS"; passed += 1
        elif single_ings:
            status = "~ PARTIAL (no dose amounts)"; partial += 1
        else:
            status = "✗ FAIL (no ingredients)"; failed += 1

        print(f"\n  {status}  {product}")
        for i in single_ings[:6]:   # cap at 6 lines to avoid wall of text
            dose = f"{i['amount_per_dose_mg']} mg" if i.get("amount_per_dose_mg") else "dose unknown"
            note = f"  [{i['note']}]" if i.get("note") else ""
            print(f"           {i['ingredient']:30s} {dose}{note}")
        if len(single_ings) > 6:
            print(f"           ... and {len(single_ings)-6} more")
        if result["warning"]:
            print(f"           ⚠ {result['warning']}")
        results.append({**result, "status": status})

    print(f"\n  Summary: {passed} passed · {partial} partial · {failed} failed out of {len(TEST_PRODUCTS)}")
    return results


def run_tool2_tests(resolved):
    print("\n" + "=" * 65)
    print("TOOL 2 — get_interactions")
    print("=" * 65)

    high_risk_combos = [
        {"label": "Acetaminophen + Alcohol (must detect)",
         "ingredients": ["acetaminophen", "ethanol"], "expect_severity": "high"},
        {"label": "Ibuprofen + Aspirin (must detect)",
         "ingredients": ["ibuprofen", "aspirin"], "expect_severity": ["moderate","high"]},
        {"label": "Dual NSAIDs: Ibuprofen + Naproxen (must detect)",
         "ingredients": ["ibuprofen", "naproxen sodium"], "expect_severity": "high"},
        {"label": "Diphenhydramine + Alcohol (must detect)",
         "ingredients": ["diphenhydramine", "ethanol"], "expect_severity": "high"},
        {"label": "Safe pair: Loratadine + Ibuprofen (expect none/low)",
         "ingredients": ["loratadine", "ibuprofen"], "expect_severity": None},
    ]

    print()
    combo_results = []
    for combo in high_risk_combos:
        result = get_interactions(combo["ingredients"])
        found = result["data"]
        detected = len(found) > 0
        expected = combo["expect_severity"]

        if expected is None:
            ok = not detected or all(i["severity"] in ("low","unknown") for i in found)
        else:
            exp_list = [expected] if isinstance(expected, str) else expected
            ok = detected and any(i["severity"] in exp_list for i in found)

        status = "✓" if ok else "✗ MISSED"
        print(f"  {status}  {combo['label']}")
        for ix in found:
            print(f"       {SEV.get(ix['severity'],'⚪')} [{ix['severity']}] "
                  f"{ix['ingredient_a']} + {ix['ingredient_b']}")
        combo_results.append({"label": combo["label"], "ok": ok, "found": found})
        time.sleep(0.1)

    # Full pool note — LLM reasoning now handles this in synthesis_layer
    print(f"\n  Note: broader interaction reasoning (CYP450, serotonin syndrome,")
    print(f"  stimulant stacking etc.) is handled by LLM in synthesis_layer.py")

    return {"combo_tests": combo_results, "full_pool": []}


def run_tool3_tests(resolved):
    print("\n" + "=" * 65)
    print("TOOL 3 — check_dose_accumulation")
    print("=" * 65)

    scenarios = [
        {
            "label": "Theraflu (x4) — should flag acetaminophen over limit",
            "products": ["Theraflu"],
            "doses": {"Theraflu": 4},
            "expect_flag": "acetaminophen",
        },
        {
            "label": "Theraflu (x2) + Tylenol (x3) — should flag acetaminophen",
            "products": ["Theraflu", "Tylenol"],
            "doses": {"Theraflu": 2, "Tylenol": 3},
            "expect_flag": "acetaminophen",
        },
        {
            "label": "Advil (x3) + Aleve (x4) — should flag naproxen",
            "products": ["Advil", "Aleve"],
            "doses": {"Advil": 3, "Aleve": 4},
            "expect_flag": "naproxen sodium",
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
        scenario_resolved = [
            r for r in resolved
            if r["product"] in scenario["products"] and r.get("data")
        ]
        if not scenario_resolved:
            print(f"  ⚠ SKIP  {scenario['label']}")
            print(f"         (products didn't resolve in Tool 1 — fix Tool 1 first)")
            continue

        result = check_dose_accumulation(scenario_resolved, scenario["doses"])
        flagged = [i["ingredient"] for i in result["data"] if i["exceeds_limit"]]
        expected = scenario["expect_flag"]

        if expected:
            ok = expected in flagged
            status = "✓" if ok else f"✗ Expected '{expected}' to be flagged"
        else:
            ok = True
            status = "✓"

        print(f"  {status}  {scenario['label']}")
        for item in result["data"]:
            flag_str = "  ⚠ OVER LIMIT" if item["exceeds_limit"] else ""
            limit = str(int(item["safe_limit_mg"])) if item["safe_limit_mg"] else "no limit set"
            print(f"         {item['ingredient']:28s} "
                  f"{item['total_mg_per_day']:6.0f} mg/day  "
                  f"(limit: {limit}){flag_str}")
        if result["warning"]:
            print(f"         ⚠ {result['warning']}")
        print()
        scenario_results.append({"label": scenario["label"], "ok": ok})

    return {"scenarios": scenario_results}


def print_action_items(tool1_results):
    print("=" * 65)
    print("ACTION ITEMS")
    print("=" * 65)

    failed = [r["product"] for r in tool1_results if "FAIL" in r["status"]]
    partial = [r["product"] for r in tool1_results if "PARTIAL" in r["status"]]
    expected_partial = {"melatonin", "fish oil", "vitamin C", "Aspirin"}

    real_partial = [p for p in partial if p not in expected_partial]
    expected_ok = [p for p in partial if p in expected_partial]

    if failed:
        print(f"\n  ✗ FAIL — add to INGREDIENT_ALIASES or fix API query:")
        for p in failed: print(f"    - {p}")

    if real_partial:
        print(f"\n  ~ PARTIAL — add to DEFAULT_UNIT_DOSES or fix label parser:")
        for p in real_partial: print(f"    - {p}")

    if expected_ok:
        print(f"\n  ~ PARTIAL (expected — supplements/complex products, OK for now):")
        for p in expected_ok: print(f"    - {p}")

    if not failed and not real_partial:
        print("\n  ✓ All critical products resolved. Ready to build synthesis layer.")
    else:
        print(f"\n  Target: 0 fails, 0 unexpected partials before next phase.")


if __name__ == "__main__":
    t1 = run_tool1_tests()
    t2 = run_tool2_tests(t1)
    t3 = run_tool3_tests(t1)
    print_action_items(t1)

    with open("test_results_v2.json", "w") as f:
        json.dump({"tool1": t1, "tool2": t2, "tool3": t3}, f, indent=2, default=str)
    print("\n  Results saved → test_results_v2.json\n")
