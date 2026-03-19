"""
OTC Self-Medication Agent — Interactive CLI
============================================
Type anything and get a full risk report.

Usage:
    PYTHONPATH=src python src/run_agent.py

Commands during session:
    quit / exit / q  — exit
    history          — show all queries this session
    clear            — clear session history
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from synthesis_layer import run_from_text, print_report


# ─────────────────────────────────────────────────────────────────────────────

def run_interactive():
    session_history = []

    print("\n" + "═" * 60)
    print("  OTC Self-Medication Risk Agent")
    print("  Phase 1 — Text Input")
    print("═" * 60)
    print("  Describe what medications you take, ask if a")
    print("  combination is safe, or ask about a product.")
    print()
    print("  Examples:")
    print("    I take NyQuil and Tylenol every night")
    print("    Can I take Advil and Aleve together?")
    print("    I use Benadryl to sleep and drink wine most nights")
    print()
    print("  Type 'quit' to exit, 'history' to review past queries.")
    print("═" * 60)

    while True:
        try:
            print()
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting.")
            break

        if not user_input:
            continue

        # ── Commands ─────────────────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nExiting. Stay safe with your medications.")
            break

        if user_input.lower() == "history":
            if not session_history:
                print("\n  No queries yet this session.")
            else:
                print(f"\n  Session history ({len(session_history)} queries):")
                for i, h in enumerate(session_history, 1):
                    risk = h["risk_level"].upper() if h["risk_level"] else "ERROR"
                    icon = {"HIGH":"🔴","MODERATE":"🟡","LOW":"🟢",
                            "NONE":"✅"}.get(risk, "⚪")
                    print(f"  {i}. {icon} [{risk}]  \"{h['input'][:55]}\"")
            continue

        if user_input.lower() == "clear":
            session_history = []
            print("  Session history cleared.")
            continue

        # ── Run pipeline ──────────────────────────────────────────────────────
        print()
        report = run_from_text(user_input)

        # Print the report
        print_report(report)

        # Save to history
        output = report.get("output") or {}
        session_history.append({
            "input":      user_input,
            "risk_level": output.get("risk_level"),
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        })

        # Optionally save full report to file
        safe_name = "".join(
            c if c.isalnum() else "_"
            for c in user_input[:30]
        ).strip("_")
        fname = f"report_{safe_name}.json"
        with open(fname, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved → {fname}")


if __name__ == "__main__":
    run_interactive()
