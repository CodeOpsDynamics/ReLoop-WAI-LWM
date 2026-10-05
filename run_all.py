"""Reproduce every number, table and figure in the ReLoop WAI report:  python run_all.py"""
import subprocess, sys, pathlib
src = pathlib.Path(__file__).parent / "src"
(pathlib.Path(__file__).parent / "outputs" / "results.json").unlink(missing_ok=True)
for step in ["a01_real_diagnostics", "a01b_state_typology", "a02_forecast", "a03_disposition_ml", "a04_network_design",
             "a05_milkrun_vrp", "a06_fuzzy_fmea", "a07_cost_benefit", "a08_flowchart_gantt", "a09_build_dashboard"]:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, f"{step}.py"], cwd=src, check=True)
print("\nAll outputs written to outputs/figures, outputs/tables and outputs/results.json")
