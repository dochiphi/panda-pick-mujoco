"""Rerun every robustness test, original method vs current default, and write results.md."""
import re
import subprocess
import sys

BASE = ["--rotate", "--align"]
ORIGINAL = ["--rgb", "--topface", "none", "--no-blur", "--no-largest", "--no-grasp-check"]
TESTS = [
    ("Normal conditions", []),
    ("Camera noise, sigma = 40", ["--noise", "40"]),
    ("Camera noise, sigma = 60", ["--noise", "60"]),
    ("Dim light (x0.2)", ["--light", "0.2"]),
    ("Bright light (x2.5)", ["--light", "2.5"]),
    ("Blue / green / orange distractors", ["--distract"]),
    ("Second red cube, >= 10 cm away", ["--two-red"]),
    ("Second red cube, 10 mm gap", ["--two-red-at", "0.06"]),
    ("Second red cube, 2 mm gap", ["--two-red-at", "0.052"]),
]

def run(args):
    out = subprocess.run([sys.executable, "eval_pick.py", *BASE, *args],
                         capture_output=True, text=True).stdout
    s = re.search(r"Success rate: (\d+)/(\d+)", out)
    d = re.search(r"Detection error: mean (\S+) mm, max (\S+) mm", out)
    success = f"{round(100 * int(s.group(1)) / int(s.group(2)))}%"
    error = "cube not found" if d.group(1) == "nan" else f"{d.group(1)} / {d.group(2)} mm"
    return success, error

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
lines = [f"Results from commit `{commit}`: 50 trials each, random position and rotation, fixed seeds.",
         "Detection error is mean / max. Original = RGB thresholds, no blur, whole-image average, no grasp check.",
         "",
         "| Test | Original: success | Original: detection error | Current: success | Current: detection error |",
         "|---|---|---|---|---|"]
for name, args in TESTS:
    o_s, o_e = run(ORIGINAL + args)
    n_s, n_e = run(args)
    row = f"| {name} | {o_s} | {o_e} | {n_s} | {n_e} |"
    lines.append(row)
    print(row, flush=True)

open("results.md", "w").write("\n".join(lines) + "\n")
print("\nSaved results.md")
