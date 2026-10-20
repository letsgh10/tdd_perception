"""Raport z testów mutacyjnych (mutmut) w Markdown: wynik + diffy ocalałych mutantów.

Użycie (po `mutmut run`): python scripts/mutation_report.py [--min-score 90]
Markdown trafia na stdout (w CI dopisywany do $GITHUB_STEP_SUMMARY).
"""

import argparse
import re
import subprocess
import sys
from collections import Counter

# mutanty ocalałe z wyjaśnieniem (równoważne: nie zmieniają zachowania programu)
SURVIVOR_NOTES: dict[str, str] = {
    "x_draw__mutmut_30": "font `None` = domyślny `FONT_HERSHEY_SIMPLEX` (0) w OpenCV",
    "x_draw__mutmut_49": "przy skali 0.5 grubość tekstu 2 i 3 daje te same piksele",
    "x_draw_tracks__mutmut_33": "polilinia z jednego punktu nic nie rysuje",
    "x_draw_tracks__mutmut_37": "`isClosed=None` jest traktowane jak `False`",
    "x_draw_tracks__mutmut_46": "`dtype=None` daje int64 (OpenCV akceptuje)",
    "x_draw_tracks__mutmut_48": "j.w. — domyślny dtype to int64, wynik identyczny",
    "x_write_video__mutmut_29": "`MP4V` = `mp4v`: kontener i tak raportuje `FMP4`",
    "x_main__mutmut_110": "bez `default=None` argparse i tak daje `None`",
}


def _mutmut(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "mutmut", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


def _diff(name: str) -> str:
    lines = _mutmut("show", name).splitlines()
    changed = (ln for ln in lines if re.match(r"^[-+]", ln))
    return "\n".join(ln for ln in changed if not ln.startswith(("---", "+++")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, default=0.0, help="próg w %% (0 = brak)")
    args = ap.parse_args()

    rows = [
        ln.strip().rsplit(": ", 1)
        for ln in _mutmut("results", "--all", "true").splitlines()
        if ": " in ln
    ]
    if not rows:
        print("Brak wyników mutmut — czy uruchomiono `mutmut run`?", file=sys.stderr)
        return 2
    counts = Counter(status for _, status in rows)
    total = len(rows)
    detected = counts["killed"] + counts["timeout"]
    score = 100 * detected / total

    print("## Testy mutacyjne `perception/cli.py`\n")
    print(f"**Mutation score: {score:.1f}%** ({detected}/{total} wykrytych)\n")
    print("| Status | Liczba |\n|---|---:|")
    for status, n in counts.most_common():
        print(f"| {status} | {n} |")

    survivors = [name for name, status in rows if status == "survived"]
    if survivors:
        print(f"\n### Ocalałe mutanty ({len(survivors)})\n")
        for name in survivors:
            short = name.removeprefix("perception.cli.")
            note = SURVIVOR_NOTES.get(short)
            print(f"<details><summary><code>{short}</code>")
            print(f"{'— ' + note if note else '— **do analizy**'}</summary>\n")
            print(f"```diff\n{_diff(name)}\n```\n</details>\n")

    if score < args.min_score:
        print(
            f"\n**Wynik {score:.1f}% poniżej progu {args.min_score:.1f}%**",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
