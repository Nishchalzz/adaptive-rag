"""Regenerates seed/graduate_regulations.pdf, the multi-page PDF in the demo corpus.

Run from backend/:  python -m scripts.build_seed_pdf
It reuses the minimal PDF writer from the test helpers, so no PDF library is needed.
"""

import textwrap
from pathlib import Path

from tests.helpers import make_pdf

PAGES = [
    (
        "HIT Graduate Study Regulations (2025 edition)",
        "Synthetic demonstration document. The institute and its rules are fictional.",
        "Section 1. Admission and credits",
        "1.1 A master's programme at the Halvorsen Institute of Technology comprises 120 ECTS credits and "
        "normally takes two years of full-time study.",
        "1.2 The master's thesis is worth 30 ECTS credits.",
        "1.3 Applicants must hold a bachelor's degree with at least 180 ECTS credits.",
        "1.4 Applications for the autumn intake must be submitted by 15 March.",
    ),
    (
        "Section 2. Thesis supervision",
        "2.1 Every master's thesis has a main supervisor who holds a doctorate and is employed by the institute.",
        "2.2 A thesis carried out in a research laboratory must also have a co-supervisor from that laboratory.",
        "2.3 A student may change supervisor once, with the approval of the dean of the school.",
        "2.4 Students meet their main supervisor at least every two weeks.",
    ),
    (
        "Section 3. Assessment",
        "3.1 Each thesis is graded by two examiners on a scale from 1 (lowest) to 5 (highest).",
        "3.2 If the two grades differ by more than one grade, the dean of the school appoints a third examiner.",
        "3.3 The minimum passing grade is 2.",
        "3.4 The thesis must be submitted no later than 31 May for graduation in June.",
        "3.5 A failed thesis may be resubmitted once, within six months of the decision.",
    ),
]


def main() -> None:
    pages = []
    for paragraphs in PAGES:
        lines: list[str] = []
        for paragraph in paragraphs:
            lines.extend(textwrap.wrap(paragraph, width=88))
            lines.append("")
        pages.append("\n".join(lines))
    target = Path(__file__).resolve().parents[1] / "seed" / "graduate_regulations.pdf"
    target.write_bytes(make_pdf(pages))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
