# Agent Systems Foundations

Independent, research-extended study chapters on AI agents and their harnesses.

**[Read the HTML edition](https://az9713.github.io/agent-systems-foundations/)** · [Development journey](https://az9713.github.io/agent-systems-foundations/development-journey.html) · [Twelve-reference editorial study](https://az9713.github.io/agent-systems-foundations/textbook-style-report.html)

The first nine available lectures of [CMU 11-768: AI Agents](https://www.cmu-agents.com/) supplied more than a syllabus. Their recordings, captions, and slides informed the chapters' topic order, technical concepts, distinctions, and some examples. The chapters re-explain that material and add independently developed mathematical models, research, code, cases, and exercises. Sixty-four of 66 main sections link to relevant recording segments. This count measures navigation, not the fraction of chapter content taken from the lectures. The [reading guide](https://az9713.github.io/agent-systems-foundations/#sources) gives a chapter-by-chapter source map. These are **not official CMU course notes** and are not endorsed by Carnegie Mellon University or the instructors. Consult the original recordings for what was actually taught.

The nine chapters cover run states and authorization, tool contracts, long-context computation, memory, planning, coding agents, computer use, supervised fine-tuning, and reinforcement learning. They are aimed at advanced readers who have used agents and want to understand how agent systems are developed. Each chapter provides local definitions, derivations, worked cases, limitations, exercises, and solutions. A cumulative [Python harness](agent_lab/README.md) implements central equations and boundaries. The [development journey](DEVELOPMENT-JOURNEY.md) records the requests and corrections that shaped the first edition; the [style report](textbook-style-report.html) explains what was learned from twelve machine-learning references and two technical-writing guides.

The repository contains authored Markdown, standalone HTML, the standard-library companion harness and tests, a renderer, a validation script, and a small [source manifest](sources/manifest.json). It does **not** contain raw captions, transcript dumps, downloaded course slides, textbook PDFs, or private preview settings. Source recordings and research references are linked from the pages.

To regenerate and check the HTML with Python:

```sh
python -m pip install -r requirements.txt
python render_notes.py
python validate_notes.py
python -m agent_lab.order_demo
python -m unittest discover -s tests -v
```

The optional browser check uses a local Google Chrome installation through Playwright:

```sh
python browser_check.py
```

The nine videos, course slides, and linked books remain the works of their respective rights holders. Their publication online does not grant this repository permission to redistribute them. No license for the independently authored repository material is specified here.
