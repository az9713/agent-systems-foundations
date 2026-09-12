# Agent Systems Foundations

Independent, research-extended study chapters on AI agents and their harnesses.

**[Read the HTML edition](https://az9713.github.io/agent-systems-foundations/)** · [Development journey](https://az9713.github.io/agent-systems-foundations/development-journey.html) · [Twelve-reference editorial study](https://az9713.github.io/agent-systems-foundations/textbook-style-report.html)

The first four public lectures of [CMU 11-768: AI Agents](https://www.cmu-agents.com/) supplied the topic sequence and the timestamps that link each major section to the [source playlist](https://www.youtube.com/playlist?list=PLSN0qpDfUvTM). These chapters are independently written extensions. Their mathematical models, examples, comparisons, and exercises may go beyond or differ from the recordings. They are **not official CMU course notes** and are not endorsed by Carnegie Mellon University or the instructors. Consult the original recordings for what was actually taught.

The four chapters cover agent run states and authorization, tool contracts and effects, long-context computation and compaction, and cross-task memory and skills. They are aimed at advanced readers who have used agents and want to understand how agent systems are developed. Each chapter provides local definitions, derivations, worked cases, limitations, exercises, and solutions. The [development journey](DEVELOPMENT-JOURNEY.md) records the requests and corrections that shaped them; the [style report](textbook-style-report.html) explains what was learned from twelve machine-learning references.

The repository contains authored Markdown, standalone HTML, a renderer, a validation script, and a small [source manifest](sources/manifest.json). It does **not** contain raw captions, transcript dumps, downloaded course slides, textbook PDFs, or private preview settings. Source recordings and research references are linked from the pages.

To regenerate and check the HTML with Python:

```sh
python -m pip install -r requirements.txt
python render_notes.py
python validate_notes.py
```

The optional browser check requires a Chromium installation for Playwright:

```sh
python -m playwright install chromium
python browser_check.py
```

The four videos, course slides, and linked books remain the works of their respective rights holders. Their publication online does not grant this repository permission to redistribute them. No license for the independently authored repository material is specified here.
