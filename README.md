# Agent Systems Foundations

Independent, research-extended study chapters on AI agents and their harnesses.

**[Read the HTML edition](https://az9713.github.io/agent-systems-foundations/)** · [Development journey](https://az9713.github.io/agent-systems-foundations/development-journey.html) · [Twelve-reference editorial study](https://az9713.github.io/agent-systems-foundations/textbook-style-report.html)

The first four public lectures of [CMU 11-768: AI Agents](https://www.cmu-agents.com/) supplied more than a syllabus. Their recordings, captions, and slides informed the chapters' topic order, technical concepts, distinctions, and some examples. For instance, Chapter 2 develops the [lecture's grocery-cart example](https://www.youtube.com/watch?v=jXChFB4JSyw&t=596s), while Chapter 3 develops its discussions of [key/value caching](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=3150s) and [compaction](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=4005s). The chapters re-explain that material and add independently developed mathematical models, research, code, cases, and exercises. Twenty-five of 27 numbered main sections link to relevant recording segments; this measures navigation, not the fraction of chapter content taken from the lectures. The [reading guide](https://az9713.github.io/agent-systems-foundations/#sources) gives a chapter-by-chapter source map and explains how a content-share audit would work. These are **not official CMU course notes** and are not endorsed by Carnegie Mellon University or the instructors. Consult the original recordings for what was actually taught.

The four chapters cover agent run states and authorization, tool contracts and effects, long-context computation and compaction, and cross-task memory and skills. They are aimed at advanced readers who have used agents and want to understand how agent systems are developed. Each chapter provides local definitions, derivations, worked cases, limitations, exercises, and solutions. A cumulative [Python harness](agent_lab/README.md) implements the equations and boundaries in a runnable order-support example. The [development journey](DEVELOPMENT-JOURNEY.md) records the requests and corrections that shaped the chapters; the [style report](textbook-style-report.html) explains what was learned from twelve machine-learning references.

The repository contains authored Markdown, standalone HTML, the standard-library companion harness and tests, a renderer, a validation script, and a small [source manifest](sources/manifest.json). It does **not** contain raw captions, transcript dumps, downloaded course slides, textbook PDFs, or private preview settings. Source recordings and research references are linked from the pages.

To regenerate and check the HTML with Python:

```sh
python -m pip install -r requirements.txt
python render_notes.py
python validate_notes.py
python -m agent_lab.order_demo
python -m unittest discover -s tests -v
```

The optional browser check requires a Chromium installation for Playwright:

```sh
python -m playwright install chromium
python browser_check.py
```

The four videos, course slides, and linked books remain the works of their respective rights holders. Their publication online does not grant this repository permission to redistribute them. No license for the independently authored repository material is specified here.
