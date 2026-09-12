"""Render the four authored Markdown notes as a local HTML reading set."""

from pathlib import Path
import html
import re

import markdown
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
NOTES = [
    ("lecture-01-agents", "What are agents?", "Run state, autonomy, authority, verification, and production harnesses"),
    ("lecture-02-tool-use", "Tool use", "Tool contracts, effects, capabilities, retries, and concurrency"),
    ("lecture-03-long-context", "Long context", "Attention, context selection, prompt economics, and compaction"),
    ("lecture-04-memory-and-skills", "Memory and skills", "Cross-task value, provenance, retrieval, and skill contracts"),
]

STYLE = """
:root {color-scheme:dark;--bg:#0c1018;--panel:#151d2a;--line:#2c3b4f;--text:#e8edf4;--muted:#aab9cc;--accent:#86c5ff;--warm:#ffd395}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--text);font:17px/1.68 system-ui,Segoe UI,sans-serif}
a{color:var(--accent);text-underline-offset:3px}a:hover{color:#b5dcff}.shell{max-width:1180px;margin:auto;padding:0 24px}.top{border-bottom:1px solid var(--line);background:#101722;position:sticky;top:0;z-index:2}.top .shell{display:flex;align-items:center;justify-content:space-between;gap:20px;padding-top:12px;padding-bottom:12px}.brand{font-weight:800;color:var(--text);text-decoration:none}.top nav{display:flex;gap:14px;flex-wrap:wrap;font-size:.87rem}.top nav a{text-decoration:none}.layout{display:grid;grid-template-columns:minmax(0,780px) 260px;gap:44px;align-items:start}.main{min-width:0;padding:46px 0 100px}.side{position:sticky;top:76px;max-height:calc(100vh - 90px);overflow:auto;padding:35px 0 20px;font-size:.83rem}.side strong{display:block;color:var(--warm);letter-spacing:.08em;text-transform:uppercase}.side ul{list-style:none;padding:0}.side li{margin:9px 0}.side a{text-decoration:none;color:var(--muted)}.side a:hover{color:var(--accent)}h1,h2,h3{line-height:1.2;letter-spacing:-.025em}h1{font-size:clamp(2.2rem,4vw,3.7rem);margin:0 0 22px}h2{font-size:1.7rem;margin:68px 0 18px;border-top:1px solid var(--line);padding-top:25px;scroll-margin-top:80px}h2 a[href*="youtube.com/watch"]{font-size:.58em;font-weight:650;letter-spacing:0;white-space:nowrap;vertical-align:middle}h3{font-size:1.2rem;color:var(--warm);margin:38px 0 12px;scroll-margin-top:80px}p{margin:0 0 1.15em}p:first-of-type{color:var(--muted)}strong{color:#fff}blockquote{border-left:3px solid var(--accent);padding:10px 18px;background:var(--panel);margin:24px 0}table{border-collapse:collapse;width:100%;margin:24px 0 34px;display:block;overflow-x:auto;font-size:.94rem}th,td{border:1px solid var(--line);padding:12px 13px;text-align:left;vertical-align:top}th{background:#192536;color:var(--warm)}tr:nth-child(even) td{background:#121b27}pre{background:#111a27;border:1px solid var(--line);border-radius:10px;padding:20px;overflow-x:auto;color:#d4e8ff;font:14px/1.5 Consolas,monospace}code{font-family:Consolas,monospace;background:#1b293a;padding:1px 4px;border-radius:3px}pre code{background:none;padding:0}ol,ul{padding-left:1.6em}li{margin-bottom:.55em}.MathJax_Display{overflow-x:auto}.hero{background:linear-gradient(135deg,#16243a,#111925);border:1px solid var(--line);border-radius:16px;padding:25px 30px;margin:36px 0}.eyebrow{font-size:.78rem;font-weight:800;color:var(--warm);letter-spacing:.12em;text-transform:uppercase}.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:28px}.card{display:block;background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:22px;text-decoration:none;color:var(--text)}.card:hover{border-color:var(--accent);transform:translateY(-2px)}.card b{font-size:1.15rem}.card span{display:block;color:var(--muted);margin-top:8px}.tiny{color:var(--muted);font-size:.85rem}.foot{border-top:1px solid var(--line);padding:24px 0 50px;color:var(--muted);font-size:.85rem}@media(max-width:900px){.layout{display:block}.side{display:none}.main{padding-top:32px}.cards{grid-template-columns:1fr}.top nav{display:none}}@media print{:root{color-scheme:light;--bg:#fff;--panel:#f5f5f5;--line:#ccc;--text:#111;--muted:#333;--accent:#064d8b;--warm:#704600}body{font-size:11pt}.top,.side{display:none}.shell{max-width:none;padding:0}.main{padding:0}h2{break-after:avoid}pre,table{break-inside:avoid}.card{break-inside:avoid}}
.chapter-nav{display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:center;border-block:1px solid var(--line);padding:18px 0;margin:0 0 42px;font-size:.93rem}.chapter-nav:last-child{margin:55px 0 0}.chapter-nav a{display:block;text-decoration:none}.chapter-nav .next{text-align:right}.chapter-nav .index{color:var(--muted)}@media(max-width:600px){.chapter-nav{grid-template-columns:1fr;gap:8px}.chapter-nav .next{text-align:left}}@media print{.chapter-nav{display:none}}
"""


def math_protected(src):
    saved = []

    def hold(match):
        saved.append(match.group(0))
        return f"MATHPLACEHOLDER{len(saved)-1}END"

    # No authored math occurs in fenced code; the only fenced example with a
    # dollar sign would be retained as literal by the Markdown renderer.
    protected = re.sub(r"\$\$[\s\S]*?\$\$|(?<!\$)\$(?!\$)[^\n$]+\$(?!\$)", hold, src)
    return protected, saved


def render_body(src):
    protected, saved = math_protected(src)
    body = markdown.markdown(
        protected, extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5"
    )
    for i, expression in enumerate(saved):
        body = body.replace(f"MATHPLACEHOLDER{i}END", html.escape(expression))
    return body


def page(title, body, sidebar=""):
    nav = "".join(
        f'<a href="{slug}.html">{html.escape(short)}</a>'
        for slug, short, _ in NOTES
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} | Agent Systems Foundations</title>
<style>{STYLE}</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script defer src="assets/mathjax-tex-svg.js"></script></head>
<body><header class="top"><div class="shell"><a class="brand" href="index.html">Agent Systems Foundations</a><nav>{nav}<a href="development-journey.html">Development journey</a><a href="textbook-style-report.html">Style study</a></nav></div></header>
<div class="shell layout"><main class="main">{body}</main><aside class="side"><strong>On this page</strong>{sidebar}</aside></div>
<footer class="foot shell">Independent study material · <a href="index.html#sources">Course attribution and scope</a></footer></body></html>"""


def sidebar_for(body):
    soup = BeautifulSoup(body, "html.parser")
    items = [
        (heading["id"], re.sub(r"\s*\d{2}:\d{2}$", "", heading.get_text(" ", strip=True)))
        for heading in soup.find_all("h2")
    ]
    return "<ul>" + "".join(
        f'<li><a href="#{html.escape(anchor)}">{text}</a></li>'
        for anchor, text in items
    ) + "</ul>"


def main():
    for i, (slug, short, _) in enumerate(NOTES):
        source = (ROOT / f"{slug}.md").read_text(encoding="utf-8")
        body = render_body(source)
        previous = NOTES[i - 1] if i else None
        following = NOTES[i + 1] if i + 1 < len(NOTES) else None
        pager = (
            '<nav class="chapter-nav" aria-label="Chapter navigation">'
            + (f'<a class="previous" href="{previous[0]}.html">← Previous: {html.escape(previous[1])}</a>'
               if previous else '<span></span>')
            + '<a class="index" href="index.html">Reading guide</a>'
            + (f'<a class="next" href="{following[0]}.html">Next: {html.escape(following[1])} →</a>'
               if following else '<span></span>')
            + '</nav>'
        )
        (ROOT / f"{slug}.html").write_text(
            page(short, pager + body + pager, sidebar_for(body)), encoding="utf-8"
        )
    cards = "".join(
        f'<a class="card" href="{slug}.html"><div class="eyebrow">Chapter {i}</div>'
        f'<b>{html.escape(short)}</b><span>{html.escape(description)}</span></a>'
        for i, (slug, short, description) in enumerate(NOTES, 1)
    )
    index_body = f"""<div class="eyebrow">Independent study · Four research-extended chapters</div>
<h1>AI agents, from action loops to reusable experience</h1>
<p>These independent chapters develop agent architecture, tool use, long-context computation, and cross-task memory for advanced graduate readers who have used agents but have not built them. A common run-state model connects mathematical definitions to authorization, execution, context selection, persistent memory, and verification. Derivations, production-system examples, limitations, and exercises extend far beyond the recorded lectures.</p>
<div class="hero"><b>Reading sequence</b><p>Begin with the agent interaction loop, then study its action interface, its growing context, and its ability to reuse experience. A timestamp beside a section title opens the corresponding source segment. Research findings and primary-source citations are integrated into the exposition.</p></div>
<div class="cards">{cards}</div>
<h2 id="sources">Role of CMU 11-768</h2>
<p>The first four public lectures of <a href="https://www.cmu-agents.com/">CMU 11-768: AI Agents</a> supplied the topic sequence and the video segments linked from section headings. The <a href="https://www.youtube.com/playlist?list=PLSN0qpDfUvTM">course playlist</a> and official course site are the authoritative sources for what the instructors actually taught. These chapters are independently written research extensions. Their formal models, derivations, examples, engineering comparisons, and exercises may go beyond or differ from the lectures. They are not official course notes and are not endorsed by Carnegie Mellon University or the instructors.</p>
<p>The <a href="sources/manifest.json">source manifest</a> records the four recordings used for this edition as of 12 September 2026. English captions were automatically generated, so technical terminology was checked against course slides and primary research sources. Captions and slide PDFs are not redistributed here. Later course topics are outside this set.</p>
<p>The implementation examples draw on official documentation from <a href="https://www.anthropic.com/engineering/managed-agents">Anthropic</a>, <a href="https://openai.com/index/unrolling-the-codex-agent-loop/">OpenAI</a>, <a href="https://docs.cursor.com/context/rules-for-ai">Cursor</a>, <a href="https://hermes-agent.nousresearch.com/docs/guides/tips">Hermes Agent</a>, and <a href="https://docs.openclaw.ai/concepts/agent-loop">OpenClaw</a>, checked on 12 September 2026. They illustrate architectural choices and are not comparative benchmarks.</p>
<h2 id="project">How the material was developed</h2>
<p>The <a href="development-journey.html">development journey</a> records the requests, revisions, failures, and checks that shaped the four chapters. The <a href="textbook-style-report.html">12-reference style study</a> explains the editorial methods adopted from machine-learning textbooks and technical papers.</p>
<p class="tiny">Editable text versions: <a href="lecture-01-agents.md">Chapter 1</a> · <a href="lecture-02-tool-use.md">Chapter 2</a> · <a href="lecture-03-long-context.md">Chapter 3</a> · <a href="lecture-04-memory-and-skills.md">Chapter 4</a>.</p>"""
    (ROOT / "index.html").write_text(
        page("Reading guide", index_body,
             '<ul><li><a href="#sources">Course attribution and scope</a></li><li><a href="#project">Development and style</a></li></ul>'),
        encoding="utf-8"
    )
    journey_file = ROOT / "DEVELOPMENT-JOURNEY.md"
    if journey_file.exists():
        journey_body = render_body(journey_file.read_text(encoding="utf-8"))
        (ROOT / "development-journey.html").write_text(
            page("Development journey", journey_body, sidebar_for(journey_body)),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
