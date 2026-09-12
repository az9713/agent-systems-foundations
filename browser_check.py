"""Check the rendered lecture pages, including MathJax, in Chromium."""

from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for file in [ROOT / "index.html", ROOT / "development-journey.html", *sorted(ROOT.glob("lecture-*.html"))]:
            page.goto(file.as_uri(), wait_until="domcontentloaded", timeout=30000)
            if file.name.startswith("lecture-"):
                page.wait_for_function(
                    "document.querySelectorAll('mjx-container').length >= 4",
                    timeout=30000,
                )
                errors = page.locator("mjx-merror").count()
                assert errors == 0, (file.name, page.locator("mjx-merror").all_text_contents())
                assert page.locator("mjx-container").count() >= 4
            assert page.locator("h1").count() == 1
            assert page.locator("a[href='index.html']").count() >= 1
            print(file.name, "rendered", "math", page.locator("mjx-container").count())
        browser.close()


if __name__ == "__main__":
    main()
