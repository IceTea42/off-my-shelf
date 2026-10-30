"""Record the screen-capture parts of the demo video, hands-free, in a real browser.

Drives the web app through the whole story on your own shelf media and saves one video per
viewport (phone and desktop) to ``--out``: upload -> live progress -> listing cards -> a
translation -> the agent acting on a spoken-style goal -> a checked buyer reply -> a video ad.
Every model call is live, so expect a few cents of API use per recording.

Once:  uv run playwright install chromium
Run:   uv run python scripts/record_demo.py shelf.mp4 --url https://icetea-off-my-shelf.hf.space
"""

import argparse
from pathlib import Path

from playwright.sync_api import Locator, Page, ViewportSize, sync_playwright

PHONE: ViewportSize = {"width": 412, "height": 915}  # Redmi-class portrait, CSS pixels
DESKTOP: ViewportSize = {"width": 1280, "height": 800}
RUN_TIMEOUT_MS = 10 * 60 * 1000  # a shelf of 10 items with shots and checks takes minutes
STEP_TIMEOUT_MS = 3 * 60 * 1000
GOAL = "I'm moving on Friday: sell the things worth more than 10 euros fast, donate the rest."
BUYER = "Hi! Would you take 3 euros for it?"


def pause(page: Page, seconds: float) -> None:
    page.wait_for_timeout(seconds * 1000)


def slow_type(field: Locator, text: str) -> None:
    field.click()
    field.press_sequentially(text, delay=45)  # readable on screen


def log_contains(page: Page, text: str, timeout: float = STEP_TIMEOUT_MS) -> None:
    page.locator("#log").get_by_text(text).last.wait_for(timeout=timeout)


def story(page: Page, media: list[Path], language: str) -> None:
    page.wait_for_selector("#language option", state="attached")
    pause(page, 2)
    page.select_option("#language", language)
    page.set_input_files("#files", [str(path) for path in media])
    pause(page, 1.5)
    page.click("#start")
    page.wait_for_selector(".card", timeout=RUN_TIMEOUT_MS)
    pause(page, 2)

    first = page.locator(".card:not(.lot)").first
    first.scroll_into_view_if_needed()
    pause(page, 3)
    other = "tr" if language != "tr" else "fr"
    first.locator(".chip", has_text=other.upper()).click()
    log_contains(page, "listing translated")
    pause(page, 3)

    goal = page.locator("#goal")
    goal.scroll_into_view_if_needed()
    slow_type(goal, GOAL)
    page.click("#run-agent")
    page.locator("#log .agent").last.wait_for(timeout=STEP_TIMEOUT_MS)
    page.locator("#log").scroll_into_view_if_needed()
    pause(page, 4)

    seller = page.locator(".card:not(.lot)").filter(has=page.locator(".badge.sell")).first
    seller.scroll_into_view_if_needed()
    slow_type(seller.locator("textarea"), BUYER)
    seller.get_by_role("button", name="Draft a reply").click()
    seller.locator(".reply").last.wait_for(timeout=STEP_TIMEOUT_MS)
    seller.locator(".reply").last.scroll_into_view_if_needed()
    pause(page, 4)

    seller.get_by_role("button", name="Video ad").click()
    video = seller.locator("video")
    video.wait_for(timeout=STEP_TIMEOUT_MS)
    video.scroll_into_view_if_needed()
    video.evaluate("v => { v.muted = true; return v.play(); }")
    pause(page, 11)


def record(url: str, media: list[Path], out: Path, language: str, phone: bool) -> Path:
    viewport = PHONE if phone else DESKTOP
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport=viewport,
            device_scale_factor=2 if phone else 1,
            is_mobile=phone,
            has_touch=phone,
            record_video_dir=str(out),
            record_video_size=viewport,
        )
        page = context.new_page()
        page.goto(url)
        story(page, media, language)
        if page.video is None:
            raise RuntimeError("the browser recorded no video")
        target = out / f"demo-{'phone' if phone else 'desktop'}.webm"
        context.close()  # the video file is complete only once its context closes
        page.video.save_as(target)
        page.video.delete()  # keep only the named copy
        browser.close()
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("media", nargs="+", type=Path, help="shelf photos or a short video")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="app address")
    parser.add_argument("--out", type=Path, default=Path("recordings"))
    parser.add_argument("--lang", default="en", choices=["en", "fr", "tr"])
    parser.add_argument("--only", choices=["phone", "desktop"], help="record one viewport")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for phone in (True, False):
        if args.only in (None, "phone" if phone else "desktop"):
            print(f"recorded {record(args.url, args.media, args.out, args.lang, phone)}")


if __name__ == "__main__":
    main()
