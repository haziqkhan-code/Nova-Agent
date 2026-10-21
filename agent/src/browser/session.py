"""Browser lifecycle for one agent job.

LiveKit runs each job in its own process, so a module-level manager is genuinely
per-session: there is no way for one caller's cookies or pages to reach another's. The
manager owns exactly one Chromium instance, one context, and one page.

Three decisions worth stating:

*Lazy launch.* Chromium costs seconds to start and most calls never browse, so the
browser is not started until a tool actually asks for it. Launching in the job
entrypoint would put that cost on the critical path of every conversation.

*A single lock.* Playwright objects are not safe to drive from several coroutines at
once. Every public method takes the lock for the whole operation.

*Page-derived state is untrusted.* Nothing read off the page is returned to a model
without going through :mod:`browser.extract`, and no page content ever reaches a
tool argument that could cause an effect without passing the confirmation gate.
"""

import asyncio
import contextlib
import json
import os
from typing import Any

from livekit.agents import inference, llm
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout

from .helpers import planner_model
from .policy import check_url

_NAVIGATION_TIMEOUT_MS = 20_000
_ACTION_TIMEOUT_MS = 10_000

# Chromium's sandbox works when the process is not root, which is how the container
# runs the agent, so it is left on. --disable-dev-shm-usage is still required because
# Docker's /dev/shm is 64MB by default and Chromium exhausts it on ordinary pages.
_CHROMIUM_ARGS = ["--disable-dev-shm-usage"]

if os.getenv("NOVA_BROWSER_NO_SANDBOX") == "1":
    # Escape hatch for local runs on a host that forbids user namespaces, where
    # Chromium exits at startup with a sandbox error. Never set this in production:
    # it removes the process isolation that keeps a malicious page away from the host.
    _CHROMIUM_ARGS.append("--no-sandbox")


class BrowserError(Exception):
    """A browser operation failed in a way that is safe to show the model."""


class BrowserManager:
    """Owns the Playwright objects for a single job."""

    def __init__(self) -> None:
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def is_running(self) -> bool:
        return self._browser is not None and self._page is not None

    async def _ensure_started(self) -> Any:
        if self._closed:
            raise BrowserError("The browser has already been closed for this call.")

        if self._page is not None:
            return self._page

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=_CHROMIUM_ARGS,
            )
            self._context = await self._browser.new_context(
                locale="en-GB",
                viewport={"width": 1280, "height": 800},
            )
            self._page = await self._context.new_page()
            self._page.set_default_timeout(_ACTION_TIMEOUT_MS)
        except PlaywrightError as err:
            await self._teardown()
            raise BrowserError(
                "I was unable to start my browser. Please try that again in a moment."
            ) from err

        return self._page

    async def open(self, url: str) -> str:
        """Navigate to a policy-checked URL and return the resulting page text."""
        async with self._lock:
            target = check_url(url)
            page = await self._ensure_started()
            try:
                response = await page.goto(
                    target,
                    timeout=_NAVIGATION_TIMEOUT_MS,
                    wait_until="domcontentloaded",
                )
            except PlaywrightTimeout as err:
                raise BrowserError(
                    f"{target} took too long to respond. It may be down or very slow."
                ) from err
            except PlaywrightError as err:
                raise BrowserError(f"I could not open {target}.") from err

            if response is not None and response.status >= 400:
                raise BrowserError(
                    f"{target} returned an error, status {response.status}."
                )

            from .extract import readable_text

            return await readable_text(page)

    async def current_page_text(self) -> str:
        """Return the current page's text, requiring a browser to already exist."""
        async with self._lock:
            if self._page is None:
                raise BrowserError("I have not opened a page yet.")
            from .extract import readable_text

            return await readable_text(self._page)

    async def current_url(self) -> str:
        async with self._lock:
            if self._page is None:
                raise BrowserError("I have not opened a page yet.")
            return self._page.url

    async def run_action(self, instruction: str) -> str:
        """Have a model plan a single browser action, then execute it.

        Planning and execution are separate model calls so the plan can be shown to the
        user for confirmation *before* anything happens. Only one action is executed
        per call: a chain of actions would defeat the point of confirming.
        """
        async with self._lock:
            if self._page is None:
                raise BrowserError("I have not opened a page yet.")

            from .extract import readable_text

            page_text = await readable_text(self._page)
            plan = await self._plan_action(instruction, page_text)

            kind = plan.get("action")
            if kind == "click":
                await self._click_target(str(plan.get("target", "")))
            elif kind == "type":
                await self._type_target(
                    str(plan.get("target", "")), str(plan.get("text", ""))
                )
            elif kind == "press_key":
                await self._press_key(str(plan.get("key", "")))
            elif kind == "scroll":
                await self._scroll(str(plan.get("direction", "down")))
            else:
                raise BrowserError(
                    f"I was not able to work out how to {kind or 'do that'}."
                )

            return await readable_text(self._page)

    async def preview_action(self, instruction: str) -> str:
        """Return what an action would do, without performing it."""
        async with self._lock:
            if self._page is None:
                raise BrowserError("I have not opened a page yet.")

            from .extract import readable_text

            plan = await self._plan_action(instruction, await readable_text(self._page))
            return _describe_plan(plan)

    async def screenshot(self) -> bytes:
        async with self._lock:
            if self._page is None:
                raise BrowserError("I have not opened a page yet.")
            try:
                return await self._page.screenshot(
                    type="png", full_page=False, timeout=_ACTION_TIMEOUT_MS
                )
            except PlaywrightError as err:
                raise BrowserError("I was unable to capture the page.") from err

    async def _plan_action(self, instruction: str, page_text: str) -> dict[str, str]:
        """Ask a text-only model to map a spoken instruction onto a page control.

        Running this on a cheap model keeps the realtime voice model out of a large
        text-only prompt, and keeps the choice of control deterministic enough to show
        the user before acting.
        """
        system = (
            "You turn a spoken instruction into a single browser action for the page "
            "below. Reply with JSON only, no prose and no code fences, with these keys: "
            'action (one of "click", "type", "press_key", "scroll"), target (the exact '
            "visible name or label of the control to use, empty for press_key and "
            'scroll), text (what to type, only for "type"), key (only for press_key), '
            'direction ("up" or "down", only for scroll). Choose a target that appears '
            "verbatim on the page. Never target a password field, a payment or card "
            "field, or a field asking for a one-time code. If no control on the page "
            'matches, reply {"action": "none"}.'
        )

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(role="system", content=system)
        chat_ctx.add_message(
            role="user",
            content=(
                f"Instruction:\n{instruction}\n\n"
                f"Current page ({self._page.url if self._page else 'unknown'}):\n"
                f"{page_text[:8_000]}"
            ),
        )

        try:
            async with inference.LLM(model=planner_model()) as planner:
                response = await planner.chat(chat_ctx=chat_ctx).collect()
        except Exception as err:
            raise BrowserError(
                "I was unable to plan that action. Please try rephrasing it."
            ) from err

        plan = _parse_plan(response.text or "")

        # The prompt tells the planner to avoid password and payment fields, and that
        # is not sufficient on its own: a planner asked to "type hunter2 into the
        # password box" has been observed returning a plan that does exactly that.
        # Re-check the result here so an instruction the model failed to refuse is
        # caught before it is described to the user or handed to Playwright.
        target = plan.get("target", "")
        if target and plan.get("action") in {"click", "type"}:
            with contextlib.suppress(BrowserError):
                _assert_not_sensitive(target, action="use")
                return plan
            return {"action": "none"}

        return plan

    async def _click_target(self, target: str) -> None:
        if not target:
            raise BrowserError("I could not tell which control you meant.")
        _assert_not_sensitive(target, action="click")
        locator = await self._resolve(target)
        try:
            await locator.click(timeout=_ACTION_TIMEOUT_MS)
        except PlaywrightTimeout as err:
            raise BrowserError(f"I could not find anything called {target!r}.") from err
        except PlaywrightError as err:
            raise BrowserError(f"I was unable to click {target!r}.") from err
        await _settle()

    async def _type_target(self, target: str, text: str) -> None:
        if not target:
            raise BrowserError("I could not tell which field you meant.")
        _assert_not_sensitive(target, action="fill in")
        locator = await self._resolve(target)
        try:
            await locator.fill(text, timeout=_ACTION_TIMEOUT_MS)
        except PlaywrightTimeout as err:
            raise BrowserError(f"I could not find a field called {target!r}.") from err
        except PlaywrightError as err:
            raise BrowserError(f"I was unable to fill in {target!r}.") from err

    async def _press_key(self, key: str) -> None:
        allowed = {"Enter", "Escape", "Tab", "Backspace", "ArrowUp", "ArrowDown"}
        if key not in allowed:
            raise BrowserError(
                "I can only press Enter, Escape, Tab, Backspace, or an arrow key."
            )
        try:
            await self._page.keyboard.press(key)
        except PlaywrightError as err:
            raise BrowserError(f"I was unable to press {key}.") from err
        await _settle()

    async def _scroll(self, direction: str) -> None:
        if direction not in {"up", "down"}:
            raise BrowserError("I can only scroll up or down.")
        delta = -600 if direction == "up" else 600
        try:
            await self._page.mouse.wheel(0, delta)
        except PlaywrightError as err:
            raise BrowserError("I was unable to scroll the page.") from err
        await _settle()

    async def _resolve(self, target: str) -> Any:
        """Find a control by its visible name, trying the common accessibility roles.

        Voice callers say "the Search button", not a CSS selector, so resolution is by
        accessible name with a text fallback.
        """
        assert self._page is not None
        for role in ("link", "button", "textbox", "combobox", "checkbox"):
            locator = self._page.get_by_role(role, name=target, exact=False)
            # count() is a coroutine, so it must be awaited. Comparing the coroutine
            # object itself is always truthy, which would hand back a locator for a
            # role that does not exist on the page.
            if await locator.count() > 0:
                return locator.first
        by_text = self._page.get_by_text(target, exact=False)
        if await by_text.count() > 0:
            return by_text.first
        return self._page.get_by_label(target, exact=False).first

    async def _teardown(self) -> None:
        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._playwright, "stop", None),
        ):
            if closer is None:
                continue
            with contextlib.suppress(Exception):
                await closer()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def aclose(self) -> None:
        self._closed = True
        async with self._lock:
            await self._teardown()


_SENSITIVE_FIELDS = (
    "password",
    "passwd",
    "card number",
    "cvv",
    "cvc",
    "security code",
    "one-time code",
    "otp",
    "verification code",
    "pin",
)


def _assert_not_sensitive(target: str, *, action: str) -> None:
    lowered = target.casefold()
    if any(marker in lowered for marker in _SENSITIVE_FIELDS):
        raise BrowserError(
            f"I will not {action} a field that looks like a password, card, or security "
            "code. Please do that yourself."
        )


async def _settle() -> None:
    """Give the page a moment to start navigating so the next read is not mid-load."""
    await asyncio.sleep(0.4)


def _parse_plan(raw: str) -> dict[str, str]:
    """Pull the JSON object out of a planner reply, tolerating stray prose."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"action": "none"}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"action": "none"}
    if not isinstance(parsed, dict):
        return {"action": "none"}
    return {str(key): str(value) for key, value in parsed.items()}


def _describe_plan(plan: dict[str, str]) -> str:
    kind = plan.get("action", "none")
    if kind == "click":
        return f"click on {plan.get('target', 'the control')}"
    if kind == "type":
        return f"type {plan.get('text', '')!r} into {plan.get('target', 'the field')}"
    if kind == "press_key":
        return f"press {plan.get('key', 'a key')}"
    if kind == "scroll":
        return f"scroll {plan.get('direction', 'down')}"
    return "do nothing, because no control on the page matches that"


browser_manager = BrowserManager()
