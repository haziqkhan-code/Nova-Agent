"""Live browser smoke test for BrowserManager.

Runs against a local HTTP server so it needs no internet access. This is the check
that the Playwright plumbing actually works; the policy tests cover the security
boundary without a browser.
"""

import asyncio
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from browser import policy
from browser import session as session_module
from browser.policy import PolicyError
from browser.session import BrowserError, BrowserManager, _assert_not_sensitive

PAGE = """<!doctype html>
<html><head><title>Nova test page</title></head>
<body>
  <nav>Skip to content | Menu | Privacy policy | Accept all</nav>
  <h1>Opening hours</h1>
  <p>The library is open from nine until six on weekdays.</p>
  <a href="/second.html">More information</a>
  <form action="/submit">
    <label for="email">Email address</label>
    <input id="email" name="email" type="email">
    <label for="pw">Password</label>
    <input id="pw" name="pw" type="password">
    <button type="submit">Send</button>
  </form>
</body></html>
"""

SECOND = """<!doctype html>
<html><head><title>Second page</title></head>
<body><h1>Second page heading</h1><p>Nothing to see here.</p></body></html>
"""


class _Handler(SimpleHTTPRequestHandler):
    """Serves the two fixture pages in memory."""

    def do_GET(self) -> None:
        body = SECOND if "second" in self.path else PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args: object) -> None:
        """Silence the default stderr request logging."""


@pytest.fixture(scope="module")
def local_site() -> Iterator[str]:
    """A loopback HTTP server holding the fixture pages."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _allow_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the policy gate so the fixture server is reachable.

    Policy blocks loopback, correctly. Testing the Playwright side against a known
    local page needs that one check lifted, and only that one check; every other
    behaviour of the policy stays in force and is covered in test_browser_policy.py.
    """
    monkeypatch.setattr(session_module, "check_url", lambda url: url)


@pytest.fixture
async def browser() -> Iterator[BrowserManager]:
    manager = BrowserManager()
    try:
        yield manager
    finally:
        await manager.aclose()


async def test_open_reads_visible_text(
    browser: BrowserManager, local_site: str
) -> None:
    """Reading a page returns its prose and drops navigation chrome."""
    text = await browser.open(local_site)
    assert "library is open from nine until six" in text
    assert "Privacy policy" not in text
    assert "Accept all" not in text
    assert "Skip to content" not in text


async def test_open_follows_a_link(browser: BrowserManager, local_site: str) -> None:
    """Clicking a link by its visible name navigates and the new page is readable."""
    await browser.open(local_site)
    await browser._click_target("More information")
    assert "Second page heading" in await browser.current_page_text()


async def test_click_fills_and_refuses_unknown_targets(
    browser: BrowserManager, local_site: str
) -> None:
    """A missing control is a clear error rather than a timeout or a wrong click."""
    await browser.open(local_site)
    with pytest.raises(BrowserError, match="could not find"):
        await browser._click_target("No Such Button Anywhere")


async def test_type_target_is_not_sensitive(
    browser: BrowserManager, local_site: str
) -> None:
    """Filling a password field is refused at the manager, not only in the planner."""
    await browser.open(local_site)
    with pytest.raises(BrowserError, match="will not"):
        await browser._type_target("Password", "hunter2")


async def test_press_key_rejects_unsafe_key(
    browser: BrowserManager, local_site: str
) -> None:
    """Only a fixed set of navigation keys may be sent to a page."""
    await browser.open(local_site)
    with pytest.raises(BrowserError, match="Enter, Escape, Tab"):
        await browser._press_key("Control+L")


async def test_press_key_accepts_allowed_key(
    browser: BrowserManager, local_site: str
) -> None:
    """Enter is permitted, so the allowlist is not simply broken."""
    await browser.open(local_site)
    await browser._press_key("Tab")


@pytest.mark.parametrize(
    "target", ["Password", "Card number", "CVV", "Security code", "One-time code"]
)
def test_sensitive_fields_are_refused(target: str) -> None:
    """Password and payment fields must never be filled, whatever the instruction."""
    with pytest.raises(BrowserError, match="will not"):
        _assert_not_sensitive(target, action="fill in")


def test_ordinary_field_is_allowed() -> None:
    """The sensitive-field guard must not block legitimate fields."""
    _assert_not_sensitive("Email address", action="fill in")
    _assert_not_sensitive("Search", action="click")


async def test_closed_manager_refuses_work(
    browser: BrowserManager, local_site: str
) -> None:
    """After shutdown the manager stays closed rather than silently restarting."""
    await browser.open(local_site)
    await browser.aclose()
    assert not browser.is_running
    with pytest.raises(BrowserError, match="already been closed"):
        await browser.open(local_site)


async def test_concurrent_opens_are_serialized(
    browser: BrowserManager, local_site: str
) -> None:
    """Parallel tool calls must not drive the page at the same time."""
    results = await asyncio.gather(
        *[browser.open(local_site) for _ in range(3)],
        return_exceptions=True,
    )
    assert all(isinstance(result, str) for result in results), results


async def test_policy_blocks_are_untouched_by_the_test_patch() -> None:
    """Guard against the autouse patch making the policy a no-op.

    If this ever fails, the other tests in this file are not proving what they claim.
    The real function is reached directly, bypassing the patched session reference.
    """
    with pytest.raises(PolicyError):
        policy.check_url("http://169.254.169.254/latest/meta-data/")
