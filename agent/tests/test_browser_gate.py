"""The confirmation gate for the act tool.

The guarantee under test: act never performs an action on its first call for a given
instruction. It returns a description and only acts once the identical instruction has
been asked for a second time. That second call is the model reporting the user's
agreement, so it is the narrowest confirmation that still works over voice.

These tests use a fake browser, so the gate is verified independently of Playwright,
of the planner model, and of the summarizer.
"""

import contextlib
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from dotenv import load_dotenv

from browser import session as session_module
from browser.session import BrowserManager
from browser.tools import BrowserTools, _looks_like_address

load_dotenv(".env.local")

PAGE = """<!doctype html>
<html><head><title>Shop</title></head>
<body>
  <h1>Order summary</h1>
  <p>One teapot, twelve pounds.</p>
  <a href="/checkout.html">Checkout</a>
</body></html>
"""


class _Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def shop() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/"
    server.shutdown()
    server.server_close()


@pytest.fixture(autouse=True)
def _allow_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_module, "check_url", lambda url: url)


@pytest.fixture
async def manager(shop: str) -> Iterator[BrowserManager]:
    m = BrowserManager()
    await m.open(shop)
    try:
        yield m
    finally:
        await m.aclose()


def _has_gateway_credentials() -> bool:
    import os

    return bool(os.getenv("LIVEKIT_API_KEY"))


requires_model = pytest.mark.skipif(
    not _has_gateway_credentials(),
    reason="LIVEKIT_API_KEY is not set; helper models unavailable",
)


class _FakeAction:
    def __init__(self) -> None:
        self.performed: list[str] = []

    async def __call__(self, instruction: str) -> str:
        self.performed.append(instruction)
        return "done"


def _returning(value: str):
    async def _inner(*args: object, **kwargs: object) -> str:
        return value

    return _inner


@pytest.fixture
def fake_action() -> _FakeAction:
    return _FakeAction()


class _FakeContext:
    """Stands in for RunContext.

    The gate logic never uses anything from the real context, but it does enter the
    filler block, so the fake has to provide that as a context manager.
    """

    @contextlib.contextmanager
    def with_filler(self, *args: object, **kwargs: object) -> Iterator[None]:
        yield


@pytest.fixture
def ctx() -> _FakeContext:
    return _FakeContext()


def test_pending_starts_empty() -> None:
    """A fresh tool set has no pre-approved action."""
    assert BrowserTools()._pending is None


async def test_first_call_only_previews(
    manager: BrowserManager,
    monkeypatch: pytest.MonkeyPatch,
    fake_action: _FakeAction,
    ctx: _FakeContext,
) -> None:
    """The first call must describe the action and not perform it."""
    monkeypatch.setattr(manager, "preview_action", _returning("click on Checkout"))
    monkeypatch.setattr(manager, "run_action", fake_action)
    monkeypatch.setattr("browser.tools.summarize", _returning("done"))

    tools = BrowserTools(manager)
    result = await tools.act(ctx, "click the checkout link")

    assert fake_action.performed == []
    assert "not done it yet" in result
    assert tools._pending == "click the checkout link"


async def test_second_identical_call_acts(
    manager: BrowserManager,
    monkeypatch: pytest.MonkeyPatch,
    fake_action: _FakeAction,
    ctx: _FakeContext,
) -> None:
    """Asking twice with the same instruction is the confirmation, so it acts."""
    monkeypatch.setattr(manager, "preview_action", _returning("click on Checkout"))
    monkeypatch.setattr(manager, "run_action", fake_action)
    monkeypatch.setattr("browser.tools.summarize", _returning("done"))

    tools = BrowserTools(manager)
    await tools.act(ctx, "click the checkout link")
    await tools.act(ctx, "click the checkout link")

    assert fake_action.performed == ["click the checkout link"]
    assert tools._pending is None


async def test_a_different_instruction_does_not_reuse_the_confirmation(
    manager: BrowserManager,
    monkeypatch: pytest.MonkeyPatch,
    fake_action: _FakeAction,
    ctx: _FakeContext,
) -> None:
    """Approval for one action must never carry over to another.

    This is the loophole that would matter: if the user agrees to clicking Checkout,
    a later instruction to click something else must start from preview again.
    """
    monkeypatch.setattr(manager, "preview_action", _returning("click on Checkout"))
    monkeypatch.setattr(manager, "run_action", fake_action)
    monkeypatch.setattr("browser.tools.summarize", _returning("done"))

    tools = BrowserTools(manager)
    await tools.act(ctx, "click the checkout link")
    await tools.act(ctx, "click the refund button")

    assert fake_action.performed == []


async def test_confirmation_is_case_insensitive(
    manager: BrowserManager,
    monkeypatch: pytest.MonkeyPatch,
    fake_action: _FakeAction,
    ctx: _FakeContext,
) -> None:
    """Speech recognition changes case freely, so casing must not reset the gate."""
    monkeypatch.setattr(manager, "preview_action", _returning("click on Checkout"))
    monkeypatch.setattr(manager, "run_action", fake_action)
    monkeypatch.setattr("browser.tools.summarize", _returning("done"))

    tools = BrowserTools(manager)
    await tools.act(ctx, "click the checkout link")
    await tools.act(ctx, "CLICK THE CHECKOUT LINK")

    assert fake_action.performed == ["CLICK THE CHECKOUT LINK"]


async def test_unmatched_target_does_not_arm_the_gate(
    manager: BrowserManager,
    monkeypatch: pytest.MonkeyPatch,
    fake_action: _FakeAction,
    ctx: _FakeContext,
) -> None:
    """If nothing matches, nothing is armed, so a later real action still previews."""
    monkeypatch.setattr(manager, "preview_action", _returning("do nothing, because"))
    monkeypatch.setattr(manager, "run_action", fake_action)
    monkeypatch.setattr("browser.tools.summarize", _returning("done"))

    tools = BrowserTools(manager)
    first = await tools.act(ctx, "click the frobnicate valve")
    assert "could not find" in first
    assert tools._pending is None

    monkeypatch.setattr(manager, "preview_action", _returning("click on Checkout"))
    await tools.act(ctx, "click the checkout link")
    assert fake_action.performed == []


async def test_act_before_browse_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    ctx: _FakeContext,
) -> None:
    """Acting with no page open is a clear usage error."""
    from livekit.agents.llm import ToolError

    manager = BrowserManager()
    tools = BrowserTools(manager)
    with pytest.raises(ToolError, match="not opened a page"):
        await tools.act(ctx, "click something")
    await manager.aclose()


async def test_empty_instruction_is_refused(
    manager: BrowserManager, ctx: _FakeContext
) -> None:
    from livekit.agents.llm import ToolError

    tools = BrowserTools(manager)
    with pytest.raises(ToolError):
        await tools.act(ctx, "   ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("example.com", "example.com"),
        ("https://example.com/x", "https://example.com/x"),
        ("www.example.com", "www.example.com"),
        ("example.com/some/path", "example.com/some/path"),
    ],
)
def test_recognises_addresses(text: str, expected: str) -> None:
    assert _looks_like_address(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "what is the capital of Australia",
        "how tall is the Shard in London",
        "example",
        "look this up for me please",
    ],
)
def test_treats_questions_as_questions(text: str) -> None:
    """A question mentioning a site must be searched, not treated as an address."""
    assert _looks_like_address(text) is None


@requires_model
async def test_gate_survives_a_real_planner(
    manager: BrowserManager, fake_action: _FakeAction, ctx: _FakeContext
) -> None:
    """End to end: a real planner still cannot act without the second call."""
    manager.run_action = fake_action  # type: ignore[method-assign]
    tools = BrowserTools(manager)

    first = await tools.act(ctx, "click the checkout link")
    assert fake_action.performed == []
    assert "not done it yet" in first

    await tools.act(ctx, "click the checkout link")
    assert fake_action.performed == ["click the checkout link"]
