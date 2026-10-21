"""Tests for the browser URL policy.

These are the security boundary for the agent's browser, so they are plain unit tests
with no network access and no browser: the point is to pin down which addresses are
refused, independent of anything Chromium or the deployment environment does.
"""

import pytest

from browser import policy
from browser.policy import PolicyError, check_url


@pytest.fixture(autouse=True)
def _no_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most cases exercise SSRF rules, so the allowlist is left unset."""
    monkeypatch.delenv("NOVA_BROWSER_ALLOWED_DOMAINS", raising=False)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/admin",
        "https://127.0.0.1/",
        "https://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://[::ffff:127.0.0.1]/",
        "http://100.64.0.1/",
    ],
)
def test_blocks_non_public_addresses(url: str) -> None:
    """Anything resolving inside the network must be refused, including IPv6 forms."""
    with pytest.raises(PolicyError):
        check_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "vbscript:msgbox(1)",
        "about:blank",
        "chrome://settings",
        "view-source:http://example.com",
    ],
)
def test_blocks_non_http_schemes(url: str) -> None:
    """Only http and https are navigable, and no scheme may smuggle past the check."""
    with pytest.raises(PolicyError):
        check_url(url)


def test_javascript_url_is_not_rewritten_into_https() -> None:
    """A scheme without slashes must still be recognised as a scheme.

    Prepending https to 'javascript:alert(1)' would produce a real URL to a host
    called 'javascript', which is both wrong and a validation bypass.
    """
    with pytest.raises(PolicyError, match="javascript"):
        check_url("javascript:alert(1)")


def test_allows_public_address() -> None:
    assert check_url("https://example.com") == "https://example.com"


def test_adds_scheme_to_bare_host() -> None:
    """Speech rarely produces a full URL, so a bare host should still work."""
    assert check_url("example.com") == "https://example.com"


def test_rejects_empty_url() -> None:
    with pytest.raises(PolicyError):
        check_url("   ")


def test_allowlist_permits_listed_domain_and_subdomains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subdomains of an allowed domain are allowed too."""
    monkeypatch.setenv("NOVA_BROWSER_ALLOWED_DOMAINS", "bbc.co.uk, example.org")
    assert check_url("https://bbc.co.uk/news") == "https://bbc.co.uk/news"
    check_url("https://www.bbc.co.uk/news")


@pytest.mark.parametrize(
    "host",
    ["www.bbc.co.uk", "sports.example.org", "a.b.c.example.org"],
)
def test_subdomains_are_allowed(host: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching is on dot boundaries, so depth of subdomain does not matter.

    DNS is stubbed because this is a test of name matching, not of resolution, and
    these hosts do not all exist.
    """
    monkeypatch.setenv("NOVA_BROWSER_ALLOWED_DOMAINS", "bbc.co.uk, example.org")
    monkeypatch.setattr(policy, "_assert_public_host", lambda host: None)
    check_url(f"https://{host}/")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.com/",
        "https://notbbc.co.uk/",
        "https://bbc.co.uk.evil.com/",
        "https://xexample.org/",
    ],
)
def test_allowlist_blocks_lookalike_domains(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suffix matching must not be fooled by a prefix or a substring."""
    monkeypatch.setenv("NOVA_BROWSER_ALLOWED_DOMAINS", "bbc.co.uk, example.org")
    with pytest.raises(PolicyError, match="permitted sites"):
        check_url(url)


def test_empty_allowlist_blocks_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty allowlist is an operator mistake and must fail closed, not open."""
    monkeypatch.setenv("NOVA_BROWSER_ALLOWED_DOMAINS", "")
    with pytest.raises(PolicyError, match="set but empty"):
        check_url("https://example.com")


def test_unset_allowlist_permits_public_web(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local development should not require configuring a list first."""
    monkeypatch.delenv("NOVA_BROWSER_ALLOWED_DOMAINS", raising=False)
    check_url("https://example.com")
