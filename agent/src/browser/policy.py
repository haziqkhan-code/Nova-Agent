"""URL policy for the agent-controlled browser.

An agent that can navigate anywhere is an SSRF and prompt-injection target, so every
navigation is checked here before Playwright sees it. Two independent gates:

1. A scheme and IP-range gate, always on. Rejects loopback, link-local, private, and
   unique-local addresses so the browser cannot be pointed at cloud instance metadata
   or an internal service on the deployment network.
2. A domain allowlist, on by default outside local development. Restricts Nova to
   sites the operator has explicitly approved.

Both gates are deliberately structural (addresses, schemes, suffixes) rather than
semantic. Deciding whether a *request* is appropriate is the model's job; deciding
whether a *host* is reachable is this module's job.
"""

import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit

_UNSAFE_SCHEMES = frozenset(
    {"file", "data", "javascript", "vbscript", "about", "blob", "chrome", "view-source"}
)
_SAFE_SCHEMES = frozenset({"http", "https"})

# A leading RFC 3986 scheme, e.g. the "https:" in "https://x" or the "javascript:" in
# "javascript:alert(1)". Matched before deciding whether to prepend a scheme.
_SCHEME_PREFIX = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


class PolicyError(Exception):
    """A navigation was refused by policy. Safe to show to the model verbatim."""


def _parse_allowlist(raw: str | None) -> tuple[frozenset[str], bool]:
    """Return the allowed domain suffixes and whether the allowlist is enforcing.

    Three distinct states, because treating them alike is a security bug:

    * unset      local development, so the whole public web is allowed.
    * set        a real allowlist, enforced.
    * set empty  an operator mistake. Enforcing an empty list blocks everything, which
                 fails loudly on the first call instead of silently removing the
                 restriction the variable exists to apply.
    """
    if raw is None:
        return frozenset(), False

    entries = [item.strip().lower().lstrip(".") for item in raw.split(",")]
    return frozenset(entry for entry in entries if entry), True


def _allowlist() -> tuple[frozenset[str], bool]:
    return _parse_allowlist(os.getenv("NOVA_BROWSER_ALLOWED_DOMAINS"))


def _domain_matches(host: str, suffixes: frozenset[str]) -> bool:
    """True when host is one of the suffixes or a subdomain of one."""
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)


def _assert_public_host(host: str) -> None:
    """Reject hosts that resolve to a non-public address.

    Checks the literal IP first so a numeric host never reaches DNS, then resolves
    and inspects every returned address so a hostname cannot smuggle a private target
    through DNS.
    """
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None

    if literal is not None:
        addresses = [literal]
    else:
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as err:
            raise PolicyError(
                f"I could not resolve the address {host!r}. It may not exist."
            ) from err
        addresses = []
        for info in infos:
            try:
                addresses.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue

    if not addresses:
        raise PolicyError(f"The address {host!r} did not resolve to any IP address.")

    for address in addresses:
        if not _is_public(address):
            raise PolicyError(
                f"{host!r} points to the non-public address {address}, which I am not "
                "allowed to reach."
            )


def _is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        return False

    # Carrier-grade NAT (100.64.0.0/10) is flagged neither private nor reserved but is
    # not reachable from the public internet either.
    if isinstance(address, ipaddress.IPv4Address) and address in ipaddress.ip_network(
        "100.64.0.0/10"
    ):
        return False

    # IPv6 unique-local (fc00::/7) and IPv4-mapped IPv6 both need an explicit check.
    return not (
        isinstance(address, ipaddress.IPv6Address)
        and address.ipv4_mapped is not None
        and not _is_public(address.ipv4_mapped)
    )


def check_url(url: str) -> str:
    """Validate a URL and return it normalized. Raises PolicyError when refused."""
    candidate = url.strip()
    if not candidate:
        raise PolicyError("No address was given to open.")

    # A bare host such as "example.com" is a common thing to say out loud, so add a
    # scheme. The test is for any scheme prefix, not "://", because "javascript:alert(1)"
    # has no slashes and must not be silently rewritten into an https address.
    if not _SCHEME_PREFIX.match(candidate):
        candidate = f"https://{candidate}"

    try:
        parts = urlsplit(candidate)
    except ValueError as err:
        raise PolicyError(f"{url!r} is not a valid web address.") from err

    scheme = parts.scheme.lower()
    if scheme in _UNSAFE_SCHEMES:
        raise PolicyError(f"I cannot open {scheme}: addresses.")
    if scheme not in _SAFE_SCHEMES:
        raise PolicyError(f"I can only open http and https addresses, not {scheme}.")

    host = (parts.hostname or "").lower()
    if not host:
        raise PolicyError(f"{url!r} does not contain a web address.")

    suffixes, enforcing = _allowlist()
    if enforcing and not _domain_matches(host, suffixes):
        if not suffixes:
            raise PolicyError(
                "NOVA_BROWSER_ALLOWED_DOMAINS is set but empty, so no site is "
                "permitted. Give it a comma-separated list of domains, or unset it to "
                "allow the public web."
            )
        allowed = ", ".join(sorted(suffixes))
        raise PolicyError(
            f"{host} is not on my list of permitted sites. I can only open: {allowed}."
        )

    _assert_public_host(host)

    return candidate
