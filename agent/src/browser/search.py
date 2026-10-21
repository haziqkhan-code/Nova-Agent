"""Pick a page to open for a question.

Kept separate from :mod:`browser.session` so the search dependency is imported only
when a question actually needs searching, not on every job start.
"""

import asyncio
from urllib.parse import urlencode

from ddgs import DDGS

from .policy import PolicyError, check_url


def _search_sync(query: str) -> list[dict[str, str]]:
    return list(DDGS().text(query, max_results=5))


async def best_result_url(query: str) -> str:
    """Search for a query and return the address of the result most worth reading.

    Results are filtered through the same policy as any other navigation, so a search
    engine cannot route Nova onto a site the operator blocked.
    """
    results = await asyncio.to_thread(_search_sync, query)
    if not results:
        raise PolicyError(f"I could not find anything about {query!r}.")

    first_failure: PolicyError | None = None
    for result in results:
        href = result.get("href") or result.get("url") or ""
        if not href:
            continue
        try:
            return check_url(href)
        except PolicyError as err:
            first_failure = first_failure or err
            continue

    raise first_failure or PolicyError(
        f"I could not find a usable result for {query!r}."
    )


def search_url(query: str) -> str:
    """Build a DuckDuckGo results URL, for when direct search is preferred."""
    return f"https://duckduckgo.com/?{urlencode({'q': query})}"
