"""Choice of the small helper models.

Nova's voice model is realtime and tuned for conversation, which makes it the wrong
tool for reducing a web page to two sentences or for mapping a spoken instruction onto
a page control. Both of those are one-shot text jobs, so they go to cheap text models
instead, and both are configured here for one reason: this project authenticates with
GOOGLE_API_KEY, so a default pointing at another provider fails on first use.

Both names can be overridden per deployment, which is how you would swap in a
provider this deployment already has credentials for.
"""

import os

# Writing spoken prose from a long page. Quality matters more than price here, because
# this text is what the user actually hears.
DEFAULT_SUMMARIZER_MODEL = "google/gemini-2.5-flash"

# Mapping an instruction to a control on a known page. A single JSON object comes back,
# so the smallest cheap model that is reliable at that is enough.
DEFAULT_PLANNER_MODEL = "google/gemini-2.5-flash-lite"


def summarizer_model() -> str:
    return os.getenv("NOVA_BROWSER_SUMMARIZER_MODEL") or DEFAULT_SUMMARIZER_MODEL


def planner_model() -> str:
    return os.getenv("NOVA_BROWSER_PLANNER_MODEL") or DEFAULT_PLANNER_MODEL
