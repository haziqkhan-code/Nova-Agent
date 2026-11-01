"""Nova's standalone tools: web search, weather, and opening websites."""

import asyncio

import aiohttp
from ddgs import DDGS
from livekit import rtc
from livekit.agents import function_tool, get_job_context
from livekit.agents.llm import ToolError

# Open-Meteo needs no API key: geocoding resolves a place name, the forecast
# endpoint answers for its coordinates.
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_TIMEOUT = 15.0

# WMO weather interpretation codes, as listed at open-meteo.com.
_WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

# What the user means when they say a site name out loud, including how speech
# recognition commonly renders it. Only these sites may be opened.
_KNOWN_SITES = {
    "youtube": "https://www.youtube.com",
    "you tube": "https://www.youtube.com",
    "youtu be": "https://www.youtube.com",
    "yt": "https://www.youtube.com",
    "google": "https://www.google.com",
    "google search": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "google maps": "https://maps.google.com",
    "wikipedia": "https://www.wikipedia.org",
    "github": "https://github.com",
    "bbc": "https://www.bbc.co.uk",
    "reddit": "https://www.reddit.com",
    "facebook": "https://www.facebook.com",
    "twitter": "https://twitter.com",
    "instagram": "https://www.instagram.com",
}


@function_tool
async def search_web(query: str) -> str:
    """Search the web for current information."""

    def run() -> str:
        results = DDGS().text(query, max_results=3)
        return "\n".join(f"{r['title']}: {r['body']}" for r in results)

    return await asyncio.to_thread(run)


@function_tool
async def get_weather(city: str) -> str:
    """Look up the current weather for a city or country.

    Use this for any question about current weather, temperature, or conditions
    in a place.

    Args:
        city: The city or country to look up, such as "London" or "Japan".
    """
    if not city.strip():
        raise ToolError("I need a city or country to check the weather for.")

    try:
        async with asyncio.timeout(_WEATHER_TIMEOUT):
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    _GEOCODE_URL,
                    params={
                        "name": city,
                        "count": 1,
                        "language": "en",
                        "format": "json",
                    },
                ) as response:
                    response.raise_for_status()
                    geo = await response.json()

                matches = geo.get("results") or []
                if not matches:
                    raise ToolError(f"I could not find a place called {city}.")

                place = matches[0]
                async with session.get(
                    _FORECAST_URL,
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                    },
                ) as response:
                    response.raise_for_status()
                    forecast = await response.json()
    except TimeoutError:
        raise ToolError("The weather service took too long to answer.") from None
    except aiohttp.ClientError as exc:
        raise ToolError(f"I could not reach the weather service: {exc}") from exc

    current = forecast.get("current") or {}
    description = _WMO_CODES.get(current.get("weather_code"), "unknown conditions")

    location = place.get("name") or city
    country = place.get("country")
    if country and country != location:
        location = f"{location}, {country}"

    parts = [f"{location}: {description}"]
    temperature = current.get("temperature_2m")
    if temperature is not None:
        parts.append(f"{temperature} degrees Celsius")
    wind = current.get("wind_speed_10m")
    if wind is not None:
        parts.append(f"wind {wind} kilometers per hour")

    return ", ".join(parts)


@function_tool
async def open_website(site: str) -> str:
    """Open a website in the user's browser.

    Use this when the user asks you to open a site, such as "Open YouTube" or
    "Open Google". Only a small allowlist of well-known sites can be opened.

    Args:
        site: The name of the website to open, such as "YouTube" or "Google".
    """
    url = _KNOWN_SITES.get(site.strip().casefold())
    if url is None:
        allowed = ", ".join(
            sorted({name.title() for name in _KNOWN_SITES if " " not in name})
        )
        raise ToolError(f"I can only open sites like {allowed}. I cannot open {site}.")

    # The agent runs in the cloud, so it asks the user's own browser to open the page.
    room = get_job_context().room
    viewers = [
        p
        for p in room.remote_participants.values()
        if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    ]
    if not viewers:
        raise ToolError("I cannot find your browser right now.")

    try:
        await room.local_participant.perform_rpc(
            destination_identity=viewers[0].identity,
            method="open_url",
            payload=url,
            response_timeout=5.0,
        )
    except rtc.RpcError:
        raise ToolError("Your browser did not accept the request to open it.") from None

    return f"Opened {site} in your browser."