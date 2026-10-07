"""``web_read`` backend: Keenable single-URL page fetch.

Fetches one URL as clean markdown via Keenable's page reader
(``GET /v1/fetch``). Keenable returns the page as LLM-ready markdown with its
title, and it works **keyless**: with no ``api_key`` in the spec the public
endpoint (``/v1/fetch/public``) is used, so it runs out of the box. Supplying
an ``api_key`` switches to the authenticated endpoint (``/v1/fetch``) and lifts
the per-IP rate limit. Same vendor as the ``keenable`` ``web_search`` backend,
so one keyless provider can cover both finding and reading a page.

Configured in the agent spec::

    tools:
      builtins:
        - name: web_read
          read_provider: keenable
          # api_key: ${KEENABLE_API_KEY}   # optional; keyless works, keyed lifts rate limits

See https://docs.keenable.ai
"""

from __future__ import annotations

from typing import Any

import httpx

# Shared with the ``keenable`` ``web_search`` backend: the same base URL (and
# its ``OMNIGENT_KEENABLE_BASE_URL`` test override) and the same
# ``X-Keenable-Title`` value, which Keenable requires on keyless calls.
from omnigent.tools.builtins.web_search_keenable import _CLIENT_TITLE, _keenable_base_url

_DEFAULT_TIMEOUT_S = 90.0

# Ask Keenable for no more than the dispatcher keeps; the value is a hint, so
# the central cap in ``web_read`` still applies to whatever comes back.
_MAX_CHARS = 50_000


def _read_keenable(url: str, config: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Fetch one URL as markdown via Keenable.

    Keyless by default: with no ``api_key`` the public endpoint is used and the
    request carries the ``X-Keenable-Title`` header Keenable requires on
    token-less calls. With an ``api_key`` the authenticated endpoint is used and
    the key is sent in the ``X-API-Key`` header.

    :param url: The page URL to read (validated by the caller).
    :param config: Spec-level config; ``api_key`` optional.
    :returns: ``(content, diagnostic)``: the markdown on success, or a
        diagnostic message on failure / empty page. Exactly one is non-None.
        Never raises.
    """
    api_key = (config.get("api_key") or "").strip()
    # Keyed endpoint with a key; keyless public endpoint without one.
    path = "/v1/fetch" if api_key else "/v1/fetch/public"
    headers = {"X-Keenable-Title": _CLIENT_TITLE}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        resp = httpx.get(
            f"{_keenable_base_url()}{path}",
            params={"url": url, "max_chars": _MAX_CHARS},
            headers=headers,
            timeout=_DEFAULT_TIMEOUT_S,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 429:
            return None, (
                f"Keenable read error: HTTP {code} (rate limit). Set an api_key in "
                "the web_read config to raise the limit."
            )
        if code in (401, 403):
            return None, (
                f"Keenable read error: HTTP {code} (check the api_key in the web_read config)."
            )
        if code == 400:
            return None, f"Keenable read error: HTTP {code} (the URL was rejected)."
        if code == 404:
            return None, f"Keenable read error: HTTP {code} (no page found at this URL)."
        return None, f"Keenable read error: HTTP {code}"
    except httpx.RequestError as exc:
        # Covers connect/timeout/redirect/protocol/decoding errors uniformly.
        return None, f"Keenable read error: {exc}"

    try:
        payload = resp.json()
    except (ValueError, TypeError):
        return None, "Keenable read error: non-JSON response."
    return _format_read(payload, url)


def _format_read(payload: dict[str, Any], url: str) -> tuple[str | None, str | None]:
    """
    Pull the markdown out of Keenable's ``/v1/fetch`` response.

    Keenable returns ``{"url", "title", "content", "description", ...}`` where
    ``content`` is the page as markdown. The title is prepended as a heading
    when the content does not already start with one, so the model sees what
    page it is reading.

    :param payload: The parsed JSON response.
    :param url: The requested URL (for the empty-result message).
    :returns: ``(content, diagnostic)``; exactly one is non-None. Central
        truncation is applied by the dispatcher, not here.
    """
    if not isinstance(payload, dict):
        return None, "Keenable read error: malformed response (not an object)."
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, f"web_read: no content extracted from {url} (page may be empty or blocked)."
    content = content.strip()
    title = payload.get("title")
    if isinstance(title, str) and title.strip() and not content.startswith("#"):
        content = f"# {title.strip()}\n\n{content}"
    return content, None
