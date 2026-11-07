"""Fetch URLs the Studio was given without reaching into the local network.

Links come from people (Your texts, news sources, style imports) and are fetched by the server.
When the Studio runs on a PC that is reachable from the internet, a crafted link could otherwise
point the server at the router's admin page, a cloud metadata address or another PC on the LAN.
So every hostname is resolved first and private, loopback, link-local and other special
addresses are refused: on the link, on every redirect hop, and on the address the connection
really went to (DNS can answer differently the second time). TEXTLLM_ALLOW_PRIVATE_URLS=1 (the
Studio setting "allow links to the local network" sets it) lets you read from a NAS or a local
web server on purpose.

    core.net.check_url(url)              -> the url, or raises ValueError
    core.net.get(url, max_bytes=...)     -> requests.Response, redirects checked hop by hop

Fixed project-owned addresses (the Gutenberg catalogue, Hugging Face) are fetched directly by
data/build_corpus.py and data/build_instructions.py; everything a person types goes through here.
"""
import ipaddress
import os
import socket
import time
from urllib.parse import urljoin, urlsplit

MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
UA = {"User-Agent": "text-llm/1.0 (+https://github.com/hovchik/textllm)"}
DNS_TTL = 60  # seconds a resolved host is reused (check_peer still verifies the real connection)

_dns = {}  # host -> (addresses, expiry)


def allow_private():
    """Whether local-network addresses may be fetched (TEXTLLM_ALLOW_PRIVATE_URLS)."""
    return os.environ.get("TEXTLLM_ALLOW_PRIVATE_URLS", "") in ("1", "true", "yes")


def is_public_ip(ip):
    a = ipaddress.ip_address(ip)
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped:
        a = a.ipv4_mapped
    # is_global is False for private, loopback, link-local, multicast, reserved, unspecified and
    # carrier-grade NAT (100.64.0.0/10, also used by Tailscale) addresses
    return a.is_global and not a.is_multicast


def resolve(host):
    """All addresses a hostname points to (a literal address is returned as itself)."""
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    ips, expiry = _dns.get(host, (None, 0))
    if ips is not None and time.time() < expiry:
        return ips
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"{host}: can't be resolved ({e.strerror or e})") from None
    ips = sorted({i[4][0] for i in infos})
    _dns[host] = (ips, time.time() + DNS_TTL)
    if len(_dns) > 5000:
        _dns.clear()
    return ips


def _refused(what, ip):
    return ValueError(f"{what} ({ip}): local-network addresses are not read (see Settings → Links)")


def check_url(url, dns=True):
    """Return url if it is an http(s) link to a public address; raise ValueError otherwise.
    dns=False only checks the link itself (scheme, local names, literal addresses) without a
    lookup: for validating a saved setting offline; the fetch checks the resolved address."""
    p = urlsplit(url.strip())
    if p.scheme not in ("http", "https"):
        raise ValueError(f"{url[:80]}: only http and https links can be read")
    if not p.hostname:
        raise ValueError(f"{url[:80]}: no host name")
    if p.username or p.password:
        raise ValueError(f"{url[:80]}: links with a user name or password are not read")
    if allow_private():
        return url
    host = p.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".localhost", ".internal")):
        raise _refused(host, "local name")
    if not dns:
        try:
            ips = [str(ipaddress.ip_address(host))]
        except ValueError:
            return url  # a name: checked when it is fetched
    else:
        ips = resolve(host)
    for ip in ips:
        if not is_public_ip(ip):
            raise _refused(host, ip)
    return url


def peer_ip(response):
    """The address a streamed requests.Response is actually connected to (None if unknown)."""
    try:
        return response.raw._connection.sock.getpeername()[0]  # noqa: SLF001
    except Exception:  # noqa: BLE001 - no live socket (mocked, or already closed)
        return None


def check_peer(response):
    """The hostname was checked before connecting, but DNS can answer differently the second time
    (rebinding). So the address the connection really went to is checked too."""
    ip = None if allow_private() else peer_ip(response)
    if ip and not is_public_ip(ip):
        response.close()
        raise _refused(response.url[:80], ip)
    return response


def get(url, headers=None, timeout=60, max_bytes=DEFAULT_MAX_BYTES, stream=False, session=None):
    """requests.get with the address check applied to the link, to every redirect and to the
    connection itself, and a size cap (Content-Length, and the body when not streaming).
    Returns the final Response."""
    import requests

    s = session or requests
    h = {**UA, **(headers or {})}
    too_big = ValueError(f"{url[:80]}: larger than {max_bytes // 1_000_000} MB, skipped")
    for _ in range(MAX_REDIRECTS + 1):
        check_url(url)
        r = check_peer(s.get(url, headers=h, timeout=timeout, stream=True, allow_redirects=False))
        if r.is_redirect or r.is_permanent_redirect:
            nxt = r.headers.get("Location")
            r.close()
            if not nxt:
                raise ValueError(f"{url[:80]}: redirect without a target")
            url = urljoin(url, nxt)
            continue
        n = r.headers.get("Content-Length")
        if n and n.isdigit() and int(n) > max_bytes:
            r.close()
            raise too_big
        if not stream:
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    r.close()
                    raise too_big
                chunks.append(chunk)
            r._content = b"".join(chunks)  # noqa: SLF001 - make .content/.text work as after a plain get()
        return r
    raise ValueError(f"{url[:80]}: too many redirects")
