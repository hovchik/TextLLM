"""Make a cluster coordinator reachable from other networks without touching the router by hand.

* upnp_open(): asks the home router (UPnP IGD, which most home routers support and have on) to
  forward a TCP port to this PC, and reads the router's public IP address. Standard library only.
* addresses(): every address other PCs might use, labelled: "internet" (public IP with the port
  opened), "tailscale" (100.x.y.z, works through any router), "local" (same network only).

Nothing here sends anything outside the local network: UPnP talks to the router only, and the
public address comes from the router, not from a web service.
"""
import ipaddress, os, re, shutil, socket, subprocess, threading, time
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET

SSDP = ("239.255.255.250", 1900)
SERVICES = ("urn:schemas-upnp-org:service:WANIPConnection:2", "urn:schemas-upnp-org:service:WANIPConnection:1",
            "urn:schemas-upnp-org:service:WANPPPConnection:1")
SEARCH = ("urn:schemas-upnp-org:device:InternetGatewayDevice:2", "urn:schemas-upnp-org:device:InternetGatewayDevice:1",
          *SERVICES)
CGNAT = ipaddress.ip_network("100.64.0.0/10")  # carrier NAT, and also Tailscale's range


def kind_of(ip):
    """'tailscale' | 'local' | 'internet' | 'loopback' for an IPv4 address."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return "local"
    if a.is_loopback:
        return "loopback"
    if a in CGNAT:
        return "tailscale"
    if a.is_private or a.is_link_local:
        return "local"
    return "internet"


def local_ips():
    out = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # picks the outgoing interface; no packet is sent
        out.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in out:
                out.append(ip)
    except OSError:
        pass
    return [ip for ip in out if not ip.startswith("127.")]


def tailscale_ip():
    """This PC's Tailscale address, if Tailscale is installed and connected."""
    exe = shutil.which("tailscale")
    if not exe and os.name == "nt":
        p = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tailscale", "tailscale.exe")
        exe = p if os.path.exists(p) else None
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5)
        ip = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
        return ip if ip and kind_of(ip) == "tailscale" else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def addresses(port, upnp=None):
    """[{"kind", "url", "note"}] best first: internet (if the router opened the port), tailscale, local."""
    out, seen = [], set()

    def add(kind, ip, note):
        if ip and ip not in seen:
            seen.add(ip)
            out.append({"kind": kind, "ip": ip, "url": f"http://{ip}:{port}", "note": note})

    if upnp and upnp.get("ok") and upnp.get("external_ip") and kind_of(upnp["external_ip"]) == "internet":
        add("internet", upnp["external_ip"], "from any network (router port opened automatically)")
    ts = tailscale_ip()
    add("tailscale", ts, "from any network, for PCs that also run Tailscale")
    for ip in local_ips():
        k = kind_of(ip)
        add(k if k != "internet" else "internet", ip,
            "from any network" if k == "internet" else
            "from any network, for PCs that also run Tailscale" if k == "tailscale" else "same network only")
    return out or [{"kind": "loopback", "ip": "127.0.0.1", "url": f"http://127.0.0.1:{port}", "note": "this PC only"}]


# ---------------------------------------------------------------- UPnP IGD
def _discover(timeout):
    """Description URLs of UPnP internet gateways that answer an SSDP search."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    s.settimeout(0.5)
    try:
        for st in SEARCH:
            msg = (f"M-SEARCH * HTTP/1.1\r\nHOST: {SSDP[0]}:{SSDP[1]}\r\nMAN: \"ssdp:discover\"\r\n"
                   f"MX: 2\r\nST: {st}\r\n\r\n").encode()
            try:
                s.sendto(msg, SSDP)
            except OSError:
                pass
        found, end = [], time.time() + timeout
        while time.time() < end:
            try:
                data, _ = s.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            m = re.search(rb"(?im)^location:\s*(\S+)", data)
            if m:
                loc = m.group(1).decode("latin-1")
                if loc not in found:
                    found.append(loc)
        return found
    finally:
        s.close()


def _control_url(location, timeout):
    """(control URL, service type) of the WAN connection service in a gateway's description."""
    with urllib.request.urlopen(location, timeout=timeout) as r:
        root = ET.fromstring(r.read())
    base = location
    for el in root.iter():
        if el.tag.endswith("URLBase") and (el.text or "").strip():
            base = el.text.strip()
    for svc in root.iter():
        if not svc.tag.endswith("service"):
            continue
        fields = {c.tag.split("}")[-1]: (c.text or "").strip() for c in svc}
        if fields.get("serviceType") in SERVICES and fields.get("controlURL"):
            return urllib.parse.urljoin(base, fields["controlURL"]), fields["serviceType"]
    return None, None


def _soap(url, service, action, args, timeout):
    body = "".join(f"<{k}>{v}</{k}>" for k, v in args.items())
    envelope = ('<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
                's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>'
                f'<u:{action} xmlns:u="{service}">{body}</u:{action}></s:Body></s:Envelope>').encode()
    req = urllib.request.Request(url, data=envelope, method="POST", headers={
        "Content-Type": 'text/xml; charset="utf-8"', "SOAPAction": f'"{service}#{action}"'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        m = re.search(r"<errorDescription>(.*?)</errorDescription>", text)
        raise RuntimeError(m.group(1) if m else f"router answered HTTP {e.code}")


def _local_ip_towards(host):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((host, 1900))
        return s.getsockname()[0]
    finally:
        s.close()


def upnp_open(port, description="Text LLM cluster", timeout=3.0, locations=None):
    """Forward TCP `port` on the router to this PC. Returns a dict:
    {"ok", "external_ip", "router", "control", "service", "lease", "error"}; never raises."""
    res = {"ok": False, "external_ip": None, "router": None, "error": None, "port": port}
    try:
        if locations is None and os.environ.get("TEXTLLM_UPNP_LOCATION"):
            # the router's description URL, for routers that don't answer the multicast search
            locations = [os.environ["TEXTLLM_UPNP_LOCATION"]]
        locs = locations if locations is not None else _discover(timeout)
        if not locs:
            res["error"] = "no UPnP router answered (UPnP may be switched off in the router's settings)"
            return res
        last = None
        for loc in locs:
            try:
                url, svc = _control_url(loc, timeout)
                if not url:
                    continue
                host = urllib.parse.urlparse(url).hostname
                me = _local_ip_towards(host)
                for lease in (0, 3600):  # some routers only take mappings that expire
                    try:
                        _soap(url, svc, "AddPortMapping", {
                            "NewRemoteHost": "", "NewExternalPort": port, "NewProtocol": "TCP",
                            "NewInternalPort": port, "NewInternalClient": me, "NewEnabled": 1,
                            "NewPortMappingDescription": description, "NewLeaseDuration": lease}, timeout)
                        res.update(ok=True, lease=lease)
                        break
                    except RuntimeError as e:
                        last = str(e)
                if not res["ok"]:
                    continue
                res.update(router=host, control=url, service=svc, internal_ip=me)
                try:
                    xml = _soap(url, svc, "GetExternalIPAddress", {}, timeout)
                    m = re.search(r"<NewExternalIPAddress>\s*([^<\s]+)\s*</NewExternalIPAddress>", xml)
                    res["external_ip"] = m.group(1) if m else None
                except RuntimeError:
                    pass
                ext = res["external_ip"]
                if ext and kind_of(ext) != "internet":
                    res["error"] = (f"the router's own internet address is {ext}, which is not public "
                                    "(a second router, or carrier NAT from your provider): use Tailscale")
                return res
            except (OSError, ET.ParseError, ValueError) as e:
                last = str(e)
        res["error"] = f"the router refused to open the port ({last})" if last else "no usable UPnP router found"
    except Exception as e:  # never let port mapping break the coordinator
        res["error"] = str(e)
    return res


def upnp_close(res, timeout=3.0):
    if not res or not res.get("ok"):
        return
    try:
        _soap(res["control"], res["service"], "DeletePortMapping",
              {"NewRemoteHost": "", "NewExternalPort": res["port"], "NewProtocol": "TCP"}, timeout)
    except Exception:
        pass


class PortKeeper:
    """Opens the port in the background, renews expiring mappings, and closes it on stop()."""

    def __init__(self, port, on_change=None):
        self.port, self.on_change, self.state = port, on_change, {"ok": False, "pending": True}
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            self.state = upnp_open(self.port)
            if self.on_change:
                self.on_change(self.state)
            # renew a mapping with a lease before it runs out; retry a failure now and then
            wait = 1800 if self.state.get("ok") and self.state.get("lease") else (None if self.state.get("ok") else 600)
            if wait is None or self._stop.wait(wait):
                return

    def stop(self):
        self._stop.set()
        upnp_close(self.state)
