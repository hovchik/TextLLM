"""netreach.py against a fake UPnP router (description XML + SOAP control endpoint)."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import netreach

DESC = """<?xml version="1.0"?><root xmlns="urn:schemas-upnp-org:device-1-0"><device>
<deviceType>urn:schemas-upnp-org:device:InternetGatewayDevice:1</deviceType><deviceList><device>
<deviceType>urn:schemas-upnp-org:device:WANDevice:1</deviceType><deviceList><device>
<deviceType>urn:schemas-upnp-org:device:WANConnectionDevice:1</deviceType><serviceList><service>
<serviceType>urn:schemas-upnp-org:service:WANIPConnection:1</serviceType>
<controlURL>/ctl/IPConn</controlURL></service></serviceList></device></deviceList></device></deviceList>
</device></root>"""


def fake_router(external_ip="81.16.10.5", refuse_permanent=False):
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/xml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, DESC)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"])).decode()
            action = self.headers["SOAPAction"].split("#")[1].strip('"')
            calls.append((action, body))
            if action == "AddPortMapping" and refuse_permanent and "<NewLeaseDuration>0<" in body:
                return self.send(500, "<s:Envelope><s:Body><s:Fault><detail><UPnPError><errorCode>725</errorCode>"
                                      "<errorDescription>OnlyPermanentLeasesSupported</errorDescription></UPnPError>"
                                      "</detail></s:Fault></s:Body></s:Envelope>")
            out = f"<NewExternalIPAddress>{external_ip}</NewExternalIPAddress>" if action == "GetExternalIPAddress" else ""
            self.send(200, f"<s:Envelope><s:Body><u:{action}Response>{out}</u:{action}Response></s:Body></s:Envelope>")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/desc.xml", calls


def test_opens_and_closes_the_port():
    srv, loc, calls = fake_router()
    try:
        r = netreach.upnp_open(7861, locations=[loc])
        assert r["ok"] and r["external_ip"] == "81.16.10.5" and not r["error"] and r["lease"] == 0
        add = next(b for a, b in calls if a == "AddPortMapping")
        assert "<NewExternalPort>7861</NewExternalPort>" in add and "<NewProtocol>TCP</NewProtocol>" in add
        addrs = netreach.addresses(7861, r)
        assert addrs[0] == {"kind": "internet", "ip": "81.16.10.5", "url": "http://81.16.10.5:7861",
                            "note": "from any network (router port opened automatically)"}
        netreach.upnp_close(r)
        assert calls[-1][0] == "DeletePortMapping"
    finally:
        srv.shutdown()


def test_falls_back_to_an_expiring_mapping():
    srv, loc, calls = fake_router(refuse_permanent=True)
    try:
        r = netreach.upnp_open(7861, locations=[loc])
        assert r["ok"] and r["lease"] == 3600
    finally:
        srv.shutdown()


def test_reports_double_nat():
    srv, loc, _ = fake_router(external_ip="100.72.1.9")  # provider's carrier NAT
    try:
        r = netreach.upnp_open(7861, locations=[loc])
        assert r["ok"] and "not public" in r["error"]
        assert all(a["kind"] != "internet" for a in netreach.addresses(7861, r))
    finally:
        srv.shutdown()


def test_no_router():
    r = netreach.upnp_open(7861, locations=[])
    assert not r["ok"] and "no UPnP router" in r["error"]


@pytest.mark.parametrize("ip,kind", [("100.101.5.6", "tailscale"), ("192.168.1.20", "local"), ("10.0.0.3", "local"),
                                     ("81.16.10.5", "internet"), ("127.0.0.1", "loopback")])
def test_address_kinds(ip, kind):
    assert netreach.kind_of(ip) == kind
