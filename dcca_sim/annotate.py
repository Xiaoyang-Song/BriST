"""CT-observable attributes and expert-style descriptions for the circle/crack simulation.

Interpretation: each 32x32 image is a CT cross-section of a small region (1 px = 10 um,
320 x 320 um field). Circles are gas/keyhole-type pores (acceptable at these sizes);
lines are cracks (reject). Attributes are computed from the recorded geometry and the image,
so the text depends on the process only through what CT shows.
"""
from __future__ import annotations

import numpy as np

PX_UM = 10
SEV = ["acceptable", "minor", "significant", "critical"]   # crack severity; pores-only -> acceptable


def region_name(x, y, n=32):
    col = ["left", "centre", "right"][min(int(3 * x / n), 2)]
    row = ["upper", "middle", "lower"][min(int(3 * y / n), 2)]     # image row 0 = top
    if row == "middle" and col == "centre":
        return "centre"
    if row == "middle":
        return f"{col} side"
    if col == "centre":
        return f"{row} centre"
    return f"{row}-{col}"


def orientation(dx, dy):
    if dx == 0 and dy == 0:
        return "point-like"
    a = np.degrees(np.arctan2(-dy, dx)) % 180       # y axis points down in the image
    if a < 22.5 or a >= 157.5:
        return "horizontal"
    if 67.5 <= a < 112.5:
        return "vertical"
    return "diagonal (lower-left to upper-right)" if a < 90 else "diagonal (upper-left to lower-right)"


def _seg_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = ccw(p3, p4, p1), ccw(p3, p4, p2)
    d3, d4 = ccw(p1, p2, p3), ccw(p1, p2, p4)
    return (d1 * d2 <= 0) and (d3 * d4 <= 0)


def measure(img, params, n=32):
    att = dict(kind=params["kind"], area_pct=float(100 * img.mean()))
    items = []
    if params["kind"] == "circle":
        for c in params["items"]:
            (x, y), r = c["center"], c["radius"]
            items.append(dict(diam_um=(2 * r + 1) * PX_UM, region=region_name(x, y, n), x=x, y=y))
        overl = False
        if len(params["items"]) == 2:
            a, b = params["items"]
            dist = np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1])
            overl = bool(dist <= a["radius"] + b["radius"] + 1)
        items.sort(key=lambda t: -t["diam_um"])
        att.update(n=len(items), items=items, overlap=overl, severity="acceptable",
                   max_size_um=items[0]["diam_um"], defective=0)
    else:
        for c in params["items"]:
            (x0, y0), (x1, y1) = c["start"], c["end"]
            L = float(np.hypot(x1 - x0, y1 - y0) + 1) * PX_UM
            edge = any(v in (0, n - 1) for v in (x0, y0, x1, y1))
            items.append(dict(length_um=L, orient=orientation(x1 - x0, y1 - y0),
                              region=region_name((x0 + x1) / 2, (y0 + y1) / 2, n), edge=bool(edge),
                              p=[x0, y0, x1, y1]))
        inter = False
        if len(items) == 2:
            a, b = items[0]["p"], items[1]["p"]
            inter = bool(_seg_intersect(a[:2], a[2:], b[:2], b[2:]))
        items.sort(key=lambda t: -t["length_um"])
        tot = sum(t["length_um"] for t in items)
        sev = "minor" if tot < 100 else ("significant" if tot < 250 else "critical")
        if inter:
            sev = "critical"
        att.update(n=len(items), items=items, intersect=inter, edge=any(t["edge"] for t in items),
                   total_length_um=tot, max_size_um=items[0]["length_um"], severity=sev, defective=1)
    return att


def _r10(x):
    return int(10 * round(x / 10))


NUM = {1: "one", 2: "two"}


def describe(att, rng, style):
    ch = rng.choice
    if att["kind"] == "circle":
        it = att["items"]
        if style == "C":
            return (f"Defect: pore | Count: {att['n']} | Max size: {_r10(att['max_size_um'])} um | "
                    f"Location: {', '.join(t['region'] for t in it)} | Cracks: none | Severity: acceptable")
        if style == "B":
            return ch([f"Acceptable: {NUM[att['n']]} small pore{'s' if att['n'] > 1 else ''}, no cracks.",
                       f"No cracking. {NUM[att['n']].capitalize()} rounded pore{'s' if att['n'] > 1 else ''} "
                       f"(max ~{_r10(att['max_size_um'])} um); acceptable."])
        lead = ch(["No cracks detected; region acceptable.", "Acceptable region without cracking.",
                   "Crack-free; quality acceptable."])
        if att["n"] == 1:
            body = (f"CT shows one near-circular pore of ~{_r10(it[0]['diam_um'])} um diameter "
                    f"in the {it[0]['region']}.")
        else:
            body = (f"CT shows two near-circular pores of ~{_r10(it[0]['diam_um'])} and "
                    f"~{_r10(it[1]['diam_um'])} um diameter, in the {it[0]['region']} and {it[1]['region']}")
            body += ch([", partially merged into a cluster." if att["overlap"] else ", well separated.",
                        "; the pores overlap." if att["overlap"] else "."])
        return f"{lead} {body} Pore area fraction {att['area_pct']:.1f}%."

    it = att["items"]
    sev = att["severity"]
    if style == "C":
        return (f"Defect: crack | Count: {att['n']} | Max length: {_r10(att['max_size_um'])} um | "
                f"Orientation: {', '.join(t['orient'] for t in it)} | Location: {', '.join(t['region'] for t in it)} | "
                f"Intersecting: {'yes' if att.get('intersect') else 'no'} | Severity: {sev}")
    if style == "B":
        s = ch([f"{sev.capitalize()} cracking: ", f"Reject - {sev} crack defect: "])
        s += f"{NUM[att['n']]} crack{'s' if att['n'] > 1 else ''}, longest ~{_r10(att['max_size_um'])} um"
        s += ", intersecting." if att.get("intersect") else "."
        return s
    lead = ch([f"Crack defect detected; region rejected ({sev}).", f"{sev.capitalize()} cracking - reject.",
               f"Region contains {sev} crack damage and should be rejected."])
    parts = []
    for t in it:
        o = t["orient"]
        parts.append(f"a {_r10(t['length_um'])} um {o} crack in the {t['region']}" if o != "point-like"
                     else f"a point-like crack indication in the {t['region']}")
    body = "CT shows " + (parts[0] if len(parts) == 1 else f"two linear cracks: {parts[0]}, and {parts[1]}") + "."
    extra = []
    if att.get("intersect"):
        extra.append(ch(["The two cracks intersect.", "The cracks cross each other."]))
    if att["edge"]:
        extra.append(ch(["Cracking extends to the edge of the field of view.",
                         "At least one crack runs to the boundary of the scanned section."]))
    extra.append(f"Total crack length ~{_r10(att['total_length_um'])} um.")
    return " ".join([lead, body] + extra)
