"""Parse circle/crack descriptions into claims and score them against CT ground truth."""
from __future__ import annotations

import re

import numpy as np

REGIONS = ["upper-left", "upper centre", "upper-right", "left side", "centre", "right side",
           "lower-left", "lower centre", "lower-right"]
ORIENTS = ["horizontal", "vertical", "diagonal (lower-left to upper-right)", "diagonal (upper-left to lower-right)",
           "point-like"]
SEV_ORD = {"acceptable": 0, "minor": 1, "significant": 2, "critical": 3}
NUMW = {"one": 1, "two": 2, "a": 1}


def parse(text):
    t = text.lower()
    out = dict(kind=None, severity=None, n=None, max_size=None, regions=[], orients=[], intersect=False, edge=False)
    no_crack = re.search(r"no crack|crack-free|cracks: none|without cracking|no cracking", t)
    if re.search(r"crack", t) and not no_crack:
        out["kind"] = "crack"
    elif re.search(r"pore", t) or no_crack:
        out["kind"] = "circle"
    if out["kind"] == "circle":
        out["severity"] = "acceptable"
    else:
        for s in ("critical", "significant", "minor"):
            if s in t:
                out["severity"] = s
                break
    m = re.search(r"count:\s*(\d)", t) or re.search(r"\b(one|two|\d)\s+(?:[a-z\-]+\s+){0,2}(?:pores?|cracks?)\b", t)
    if m:
        g = m.group(1)
        out["n"] = NUMW.get(g, int(g) if g.isdigit() else None)
    elif out["kind"] == "crack" and re.search(r"shows a ", t):
        out["n"] = 1
    t_nototal = re.sub(r"total crack length[^.|]*", "", t)
    sizes = [int(x) for x in re.findall(r"(\d+)\s*um", t_nototal)]
    if sizes:
        out["max_size"] = max(sizes)
    t_loc = re.sub(r"diagonal \((?:lower-left to upper-right|upper-left to lower-right)\)", "diagonal", t)
    spans = [(m.start(), m.end(), r) for r in REGIONS for m in re.finditer(re.escape(r), t_loc)]
    keep = [(a, r) for a, b, r in spans
            if not any(a2 <= a and b <= b2 and (b2 - a2) > (b - a) for a2, b2, _ in spans)]
    regs = [r for _, r in sorted(keep)]
    out["regions"] = regs
    pos = sorted((m.start(), o) for o in ORIENTS for m in re.finditer(re.escape(o), t)
                 if not (o == "diagonal" and False))
    out["orients"] = [o for _, o in pos]
    out["intersect"] = bool(re.search(r"intersect(?!ing: no)|cross each other", t)) and "intersecting: no" not in t
    out["edge"] = bool(re.search(r"edge of the field|boundary of the scanned", t))
    return out


def score(pred_texts, atts):
    rows = []
    for txt, a in zip(pred_texts, atts):
        p = parse(txt)
        r = dict(parsed=float(p["kind"] is not None))
        r["alarm_pred"] = p["kind"] == "crack"
        r["alarm_true"] = a["kind"] == "crack"
        r["type_acc"] = float(p["kind"] == a["kind"])
        r["sev_acc"] = float(p["severity"] == a["severity"])
        r["count_acc"] = float(p["n"] == a["n"])
        if p["max_size"] is not None:
            r["size_relerr"] = abs(p["max_size"] - a["max_size_um"]) / a["max_size_um"]
        r["loc_acc"] = float(bool(p["regions"]) and p["regions"][0] == a["items"][0]["region"])
        if a["kind"] == "crack":
            r["orient_acc"] = float(bool(p["orients"]) and p["orients"][0] == a["items"][0]["orient"])
            r["intersect_claim"] = float(p["intersect"]); r["intersect_true"] = float(a["intersect"])
            r["edge_claim"] = float(p["edge"]); r["edge_true"] = float(a["edge"])
        rows.append(r)
    return aggregate(rows), rows


def aggregate(rows):
    def mean(k):
        v = [r[k] for r in rows if k in r]
        return float(np.mean(v)) if v else float("nan")

    def pr(c, t):
        cl = np.array([r[c] for r in rows if c in r]); tr = np.array([r[t] for r in rows if t in r])
        if len(cl) == 0:
            return float("nan"), float("nan")
        tp = (cl * tr).sum()
        return (tp / cl.sum() if cl.sum() else float("nan")), (tp / tr.sum() if tr.sum() else float("nan"))
    out = dict(n=len(rows), parse_rate=mean("parsed"), type_acc=mean("type_acc"), sev_acc=mean("sev_acc"),
               count_acc=mean("count_acc"), loc_acc=mean("loc_acc"), orient_acc=mean("orient_acc"),
               size_relerr=float(np.median([r["size_relerr"] for r in rows if "size_relerr" in r]))
               if any("size_relerr" in r for r in rows) else float("nan"))
    out["intersect_prec"], out["intersect_rec"] = pr("intersect_claim", "intersect_true")
    out["edge_prec"], out["edge_rec"] = pr("edge_claim", "edge_true")
    ap = np.array([r["alarm_pred"] for r in rows]); at = np.array([r["alarm_true"] for r in rows])
    tp = (ap & at).sum()
    out["alarm_prec"] = tp / ap.sum() if ap.sum() else float("nan")
    out["alarm_rec"] = tp / at.sum() if at.sum() else float("nan")
    out["alarm_acc"] = float((ap == at).mean())
    return out


def fidelity(agg):
    return float(np.nanmean([agg["type_acc"], agg["sev_acc"], agg["count_acc"], agg["loc_acc"], agg["orient_acc"]]))


def alarm_score(text):
    p = parse(text)
    return float(SEV_ORD.get(p["severity"], 0)) if p["kind"] == "crack" else 0.0


# ------------------------------------------------------------------ claim-level view (selective transfer)
CLAIM_ATTRS = ["type", "severity", "count", "location", "orientation"]


def claims(text):
    """Categorical claims stated in a description (None = not stated)."""
    p = parse(text)
    return dict(type=p["kind"], severity=p["severity"], count=p["n"],
                location=p["regions"][0] if p["regions"] else None,
                orientation=(p["orients"][0] if p["orients"] else None) if p["kind"] == "crack" else None)


def gt_claims(att):
    return dict(type=att["kind"], severity=att["severity"], count=att["n"], location=att["items"][0]["region"],
                orientation=att["items"][0]["orient"] if att["kind"] == "crack" else None)
