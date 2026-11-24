"""Parse quality descriptions into factual claims and score them against CT ground truth."""
from __future__ import annotations

import re

import numpy as np

SEV_PATTERNS = [
    ("severe", r"severe|heavily defected|fail acceptance"),
    ("moderate", r"moderate"),
    ("minor", r"minor porosity|low defect content|severity:\s*minor"),
    ("none", r"fully dense|no detectable porosity|essentially dense|no defects detected|negligible porosity|severity:\s*none"),
]
TYPE_PATTERNS = {
    "keyhole": r"keyhole",
    "lack_of_fusion": r"lack[- ]of[- ]fusion|unfused",
    "gas": r"gas pore|gas porosity|\bgas\b",
}
NUMW = r"(\d+)"


def _first_sentence(t):
    return re.split(r"(?<=[.|])\s", t.strip(), maxsplit=1)[0]


def parse(text):
    t = text.lower()
    out = dict(severity=None, types=set(), dominant=None, counts={}, void=None,
               z_window=None, z_third=None, z_spread=False, lateral=False, max_size=None)
    # severity: first matching pattern in the lead sentence, else anywhere
    for scope in (_first_sentence(t), t):
        for sev, pat in SEV_PATTERNS:
            if re.search(pat, scope):
                out["severity"] = sev
                break
        if out["severity"]:
            break
    # defect types, in order of first mention after "shows"/"dominated by"
    body = t.split("ct shows", 1)[-1] if "ct shows" in t else t
    pos = {}
    for k, pat in TYPE_PATTERNS.items():
        m = re.search("(?:" + pat + ")", body)
        if m and not re.search(r"\bno (?:porosity or )?(?:" + pat + ")", body):
            pos[k] = m.start()
            out["types"].add(k)
    if out["severity"] == "none" and "gas" not in out["types"]:
        out["types"] = set()
        pos = {}
    if pos:
        out["dominant"] = min(pos, key=pos.get)
        if "dominated by" in t:
            dm = t.split("dominated by", 1)[1]
            for k, pat in TYPE_PATTERNS.items():
                if re.match(r"\s*(some |small |minor )?" + "(" + pat + ")", dm):
                    out["dominant"] = k
    # counts: "<n> ... keyhole", "<n> ... lack-of-fusion / unfused", "<n> ... gas"
    for k, pat in TYPE_PATTERNS.items():
        m = re.search(NUMW + r"\s+(?:[a-z,\-]+\s+){0,4}?(?:" + pat + ")", body)
        if not m and k == "gas":
            m = re.search(r"\((\d+) pores?", body)
        if m:
            out["counts"][k] = int(m.group(1))
    m = re.search(r"void fraction:?\s*(<)?\s*([\d.]+)\s*%", t)
    if m:
        out["void"] = 0.005 if m.group(1) else float(m.group(2))
    m = re.search(r"layers? (\d+)(?:\s*-\s*(\d+))?", t)
    if m:
        a = int(m.group(1)); b = int(m.group(2) or a)
        out["z_window"] = (a - 1, b - 1)
    m = re.search(r"(lower|middle|upper) third", t)
    if m:
        out["z_third"] = ["lower", "middle", "upper"].index(m.group(1))
    if re.search(r"throughout|full region height|no clear vertical", t):
        out["z_spread"] = True
    if re.search(r"cluster", t):
        out["lateral"] = True
    sizes = [int(x) for x in re.findall(r"(\d+)\s*um", t)]
    if sizes:
        out["max_size"] = max(sizes)
    return out


def gt_claims(att):
    """Ground truth in the same claim space (from CT label analysis)."""
    types = set(att["types"].keys())
    dom = att["dominant"] if att["dominant"] != "none" else None
    zt = None
    if att["z_window"] is not None and not att["z_localized"]:
        zs = np.asarray(att["z_share"])
        zt = int(zs.argmax()) if zs.max() > 0.6 else None
    return dict(severity=att["severity"], types=types, dominant=dom,
                counts={k: v["count"] for k, v in att["types"].items()},
                void=att["void_fraction_pct"], z_window=tuple(att["z_window"]) if att["z_localized"] else None,
                z_third=zt, z_any=att["z_window"], lateral=att["xy_clustered"], max_size=att["max_size_um"])


SEV_ORD = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}


def score(pred_texts, atts):
    """Aggregate description-fidelity metrics over a set of samples."""
    rows = []
    for txt, att in zip(pred_texts, atts):
        p, g = parse(txt), gt_claims(att)
        r = {}
        r["parsed"] = p["severity"] is not None
        r["sev_acc"] = float(p["severity"] == g["severity"])
        r["sev_within1"] = float(p["severity"] is not None and abs(SEV_ORD[p["severity"]] - SEV_ORD[g["severity"]]) <= 1)
        r["alarm_pred"] = p["severity"] in ("moderate", "severe")
        r["alarm_true"] = g["severity"] in ("moderate", "severe")
        r["dom_acc"] = float(p["dominant"] == g["dominant"])
        # type-level factual precision / recall (ignore gas: sub-resolution, weakly observable)
        for name, keep in (("major", {"keyhole", "lack_of_fusion"}), ("all", {"keyhole", "lack_of_fusion", "gas"})):
            P, G = p["types"] & keep, g["types"] & keep
            r[f"tp_{name}"], r[f"np_{name}"], r[f"ng_{name}"] = len(P & G), len(P), len(G)
        # void fraction (log10 error, floor 0.005%)
        if p["void"] is not None:
            r["void_logerr"] = abs(np.log10(max(p["void"], 5e-3)) - np.log10(max(g["void"], 5e-3)))
        # vertical localisation: predicted window overlaps the GT defect window (when GT is localised)
        if g["z_window"] is not None and g["severity"] != "none":
            r["gt_localized"] = 1
            if p["z_window"] is not None:
                a, b = p["z_window"]; c, d = g["z_window"]
                inter = max(0, min(b, d) - max(a, c) + 1); uni = max(b, d) - min(a, c) + 1
                r["z_iou"] = inter / uni
                r["z_hit"] = float(inter > 0)
            else:
                r["z_iou"] = 0.0; r["z_hit"] = 0.0
        if p["z_window"] is not None and g["z_any"] is not None and g["z_window"] is None:
            r["z_false_local"] = 1.0       # claimed a narrow band that does not exist
        # dominant count relative error
        if g["dominant"] and g["dominant"] in p["counts"] and g["dominant"] in g["counts"]:
            r["count_relerr"] = abs(p["counts"][g["dominant"]] - g["counts"][g["dominant"]]) / max(g["counts"][g["dominant"]], 1)
        # lateral claims: not observable from X; any claim is supported only if GT is clustered
        r["lateral_claim"] = float(p["lateral"])
        r["lateral_unsupported"] = float(p["lateral"] and not g["lateral"])
        rows.append(r)
    return aggregate(rows), rows


def aggregate(rows):
    def mean(k):
        v = [r[k] for r in rows if k in r]
        return float(np.mean(v)) if v else float("nan")
    out = dict(n=len(rows), parse_rate=mean("parsed"), sev_acc=mean("sev_acc"), sev_within1=mean("sev_within1"),
               dom_acc=mean("dom_acc"), void_logerr=mean("void_logerr"), z_iou=mean("z_iou"), z_hit=mean("z_hit"),
               z_false_local=float(np.sum([r.get("z_false_local", 0) for r in rows]) / max(1, len(rows))),
               count_relerr=float(np.median([r["count_relerr"] for r in rows if "count_relerr" in r]))
               if any("count_relerr" in r for r in rows) else float("nan"),
               lateral_claim_rate=mean("lateral_claim"), lateral_unsupported_rate=mean("lateral_unsupported"))
    for name in ("major", "all"):
        tp = sum(r[f"tp_{name}"] for r in rows); npred = sum(r[f"np_{name}"] for r in rows)
        ng = sum(r[f"ng_{name}"] for r in rows)
        out[f"type_prec_{name}"] = tp / npred if npred else float("nan")
        out[f"type_rec_{name}"] = tp / ng if ng else float("nan")
    ap = np.array([r["alarm_pred"] for r in rows]); at = np.array([r["alarm_true"] for r in rows])
    tp = (ap & at).sum()
    out["alarm_prec"] = tp / ap.sum() if ap.sum() else float("nan")
    out["alarm_rec"] = tp / at.sum() if at.sum() else float("nan")
    out["alarm_acc"] = float((ap == at).mean())
    return out


# ------------------------------------------------------------------ claim-level view (selective transfer)
CLAIM_ATTRS = ["severity", "dominant", "vertical", "lateral"]


def _vertical(p_or_g, is_gt=False):
    if is_gt:
        if p_or_g["z_window"] is not None:
            a, b = p_or_g["z_window"]
            return f"band@{'lower' if (a + b) / 2 < 16 / 3 else 'middle' if (a + b) / 2 < 32 / 3 else 'upper'}"
        return ["lower", "middle", "upper"][p_or_g["z_third"]] if p_or_g["z_third"] is not None else "spread"
    if p_or_g["z_window"] is not None:
        a, b = p_or_g["z_window"]
        return f"band@{'lower' if (a + b) / 2 < 16 / 3 else 'middle' if (a + b) / 2 < 32 / 3 else 'upper'}"
    if p_or_g["z_third"] is not None:
        return ["lower", "middle", "upper"][p_or_g["z_third"]]
    return "spread" if p_or_g["z_spread"] else None


def claims(text):
    p = parse(text)
    return dict(severity=p["severity"], dominant=p["dominant"], vertical=_vertical(p),
                lateral=True if p["lateral"] else None)


def gt_claims_cat(att):
    g = gt_claims(att)
    return dict(severity=g["severity"], dominant=g["dominant"],
                vertical=_vertical(g, True) if g["severity"] != "none" else None, lateral=g["lateral"])
