"""Dataset adapters: one interface for the LPBF simulation and the circle/crack (DCCA-paper) simulation.

Common interface used by all stage scripts:
  N, split, annotated, defective, meta[i]["attributes"], text_all[i] = {"text", "annotator"}
  idx(split), annotated_idx(split, ratio, seed)
  score(texts, atts) -> (agg, rows); fidelity(agg) -> float
  alarm_score(text) -> float; alarm_threshold
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import textparse as tp_lpbf


class _Base:
    def idx(self, split):
        return np.where(self.split == split)[0]

    def annotated_idx(self, split="train", ratio=None, seed=0):
        pool = self.idx(split)
        if ratio is None:
            return pool[self.annotated[pool]]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(pool)
        return np.sort(perm[: max(1, int(round(ratio * len(pool))))])


class LPBF(_Base):
    name = "lpbf"
    alarm_threshold = 1.5            # moderate / severe

    def __init__(self, root):
        from .data import SynthData
        s = SynthData(root)
        self.__dict__.update(s.__dict__)
        self.h5 = s.h5

    score = staticmethod(tp_lpbf.score)

    @staticmethod
    def fidelity(agg):
        return float(np.nanmean([agg["sev_acc"], agg["dom_acc"], agg["type_prec_major"], agg["type_rec_major"],
                                 agg["z_hit"]]))

    @staticmethod
    def alarm_score(text):
        return float(tp_lpbf.SEV_ORD.get(tp_lpbf.parse(text)["severity"], 1))

    quant_name = "void_fraction"

    @staticmethod
    def quant(text, att):
        """(stated, true) key quantity used for the specificity analysis"""
        return tp_lpbf.parse(text)["void"], att["void_fraction_pct"]


class CircleCrack(_Base):
    name = "circlecrack"
    alarm_threshold = 0.5            # any crack

    def __init__(self, root):
        from dcca_sim import textparse as tp
        self._tp = tp
        self.root = root
        self.meta = [json.loads(l) for l in open(os.path.join(root, "metadata.jsonl"))]
        self.N = len(self.meta)
        self.text_all = {m["idx"]: dict(text=m["text"], annotator=m["annotator"]) for m in self.meta}
        self.split = np.array([m["split"] for m in self.meta], dtype="<U8")
        # generation is the expensive step: evaluate on fixed subsets (first 300 val, first 1000 test)
        for sp, k in (("val", 300), ("test", 1000)):
            ids = np.where(self.split == sp)[0]
            self.split[ids[k:]] = sp + "_x"
        self.annotated = np.array([m["annotated"] for m in self.meta])
        self.defective = np.array([m["attributes"]["defective"] for m in self.meta], dtype=np.int64)
        self.severity = np.array([m["attributes"]["severity"] for m in self.meta])

    def score(self, texts, atts):
        return self._tp.score(texts, atts)

    def fidelity(self, agg):
        return self._tp.fidelity(agg)

    def alarm_score(self, text):
        return self._tp.alarm_score(text)

    quant_name = "max_defect_size"

    def quant(self, text, att):
        return self._tp.parse(text)["max_size"], att["max_size_um"]


def load_dataset(root):
    if os.path.exists(os.path.join(root, "dataset.h5")):
        return LPBF(root)
    return CircleCrack(root)
