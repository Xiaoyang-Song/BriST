"""In situ optical emission spectra (X), one integrated frame per layer.

Each frame = thermal continuum of the melt pool (Planck at T, scaled by
melt-pool area) + plume emission lines (Boltzmann-weighted, Stark-broadened)
+ glowing spatter + detector effects and build-level nuisances.
"""
from __future__ import annotations

import numpy as np

from .latent import N_LAYERS

N_CH = 1024
WL = np.linspace(350.0, 900.0, N_CH)       # nm
C2 = 1.4388e7                              # second radiation constant (nm K)

# (wavelength nm, species, upper-level energy eV, relative gA)
LINES = [
    (453.32, "Ti", 3.58, 1.0), (498.17, "Ti", 3.34, 1.6), (499.11, "Ti", 3.32, 1.3),
    (499.95, "Ti", 3.30, 1.0), (500.72, "Ti", 3.30, 0.8), (519.30, "Ti", 3.43, 0.5),
    (521.04, "Ti", 2.40, 0.35), (365.35, "Ti", 3.44, 0.9),
    (394.40, "Al", 3.14, 1.0), (396.15, "Al", 3.14, 2.0),
    (437.92, "V", 3.13, 1.0), (438.47, "V", 3.11, 0.7), (411.18, "V", 3.32, 0.5),
    (656.28, "H", 12.09, 1.0),
    (777.19, "O", 10.74, 1.0),
    (750.39, "Ar", 13.48, 0.6), (763.51, "Ar", 13.17, 1.0), (811.53, "Ar", 13.08, 1.2),
]


def planck(wl, T):
    return wl ** -5.0 / np.expm1(C2 / (wl * T))


_B_REF = planck(700.0, 2400.0)


def detector_response(wl):
    return np.exp(-0.5 * ((wl - 620) / 170) ** 2) * (0.2 + 0.8 / (1 + np.exp(-(wl - 380) / 15)))


_RESP = detector_response(WL)


def pseudo_voigt(wl, c, fwhm_g, fwhm_l):
    f = (fwhm_g ** 5 + 2.69269 * fwhm_g ** 4 * fwhm_l + 2.42843 * fwhm_g ** 3 * fwhm_l ** 2
         + 4.47163 * fwhm_g ** 2 * fwhm_l ** 3 + 0.07842 * fwhm_g * fwhm_l ** 4 + fwhm_l ** 5) ** 0.2
    r = fwhm_l / f
    eta = 1.36603 * r - 0.47719 * r ** 2 + 0.11116 * r ** 3
    x = wl - c
    g = np.exp(-4 * np.log(2) * x ** 2 / f ** 2) * (2 * np.sqrt(np.log(2) / np.pi) / f)
    lo = (f / (2 * np.pi)) / (x ** 2 + (f / 2) ** 2)
    return eta * lo + (1 - eta) * g


def render_spectra(rng, build, region, return_clean=False):
    """Return (N_LAYERS, N_CH) float32 spectrometer counts."""
    wl = WL + build["wl_shift"]                          # calibration error
    tau = 1.0 - build["lens_fouling"] * region["z_frac"]  # window soot transmission
    x = np.linspace(-1, 1, N_CH)
    stray = 20 * (build["stray"][0] + build["stray"][1] * x + build["stray"][2] * x ** 2) + 40
    out = np.zeros((N_LAYERS, N_CH))
    clean = np.zeros((N_LAYERS, N_CH))
    for l in range(N_LAYERS):
        e, T, vap = region["e"][l], region["T"][l], region["vapor"][l]
        ins = region["instab"][l]
        area = e ** 0.7
        cont = 3000 * area * planck(wl, T) / _B_REF

        # plume emission
        kTe = 0.55 + 0.25 * min(vap, 1.5)                 # plasma excitation temperature (eV)
        stark = 0.05 + 0.6 * min(vap, 1.5)                # Lorentzian FWHM (nm)
        fluct = rng.lognormal(0.0, 0.1 + 0.5 * ins)       # frame-to-frame plume fluctuation
        lines = np.zeros(N_CH)
        for c, sp, Eu, gA in LINES:
            if sp in ("Ti", "V"):
                a = 4000 * vap * gA * np.exp(-(Eu - 3.0) / kTe) * (0.6 if sp == "V" else 1.0)
            elif sp == "Al":                              # selective Al evaporation at high T
                a = 4000 * vap * gA * np.exp(-(Eu - 3.0) / kTe) * 0.9 * (1 + 0.8 * vap)
            elif sp == "Ar":
                a = 1500 * gA * vap ** 1.5
            elif sp == "O":
                a = 600 * gA * build["oxygen"] * (0.3 + vap)
            else:  # H from powder moisture dissociated in plume
                a = 500 * gA * build["moisture"] * (0.2 + vap)
            lines += a * pseudo_voigt(wl, c, 0.8, stark)
        lines *= fluct

        # spatter: hot ejected particles crossing the field of view
        spat = np.zeros(N_CH)
        for _ in range(rng.poisson(0.3 + 2.0 * region["s_kh"][l] + 0.5 * region["s_lof"][l])):
            # small glowing particles: a few % of melt-pool emission, scaling with energy input
            spat += rng.uniform(20, 150) * min(e, 2.0) * planck(wl, rng.uniform(2300, 3000)) / _B_REF

        sig = (cont + lines + spat) * _RESP * tau * build["gain"] + stray
        clean[l] = sig
        noisy = rng.poisson(np.clip(sig, 0, None)).astype(float) + rng.normal(0, 12, N_CH)
        out[l] = noisy + build["dark"]
    if return_clean:
        return out.astype(np.float32), clean.astype(np.float32)
    return out.astype(np.float32)
