"""Shared utilities for the CRYOWRF remake notebooks.

This module collects everything the three validation notebooks need:

* event configuration (`SIMULATIONS`) and per-event instrument heights
  (`INSTRUMENT_HEIGHTS`, from Table 1.1);
* low-level helpers for the time-varying fine mesh (`interp_wind_sfc`,
  `read_sn_c_height_at_t`);
* data builders (one per event) and the matching plotting functions.

The underlying interpolation code lives in ``funcs_interp_vars`` and
``funcs_remote_obs`` in the parent ``wrf_notebooks`` folder; this module just
imports and orchestrates it.

Three switches drive the notebooks:

    SIM_KEY    selected event (used when RUN_ALL is False)
    RUN_ALL    True  -> run every event in SIMULATIONS
               False -> run only SIM_KEY
    SAVE_PLOTS True  -> write figures to OUTPUT_DIR

Use ``get_run_keys(SIM_KEY, RUN_ALL)`` to expand the selection and
``savefig(fig, name, save=SAVE_PLOTS)`` to honour the save switch.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from netCDF4 import Dataset as _NC
from scipy.stats import gaussian_kde, pearsonr
from scipy.stats import norm as _norm

# Make the support modules in the parent folder importable.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import funcs_interp_vars as fiv      # noqa: E402
import funcs_remote_obs as fro       # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SIMULATIONS = {
    "cold_kat": {
        "sim_dir":  r"D:\cold_kat_CRYOWRF",
        "start":    "2024-05-13 01:00",
        "end":      "2024-05-14 14:00",
        "aws_file": r"D:\data\slowdata_cleaned\slowdata_cleaned_2024.pkl",
        "spc_file": r"D:\data\spc2024.pkl",
        "label":    "Cold katabatic - May 2024",
        "color":    "#1f77b4",
    },
    "trans_syn": {
        "sim_dir":  r"D:\trans_syn_CRYOWRF",
        "start":    "2025-04-23 05:00",
        "end":      "2025-04-24 00:00",
        "aws_file": r"D:\data\slowdata_cleaned\slowdata_cleaned_2025.pkl",
        "spc_file": r"D:\data\spc2025.pkl",
        "label":    "Transitional synoptic - Apr 2025",
        "color":    "#ff7f0e",
    },
    "warm_syn": {
        "sim_dir":  r"D:\warm_syn_CRYOWRF",
        "start":    "2025-04-24 12:00",
        "end":      "2025-04-25 12:00",
        "aws_file": r"D:\data\slowdata_cleaned\slowdata_cleaned_2025.pkl",
        "spc_file": r"D:\data\spc2025.pkl",
        "label":    "Warm synoptic - Apr 2025",
        "color":    "#d62728",
    },
}

# Instrument heights above the snow surface [m], from Table 1.1 survey
# measurements.
#   spc_z   : SPC sensor centre height
#   fc_z_lo : FlowCapt lower bar height (upper bound = lower + 1 m sensor length)
#   fc_z_hi : FlowCapt upper bound
#   ws_z    : Young wind 2 (lower anemometer) height
INSTRUMENT_HEIGHTS = {
    "cold_kat":  {"spc_z": 0.598, "fc_z_lo": 0.168, "fc_z_hi": 1.168, "ws_z": 1.748},
    "trans_syn": {"spc_z": 0.488, "fc_z_lo": 0.148, "fc_z_hi": 1.148, "ws_z": 1.768},
    "warm_syn":  {"spc_z": 0.488, "fc_z_lo": 0.148, "fc_z_hi": 1.148, "ws_z": 1.768},
}

# Saltation lower boundary condition (model parameter, NOT an instrument height).
H_SALT       = 0.15    # [m]
LAMBDA_CSALT = 0.45

DEFAULT_DOMAIN = "d05"
OUTPUT_DIR     = r"C:\Users\lory2\Desktop\tesi\wrf_notebooks\validation_plots"
DPI            = 180

# Observation column names / alignment.
SPC_FLUX_COL      = "Corrected Mass Flux(kg/m^2/s)"
FC_FLUX_COL       = "PF_FC4"
SPC_BIN_DIAMETERS = None       # auto-detect from column names
ALIGN_WINDOW_MIN  = 10

# SPC loading: the yearly SPC pickles are too large to be read repeatedly.
# Each file is read in full only once, cut to the event windows (+/- SPC_MARGIN)
# and saved next to the original as <name>_events.pkl; later calls use the
# small file (kept in memory for the rest of the session).
SPC_MARGIN = pd.Timedelta("1D")
_SPC_CACHE = {}


def _spc_windows(pkl_file):
    """Event windows (with margin) of all simulations that use `pkl_file`."""
    return [(pd.Timestamp(s["start"]) - SPC_MARGIN, pd.Timestamp(s["end"]) + SPC_MARGIN)
            for s in SIMULATIONS.values() if s["spc_file"] == pkl_file]


def _read_pickle_mmap(pkl_file):
    """Unpickle a (protocol 5) DataFrame pickle mapping its large data buffers
    from disk (np.memmap) instead of reading them into RAM, so that files larger
    than the available memory can be opened and cut."""
    import pickle
    import struct

    class _MMUnpickler(pickle._Unpickler):
        dispatch = dict(pickle._Unpickler.dispatch)

        def _load_bytearray8(self):
            n, = struct.unpack("<Q", self.read(8))
            fr = self._unframer.current_frame
            if fr is not None and fr.tell() < len(fr.getbuffer()):
                b = bytearray(n)              # small buffer inside a frame
                self.readinto(b)
                self.append(b)
                return
            off = fh.tell()
            fh.seek(n, 1)
            self.append(np.memmap(pkl_file, dtype=np.uint8, mode="r", offset=off, shape=(n,))
                        if n else bytearray())
        dispatch[pickle.BYTEARRAY8[0]] = _load_bytearray8

    with open(pkl_file, "rb") as fh:
        df = _MMUnpickler(fh).load()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_spc_events(pkl_file):
    """SPC data restricted to the event windows (copy, safe to modify)."""
    if pkl_file not in _SPC_CACHE:
        import gc
        wins = _spc_windows(pkl_file)
        small = os.path.splitext(pkl_file)[0] + "_events.pkl"
        df = None
        if os.path.exists(small) and os.path.getmtime(small) >= os.path.getmtime(pkl_file):
            df = pd.read_pickle(small)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            covered = len(df) and all(df.index.min() <= a and df.index.max() >= b - SPC_MARGIN
                                      for a, b in wins)
            if not covered:
                df = None
        if df is None:
            try:
                full = _read_pickle_mmap(pkl_file)       # low-memory read
            except MemoryError:
                raise
            except Exception as e:                        # e.g. older pickle protocol
                print(f"SPC: memory-mapped read failed ({e}); using pd.read_pickle")
                full = fiv.load_spc(pkl_file)
            df = pd.concat([full[(full.index >= a) & (full.index <= b)] for a, b in wins]).sort_index()
            df = df[~df.index.duplicated(keep="first")].copy()
            del full
            gc.collect()
            df.to_pickle(small)
            print(f"SPC: cut {os.path.basename(pkl_file)} to the event windows "
                  f"({len(df)} rows) -> {small}")
        _SPC_CACHE[pkl_file] = df
    return _SPC_CACHE[pkl_file].copy()

# EddyPro u* files (section 8 of the blowing-snow notebook).
EDDY_USTAR_FILES = {
    2025: (r"Y:\CRYOS\Projects\Antarctica\PrincessElisabeth\Season_2025-2026"
           r"\DATA\MET\processed_HF\2025\SFC\Eddypro_ouput\202504"
           r"\eddypro_Apr2025_full_output_2026-02-23T212943_adv.csv"),
    2024: (r"Y:\CRYOS\Projects\Antarctica\PrincessElisabeth\Season_2024-2025"
           r"\DATA\processed\SFC_DR\Eddypro_output\202405"
           r"\eddypro_May2024_full_output_2025-05-22T145422_adv.csv"),
}

# Styling.
# Per-event colour mapping comes from SIMULATIONS[...]["color"]
#   cold_kat -> blue, trans_syn -> orange, warm_syn -> red.
# Observations use a neutral dark grey that reads cleanly against all three
# simulation colours; model curves/points take the event colour.
COLOR_OBS  = "#444444"   # observations (AWS / SPC / FlowCapt / ceilometer)
COLOR_AWS  = "#444444"   # alias kept for readability
COLOR_WRF  = "#FF37A6"   # fallback model colour when no single event applies
COLOR_SPC  = "#2ca02c"
COLOR_FC   = "#9467bd"
ALPHA_PDF  = 0.35
ALPHA_SC   = 0.55
SC_SIZE    = 25
GRID_COLOR = "#cccccc"
GRID_LW    = 0.9
LABEL_FS   = 13
TICK_FS    = 11

# Global figure-size multiplier. Lower => smaller figures with the same font
# sizes, so the text reads larger relative to the plot.
FIG_SCALE = 0.72


def _fs(w, h):
    """Scale a (width, height) figsize tuple by FIG_SCALE."""
    return (w * FIG_SCALE, h * FIG_SCALE)


def event_color(sim_key):
    """Model colour for an event (cold=blue, trans=orange, warm=red)."""
    return SIMULATIONS[sim_key]["color"]

# Convenient unicode bits for axis labels.
_SUP_M2 = "⁻\xb2"          # superscript -2
_SUP_M1 = "⁻\xb9"          # superscript -1
FLUX_UNIT = f"g m{_SUP_M2} s{_SUP_M1}"
WS_UNIT   = f"m s{_SUP_M1}"


# ─────────────────────────────────────────────────────────────────────────────
# Run / save plumbing
# ─────────────────────────────────────────────────────────────────────────────

def get_run_keys(sim_key="trans_syn", run_all=False):
    """Expand the run selection into a list of event keys."""
    if run_all:
        return list(SIMULATIONS.keys())
    if sim_key not in SIMULATIONS:
        raise KeyError(f"Unknown SIM_KEY {sim_key!r}; choose from {list(SIMULATIONS)}")
    return [sim_key]


def savefig(fig, name, save=True, output_dir=OUTPUT_DIR, dpi=DPI):
    """Save *fig* under *output_dir/name* only when *save* is True."""
    if not save:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fine-mesh helpers (time-varying SN_C_HEIGHT)
# ─────────────────────────────────────────────────────────────────────────────

def read_sn_c_height_at_t(nc, lats, lons, t_idx):
    """Return the 8 fine-mesh level heights [m AGL] at PEA for timestep t_idx."""
    snc = nc.variables["SN_C_HEIGHT"]
    return np.array([
        fiv._interp_pt(fiv._build_interp(lats, lons, np.asarray(snc[t_idx, k], float)),
                       fiv.PEA_LAT, fiv.PEA_LON)
        for k in range(snc.shape[1])
    ])


def interp_wind_sfc(wrf_files, target_z, lat_pt=fiv.PEA_LAT, lon_pt=fiv.PEA_LON):
    """Wind speed [m/s] at the fine-mesh level closest to *target_z* [m AGL].

    Level selection uses the time-varying SN_C_HEIGHT so burial or erosion of
    the snow surface is accounted for at each timestep. VW_SFC is a scalar speed
    magnitude; wind direction is not available on the fine mesh and must be taken
    from the WRF 10 m diagnostic.

    Returns
    -------
    times    : list of pd.Timestamp
    ws       : np.ndarray [m/s]
    h_actual : np.ndarray [m]   height of the chosen level at each timestep
    """
    times, ws_out, h_out = [], [], []
    for f in wrf_files:
        nc = _NC(f)
        if "VW_SFC" not in nc.variables:
            nc.close(); continue
        ft     = fiv._decode_wrf_times(nc)
        la, lo = fiv._get_latlon(nc)
        vw     = nc.variables["VW_SFC"]
        snc    = nc.variables["SN_C_HEIGHT"]
        for ti in range(len(ft)):
            h_t = np.array([
                fiv._interp_pt(fiv._build_interp(la, lo, np.asarray(snc[ti, k], float)),
                               lat_pt, lon_pt)
                for k in range(snc.shape[1])
            ])
            lev = int(np.argmin(np.abs(h_t - target_z)))
            ws_out.append(fiv._interp_pt(
                fiv._build_interp(la, lo, np.asarray(vw[ti, lev], float)), lat_pt, lon_pt))
            h_out.append(h_t[lev])
        times.extend(ft)
        nc.close()
    return times, np.array(ws_out), np.array(h_out)


def _ws_series(times, ws):
    """Build a tz-naive Series from interp_wind_sfc output, dropping any
    duplicate timestamps (CRYOWRF output files can overlap at boundaries)."""
    s = pd.Series(ws, index=pd.DatetimeIndex(times).tz_localize(None)).sort_index()
    return s[~s.index.duplicated(keep="first")]


def enumerate_times(files):
    """List (timestamp, file, time_index) for every timestep across *files*."""
    recs = []
    for f in files:
        with _NC(f) as nc:
            if nc.dimensions["Time"].size == 0:
                continue
            for i, t in enumerate(fiv._decode_wrf_times(nc)):
                recs.append((pd.Timestamp(t).tz_localize(None), f, i))
    return recs


def obs_at_time(df, col, ts, window_min=ALIGN_WINDOW_MIN):
    """Mean of *col* in *df* within +/- window/2 around *ts* (NaN if empty)."""
    if col not in df.columns:
        return np.nan
    half = pd.Timedelta(minutes=window_min / 2)
    w = df.loc[(df.index >= ts - half) & (df.index < ts + half), col]
    return float(w.mean()) if len(w) else np.nan


def salt_flux_vec(snow_drift, ust):
    """Saltation mass flux at H_SALT from SNOW_DRIFT and UST (vectorised)."""
    snow_drift = np.asarray(snow_drift, float)
    ust = np.asarray(ust, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(ust > 0,
                         np.maximum(LAMBDA_CSALT * 9.81 / ust ** 2, 1.0 / 0.04),
                         1.0 / 0.04)
    return (snow_drift * alpha * np.exp(-alpha * H_SALT)).clip(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finite_pair(a, b):
    # Align on the common index when both are pandas Series (their lengths or
    # timestamps may differ); fall back to positional otherwise.
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        a = a[~a.index.duplicated(keep="first")]
        b = b[~b.index.duplicated(keep="first")]
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
        a, b = joined["a"].to_numpy(float), joined["b"].to_numpy(float)
    else:
        a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def mae(o, m, circular=False):
    o, m = np.asarray(o, float), np.asarray(m, float)
    d = (m - o + 180) % 360 - 180 if circular else m - o
    return np.nanmean(np.abs(d))


def bias(o, m, circular=False):
    if circular:
        return float("nan")
    return np.nanmean(np.asarray(m, float) - np.asarray(o, float))


def corr(o, m):
    a, b = _finite_pair(o, m)
    return pearsonr(a, b)[0] if len(a) >= 3 else float("nan")


def circ_corr(o, m):
    """Circular-circular correlation for angles in DEGREES.

    Jammalamadaka & SenGupta coefficient:

        r_c = sum(sin(a_i - a_bar) sin(b_i - b_bar))
              / sqrt( sum(sin^2(a_i - a_bar)) * sum(sin^2(b_i - b_bar)) )

    where a_bar, b_bar are the circular means. A plain Pearson correlation is
    invalid for a circular variable (e.g. directions oscillating across 0/360
    would give a meaningless linear variance), hence this is used for wind
    direction. Returns a value in [-1, 1].
    """
    a, b = _finite_pair(o, m)
    if len(a) < 3:
        return float("nan")
    ar, br = np.deg2rad(a), np.deg2rad(b)
    a_bar = np.arctan2(np.mean(np.sin(ar)), np.mean(np.cos(ar)))
    b_bar = np.arctan2(np.mean(np.sin(br)), np.mean(np.cos(br)))
    sa, sb = np.sin(ar - a_bar), np.sin(br - b_bar)
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(np.sum(sa * sb) / den) if den > 0 else float("nan")


def rmse(o, m, circular=False):
    """Root-mean-square error (circular-aware for direction)."""
    a, b = _finite_pair(o, m)
    d = (b - a + 180) % 360 - 180 if circular else b - a
    return float(np.sqrt(np.mean(d ** 2))) if len(a) else float("nan")


def pbias(o, m):
    """Percent bias = 100 * sum(mod - obs) / sum(obs).

    Errore sistematico in percentuale: valori positivi = il modello sovrastima,
    negativi = sottostima. Metrica standard in validazione di modelli climatici.
    Restituisce NaN se la somma delle osservazioni e' 0 (percentuale indefinita).
    """
    a, b = _finite_pair(o, m)
    denom = float(np.sum(a))
    return float(100.0 * np.sum(b - a) / denom) if (len(a) and denom != 0) else float("nan")


def nrmse(o, m, norm="mean"):
    """RMSE normalizzato in percentuale.

    NRMSE% = 100 * RMSE / D, con D = media delle osservazioni ("mean", default),
    deviazione standard ("std") o range max-min ("range"). Sconsigliato per
    variabili con zero convenzionale (temperatura in deg C, direzione del vento).
    """
    a, b = _finite_pair(o, m)
    if len(a) == 0:
        return float("nan")
    if norm == "std":
        denom = float(np.std(a))
    elif norm == "range":
        denom = float(np.max(a) - np.min(a))
    else:  # "mean"
        denom = float(np.mean(a))
    r = float(np.sqrt(np.mean((b - a) ** 2)))
    return float(100.0 * r / denom) if denom != 0 else float("nan")


def kde_curve(data, n_pts=400):
    d = np.asarray(data, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.array([]), np.array([])
    kde = gaussian_kde(d, bw_method="scott")
    pad = (d.max() - d.min()) * 0.1 or 1.0
    x = np.linspace(d.min() - pad, d.max() + pad, n_pts)
    return x, kde(x)


def mask_direction_wraps(vals, threshold=180):
    """Insert NaNs across 0/360 wraps so direction lines don't draw verticals."""
    v = np.asarray(vals, float).copy()
    idx = np.where(np.isfinite(v))[0]
    if len(idx) > 1:
        for b in np.where(np.abs(np.diff(v[idx])) > threshold)[0]:
            v[idx[b + 1]] = np.nan
    return v


# ═════════════════════════════════════════════════════════════════════════════
# 1) AWS surface validation  (validation_plot.ipynb)
# ═════════════════════════════════════════════════════════════════════════════

def build_merged_event(sim_key, domain=DEFAULT_DOMAIN):
    """Merge WRF surface diagnostics + fine-mesh wind with aligned AWS data."""
    sim     = SIMULATIONS[sim_key]
    ws_z    = INSTRUMENT_HEIGHTS[sim_key]["ws_z"]
    files   = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])

    wrf_full = fiv.build_wrf_df(domain=domain, input_dir=sim["sim_dir"])
    tt, ws_fm, h_fm = interp_wind_sfc(files, target_z=ws_z)
    wrf_full["ws_sfc_ms"] = _ws_series(tt, ws_fm).reindex(wrf_full.index)

    aws_h = fiv.align_aws(fiv.load_aws(sim["aws_file"]), wrf_full, window_min=10)
    aws_h["rhw_pct"] = aws_h["RH"]

    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    merged = wrf_full.loc[t0:t1].merge(aws_h.loc[t0:t1], left_index=True,
                                       right_index=True, suffixes=("_wrf", "_aws"))
    merged.attrs["h_fm_mean"] = float(np.nanmean(h_fm))
    merged.attrs["ws_z"]      = ws_z
    merged.attrs["sim_key"]   = sim_key
    return merged


# (obs_col, mod_col, label, unit, circular, pct_ok, pct_offset)
# pct_ok  = False per la direzione del vento (variabile circolare): PBIAS/NRMSE
#           non sono interpretabili e restano vuoti.
# pct_offset = costante sommata a obs e mod PRIMA di calcolare PBIAS/NRMSE, per
#           riportarli su una scala di rapporti con zero fisico. La temperatura
#           e' in deg C (zero convenzionale): si converte in Kelvin (+273.15)
#           cosi' la percentuale e' ben definita. MAE/Bias restano in deg C.
VAL_VARS = [
    ("psfc_hPa_aws", "psfc_hPa_wrf",  "Surface pressure",     "hPa",   False, True,  0.0),
    ("theta_C",      "theta2_C",  "Potential temperature", "deg C", False, True,  273.15),
    ("WS2_Avg",      "ws_sfc_ms", "Wind speed at AWS height", WS_UNIT, False, True,  0.0),
    ("WD2",          "wd10_deg",  "Wind direction",       "deg",   True,  False, 0.0),
    ("rhw_pct",      "rhw2_pct",  "Relative humidity",    "%",     False, True,  0.0),
]


def metrics_table(merged_by_event):
    """Per-event + OVERALL MAE/Bias/r (+ PBIAS %/NRMSE %) per ogni variabile."""
    groups = ([(SIMULATIONS[k]["label"], [k]) for k in merged_by_event]
              + [("OVERALL", list(merged_by_event))])
    tables = {}
    for obs_col, mod_col, label, unit, circ, pct_ok, pct_off in VAL_VARS:
        rows = []
        for glabel, keys in groups:
            o_all, m_all = [], []
            for key in keys:
                mdf = merged_by_event.get(key)
                if mdf is None or obs_col not in mdf or mod_col not in mdf:
                    continue
                o_all.append(np.asarray(mdf[obs_col], float))
                m_all.append(np.asarray(mdf[mod_col], float))
            if not o_all:
                continue
            o = np.concatenate(o_all); m = np.concatenate(m_all)
            mask = np.isfinite(o) & np.isfinite(m)
            # PBIAS/NRMSE su scala con zero fisico (pct_off): per la temperatura
            # deg C -> K, cosi' la percentuale non e' falsata dallo zero convenzionale.
            rows.append({"Event": glabel, "N": int(mask.sum()),
                         "MAE": mae(o, m, circ), "Bias": bias(o, m, circ),
                         "PBIAS %": pbias(o + pct_off, m + pct_off) if pct_ok else float("nan"),
                         "NRMSE %": nrmse(o + pct_off, m + pct_off) if pct_ok else float("nan"),
                         "r": circ_corr(o, m) if circ else corr(o, m)})
        tables[f"{label} [{unit}]"] = pd.DataFrame(rows).round(
            {"MAE": 2, "Bias": 2, "PBIAS %": 1, "NRMSE %": 1, "r": 3})
    return tables


def plot_timeseries(merged, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Five-row time-series + marginal-PDF figure for one event."""
    sim_key   = merged.attrs["sim_key"]
    sim       = SIMULATIONS[sim_key]
    h_fm_mean = merged.attrs["h_fm_mean"]
    t0, t1    = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])

    ws_label = f"Wind speed at AWS height\n({h_fm_mean:.2f} m AGL, {WS_UNIT})"
    specs = [
        ("psfc_hPa_aws", "psfc_hPa_wrf",  "Surface pressure (hPa)",      "hPa"),
        ("theta_C",      "theta2_C",  "Potential temp. (deg C)",     "deg C"),
        ("WS2_Avg",      "ws_sfc_ms", ws_label,                      "m/s"),
        ("WD2",          "wd10_deg",  "Wind direction (deg)",        "deg"),
        ("rhw_pct",      "rhw2_pct",  "Relative humidity (%)",       "%"),
    ]
    circ = [False, False, False, True, False]
    mod_color = event_color(sim_key)

    fig = plt.figure(figsize=_fs(16, 13))
    gs  = gridspec.GridSpec(5, 2, width_ratios=[3.5, 1], hspace=0.18, wspace=0.04,
                            top=0.90, bottom=0.09, left=0.09, right=0.97)
    ax_ts  = [fig.add_subplot(gs[r, 0]) for r in range(5)]
    ax_pdf = [fig.add_subplot(gs[r, 1]) for r in range(5)]
    for i in range(1, 5):
        ax_ts[i].sharex(ax_ts[0])
    for r in range(5):
        ax_pdf[r].sharey(ax_ts[r])

    for r, ((obs_col, mod_col, ylabel, unit), is_circ) in enumerate(zip(specs, circ)):
        ax = ax_ts[r]
        ov = merged[obs_col].values
        mv = merged[mod_col].values
        if is_circ:
            ov = mask_direction_wraps(ov)
            mv = mask_direction_wraps(mv)
        ax.plot(merged.index, ov, color=COLOR_OBS, lw=1.2)
        ax.plot(merged.index, mv, color=mod_color, lw=1.2)
        ax.set_ylabel(ylabel)
        ax.text(0.02, 0.96, f"MAE = {mae(merged[obs_col], merged[mod_col], is_circ):.2f} {unit}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
        if r == 2:
            ax.set_ylim(bottom=0)
        if r == 3:
            #ax.set_ylim(0, 360); ax.set_yticks([0, 90, 180, 270, 360])
            ax.set_ylim(70, 120) #ax.set_yticks([0, 90, 180, 270, 360])

        if r == 4:
            ax.axhline(100, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.grid(True, color=GRID_COLOR, lw=GRID_LW, ls="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        if r < 4:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        ax_p = ax_pdf[r]
        for col, color in [(obs_col, COLOR_OBS), (mod_col, mod_color)]:
            x, y = kde_curve(merged[col])
            if len(x):
                ax_p.fill_betweenx(x, y, alpha=ALPHA_PDF, color=color)
                ax_p.plot(y, x, color=color, lw=1.2)
        ax_p.grid(True, color=GRID_COLOR, lw=GRID_LW, ls="--")
        ax_p.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
        if r < 4:
            ax_p.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax_p.tick_params(axis="x", labelsize=9)
            ax_p.set_xlabel("Probability density [-]")

    fig.suptitle(f"CRYOWRF vs PEA AWS - {sim['label']} ({domain})\n"
                 f"{t0:%Y-%m-%d %H:%M} - {t1:%Y-%m-%d %H:%M} UTC",
                 fontsize=13, fontweight="bold", y=0.998)
    handles = [Line2D([0], [0], color=COLOR_OBS, lw=1.5, label="AWS"),
               Line2D([0], [0], color=mod_color, lw=1.5, label=f"CRYOWRF ({domain})")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.965), framealpha=0.9, fontsize=12)
    plt.tight_layout()
    savefig(fig, f"timeseries_{domain}_{sim_key}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_scatter_event(merged, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Five-panel obs-vs-model scatter for one event."""
    sim_key   = merged.attrs["sim_key"]
    sim       = SIMULATIONS[sim_key]
    h_fm_mean = merged.attrs["h_fm_mean"]
    specs = [
        ("psfc_hPa_aws", "psfc_hPa_wrf",  "Surface pressure (hPa)",                 "a"),
        ("theta_C",      "theta2_C",  "Potential temperature (deg C)",          "b"),
        ("WS2_Avg",      "ws_sfc_ms", f"Wind speed at {h_fm_mean:.2f} m AGL (m/s)", "c"),
        ("WD2",          "wd10_deg",  "Wind direction (deg)",                   "d"),
        ("rhw_pct",      "rhw2_pct",  "Relative humidity (%)",                  "e"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=_fs(22, 4.5))
    for ax, (obs_col, mod_col, label, letter) in zip(axes, specs):
        x, y = _finite_pair(merged[obs_col], merged[mod_col])
        ax.scatter(x, y, color=event_color(sim_key), s=SC_SIZE, alpha=ALPHA_SC, linewidths=0)
        _add_one_to_one(ax, x, y)
        ax.set_xlabel("Measurements"); ax.set_ylabel("Model")
        ax.set_title(f"({letter}) {label}", fontsize=10, pad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", lw=0.6, zorder=0)
    fig.suptitle(f"CRYOWRF vs PEA AWS - {sim['label']} ({domain})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, f"scatter_{domain}_{sim_key}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_scatter_combined(merged_by_event, domain=DEFAULT_DOMAIN,
                          save=True, output_dir=OUTPUT_DIR,
                          # ── DIMENSIONI / STILE (modificabili facilmente) ──────
                          fig_w=30, fig_h=6.8,          # dimensione figura (pollici)
                          suptitle_fs=22,               # titolo generale
                          title_fs=18,                  # titolo di ogni pannello
                          label_fs=16,                  # etichette assi (Measurements/Model)
                          tick_fs=14,                   # numeri sugli assi
                          legend_fs=16,                 # legenda
                          r_fs=17,                      # scritta "r = ..."
                          point_size=None,              # dim. punti (None -> SC_SIZE)
                          point_alpha=None,             # trasparenza punti (None -> ALPHA_SC)
                          one_to_one_lw=1.5,            # spessore linea 1:1
                          title_pad=8,                  # distanza titolo-pannello
                          circular_wd=True):            # r circolare per la direzione del vento
    """Five-panel scatter with all events overlaid by colour.

    Tutte le *grandezze* sono parametri con valori di default: per cambiarle
    basta passarle dalla cella del notebook (vedi sezione 3), senza toccare
    questo file.

    Ogni "r = ..." annotato nei pannelli e' calcolato con la STESSA logica di
    metrics_table (riga OVERALL della tabella, sezione 4), quindi i valori
    coincidono con la tabella. circular_wd=True usa la correlazione circolare
    per la direzione del vento (come la tabella); False torna al Pearson
    lineare (non appropriato per una variabile angolare).
    """
    s  = SC_SIZE  if point_size  is None else point_size
    al = ALPHA_SC if point_alpha is None else point_alpha
    specs = [
        ("psfc_hPa_aws", "psfc_hPa_wrf",  "Surface pressure (hPa)",        "a", False),
        ("theta_C",      "theta2_C",  "Potential temp. (deg C)",       "b", False),
        ("WS2_Avg",      "ws_sfc_ms", "Wind speed at AWS height (m/s)", "c", False),
        ("WD2",          "wd10_deg",  "Wind direction (deg)",          "d", True),
        ("rhw_pct",      "rhw2_pct",  "Relative humidity (%)",         "e", False),
    ]
    fig, axes = plt.subplots(1, 5, figsize=_fs(fig_w, fig_h))
    for ax, (obs_col, mod_col, label, letter, is_circ) in zip(axes, specs):
        all_x, all_y = [], []          # coppie ripulite -> punti dello scatter
        raw_o, raw_m = [], []          # array grezzi -> r IDENTICO alla tabella sez.4
        for key, mdf in merged_by_event.items():
            if obs_col not in mdf or mod_col not in mdf:
                continue
            x, y = _finite_pair(mdf[obs_col], mdf[mod_col])
            all_x.append(x); all_y.append(y)
            raw_o.append(np.asarray(mdf[obs_col], float))
            raw_m.append(np.asarray(mdf[mod_col], float))
            ax.scatter(x, y, color=SIMULATIONS[key]["color"], s=s,
                       alpha=al, linewidths=0, label=SIMULATIONS[key]["label"])
        if all_x:
            X = np.concatenate(all_x); Y = np.concatenate(all_y)
            # r calcolato ESATTAMENTE come metrics_table (riga OVERALL): array
            # grezzi senza scartare i timestamp duplicati, circ_corr per la
            # direzione del vento e Pearson lineare per le altre variabili.
            O = np.concatenate(raw_o); M = np.concatenate(raw_m)
            r_val = (circ_corr(O, M) if (is_circ and circular_wd) else corr(O, M))
            _add_one_to_one(ax, X, Y, fs=r_fs, r_value=r_val, lw=one_to_one_lw)
        ax.set_xlabel("Measurements", fontsize=label_fs)
        ax.set_ylabel("Model", fontsize=label_fs)
        ax.set_title(f"({letter}) {label}", fontsize=title_fs, pad=title_pad)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", lw=0.6, zorder=0)
        ax.tick_params(labelsize=tick_fs)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=legend_fs, framealpha=0.9)
    fig.suptitle("CRYOWRF vs PEA AWS - all events",
                 fontsize=suptitle_fs, fontweight="bold", y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, f"scatter_combined_all_events_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def _add_one_to_one(ax, x, y, fs=10, r_value=None, lw=1.5):
    """Draw a 1:1 line, square the axes, and annotate r.

    If *r_value* is given it is used verbatim (so the annotation can match the
    statistics table, e.g. a circular correlation for wind direction).
    Otherwise a plain linear Pearson r is computed (backward compatible).
    """
    if not len(x):
        return
    vmin = min(x.min(), y.min()); vmax = max(x.max(), y.max())
    pad = (vmax - vmin) * 0.05 or 1.0
    lims = (vmin - pad, vmax + pad)
    ax.plot(lims, lims, "k-", lw=lw, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", adjustable="box")
    r = r_value if r_value is not None else (pearsonr(x, y)[0] if len(x) >= 3 else np.nan)
    if np.isfinite(r):
        ax.text(0.05, 0.93, f"r = {r:.2f}",
                transform=ax.transAxes, fontsize=fs, fontweight="bold", va="top")


# ═════════════════════════════════════════════════════════════════════════════
# 2) Drift-flux profiles  (drift_flux_profiles.ipynb)
# ═════════════════════════════════════════════════════════════════════════════

def _read_snow_drift_ust(fpath, tidx):
    with _NC(fpath) as nc:
        la, lo = fiv._get_latlon(nc)
        sd  = fiv._interp_pt(fiv._build_interp(la, lo, np.array(nc.variables["SNOW_DRIFT"][tidx])),
                             fiv.PEA_LAT, fiv.PEA_LON)
        ust = fiv._interp_pt(fiv._build_interp(la, lo, np.array(nc.variables["UST"][tidx])),
                             fiv.PEA_LAT, fiv.PEA_LON)
    return float(sd), float(ust)


def build_regime(sim_key, domain=DEFAULT_DOMAIN, n_timesteps=5, timestep_mode="even"):
    """Profiles at a handful of timesteps, with SN_C_HEIGHT read per timestep."""
    sim   = SIMULATIONS[sim_key]
    h_cfg = INSTRUMENT_HEIGHTS[sim_key]
    spc_z, fc_lo, fc_hi = h_cfg["spc_z"], h_cfg["fc_z_lo"], h_cfg["fc_z_hi"]
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])

    bs_times, drift = fiv.interp_bs_drift_profile(files)
    idx = pd.DatetimeIndex(bs_times).tz_localize(None)

    with _NC(files[0]) as nc0:
        la, lo = fiv._get_latlon(nc0)
        h_ref = read_sn_c_height_at_t(nc0, la, lo, 0)

    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    in_win = (idx >= t0) & (idx <= t1)
    Q01    = fiv.integrate_drift(drift, h_ref, z_lo=fc_lo, z_hi=fc_hi)

    win_idx = np.where(in_win)[0]
    n = min(n_timesteps, len(win_idx))
    if timestep_mode == "peak":
        sel_idx = sorted(win_idx[np.argsort(Q01[win_idx])[::-1][:n]].tolist())
    else:
        pick = np.linspace(0, len(win_idx) - 1, n).round().astype(int)
        sel_idx = win_idx[pick].tolist()

    recs    = enumerate_times(files)
    fc_raw  = fiv.load_flowcapt(sim["aws_file"])
    spc_raw = load_spc_events(sim["spc_file"])

    steps = []
    for si in sel_idx:
        ts = idx[si]
        rec = min(recs, key=lambda r: abs(r[0] - ts))
        with _NC(rec[1]) as nc_t:
            la, lo = fiv._get_latlon(nc_t)
            h_t = read_sn_c_height_at_t(nc_t, la, lo, rec[2])

        fc_int = obs_at_time(fc_raw,  FC_FLUX_COL,  ts)
        spc_q  = obs_at_time(spc_raw, SPC_FLUX_COL, ts)
        try:
            sd_v, ust_v = _read_snow_drift_ust(rec[1], rec[2])
            q_s = float(salt_flux_vec(sd_v, ust_v)) * 1e3
        except Exception:
            q_s = np.nan

        h_aug = np.concatenate([[H_SALT], h_t])
        p_aug = np.concatenate([[q_s / 1e3 if np.isfinite(q_s) else 0.0], drift[si]])
        q_spc_salt = float(np.interp(spc_z, h_aug, p_aug)) * 1e3

        steps.append({
            "ts": ts, "profile": drift[si] * 1e3, "heights": h_t,
            "spc_q": spc_q * 1e3 if np.isfinite(spc_q) else np.nan,
            "fc_int": fc_int,
            "fc_mean": fc_int / (fc_hi - fc_lo) if np.isfinite(fc_int) else np.nan,
            "q_salt": q_s, "q_spc_salt": q_spc_salt,
        })

    return {"label": sim["label"], "color": sim["color"], "steps": steps,
            "spc_z": spc_z, "fc_lo": fc_lo, "fc_hi": fc_hi}


def plot_profiles(regimes, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Grid of per-timestep flux profiles (rows = events, cols = timesteps)."""
    keys = list(regimes.keys())
    n_ts = max(len(regimes[k]["steps"]) for k in keys)
    fig, axes = plt.subplots(len(keys), n_ts, figsize=_fs(3.6 * n_ts, 5.2 * len(keys)),
                             sharey="row", squeeze=False)
    for i, key in enumerate(keys):
        r = regimes[key]
        for j in range(n_ts):
            ax = axes[i][j]
            if j >= len(r["steps"]):
                ax.axis("off"); continue
            s = r["steps"][j]; h = s["heights"]
            ax.plot(s["profile"], h, color=r["color"], lw=2.0, marker="o", ms=5,
                    label="CRYOWRF mesh", zorder=5)
            if np.isfinite(s["q_salt"]):
                ax.plot([s["q_salt"], s["profile"][0]], [H_SALT, h[0]],
                        color=r["color"], lw=1.6, ls="--", zorder=4)
                ax.plot(s["q_salt"], H_SALT, marker="D", ms=8, color=r["color"],
                        markeredgecolor="black", markeredgewidth=0.5, zorder=6,
                        label=f"Salt BC @ {H_SALT:.2f} m")
            if np.isfinite(s["q_spc_salt"]):
                ax.plot(s["q_spc_salt"], r["spc_z"], marker="^", ms=11, color=r["color"],
                        markeredgecolor="black", markeredgewidth=0.5, zorder=8,
                        label=f"WRF @ {r['spc_z']:.2f} m")
            if np.isfinite(s["spc_q"]):
                ax.plot(s["spc_q"], r["spc_z"], marker="*", ms=15, color=COLOR_OBS,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=9,
                        label=f"SPC obs @ {r['spc_z']:.2f} m")
            if np.isfinite(s["fc_mean"]):
                ax.axhspan(r["fc_lo"], r["fc_hi"], color=GRID_COLOR, alpha=0.25, zorder=1)
                ax.vlines(s["fc_mean"], r["fc_lo"], r["fc_hi"], color=COLOR_OBS, lw=2.0,
                          zorder=4, label=f"FC obs {r['fc_lo']:.2f}-{r['fc_hi']:.2f} m")
            for hh in h:
                ax.axhline(hh, color=GRID_COLOR, lw=0.5, zorder=0)
            ax.axhline(H_SALT, color=GRID_COLOR, lw=0.5, ls="--", zorder=0)
            ax.set_yscale("log")
            ax.set_ylim(H_SALT * 0.75, h[-1] * 1.4)
            ax.set_title(f"{s['ts']:%d %b %H:%M}", fontsize=TICK_FS)
            ax.tick_params(labelsize=TICK_FS - 1)
            ax.grid(True, which="both", color=GRID_COLOR, lw=GRID_LW, alpha=0.4)
            if j == 0:
                ax.set_ylabel(f"{r['label']}\n\nHeight AGL [m]", fontsize=TICK_FS)
            if i == len(keys) - 1:
                ax.set_xlabel(f"Drift flux [{FLUX_UNIT}]", fontsize=TICK_FS)
            if i == 0 and j == n_ts - 1:
                ax.legend(fontsize=TICK_FS - 3, loc="upper right", framealpha=0.92)
    fig.suptitle("CRYOWRF - drifting-snow mass-flux profiles with saltation BC",
                 fontsize=LABEL_FS + 2, fontweight="bold", y=1.005)
    plt.tight_layout()
    savefig(fig, f"drift_flux_profiles_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def build_regime_full(sim_key, domain=DEFAULT_DOMAIN):
    """Event-level statistics: every active timestep, SN_C_HEIGHT per step."""
    sim   = SIMULATIONS[sim_key]
    h_cfg = INSTRUMENT_HEIGHTS[sim_key]
    spc_z, fc_lo, fc_hi = h_cfg["spc_z"], h_cfg["fc_z_lo"], h_cfg["fc_z_hi"]
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])

    bs_times, drift = fiv.interp_bs_drift_profile(files)
    idx = pd.DatetimeIndex(bs_times).tz_localize(None)
    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    win = (idx >= t0) & (idx <= t1)
    times = idx[win]
    prof  = drift[win] * 1e3

    recs_dict = {r[0]: (r[1], r[2]) for r in enumerate_times(files)}
    heights_all = []
    for ts in times:
        (fpath, tidx) = min(recs_dict.items(), key=lambda x: abs(x[0] - ts))[1]
        with _NC(fpath) as nc_t:
            la, lo = fiv._get_latlon(nc_t)
            heights_all.append(read_sn_c_height_at_t(nc_t, la, lo, tidx))
    heights_all = np.array(heights_all)

    _, sd_full  = fro.interp_surface(files, "SNOW_DRIFT")
    _, ust_full = fro.interp_surface(files, "UST")
    q_salt = salt_flux_vec(np.array(sd_full)[win], np.array(ust_full)[win]) * 1e3

    fc_raw  = fiv.load_flowcapt(sim["aws_file"])
    spc_raw = load_spc_events(sim["spc_file"])
    spc_s = np.array([obs_at_time(spc_raw, SPC_FLUX_COL, t) * 1e3 for t in times])
    fc_s  = np.array([obs_at_time(fc_raw,  FC_FLUX_COL,  t)       for t in times])

    Q01 = np.array([
        fiv.integrate_drift(prof[i:i + 1], heights_all[i], z_lo=fc_lo, z_hi=fc_hi)[0]
        for i in range(len(times))
    ])
    active = Q01 > 0.0

    return {
        "label": sim["label"], "color": sim["color"],
        "prof": prof[active], "h_mat": heights_all[active],
        "q_salt": q_salt[active],
        "spc_obs": spc_s[active][np.isfinite(spc_s[active])],
        "fc_obs": (fc_s[active] / (fc_hi - fc_lo))[np.isfinite(fc_s[active])],
        "spc_z": spc_z, "fc_lo": fc_lo, "fc_hi": fc_hi,
    }


def plot_mean_profile(regimes_full, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR,
                      band_outer=(10, 90), band_inner=(25, 75)):
    """Event-mean flux profile with inter-quantile shading per event."""
    keys = list(regimes_full.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=_fs(5 * len(keys), 6.5),
                             sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        r = regimes_full[key]
        if not r["prof"].shape[0]:
            ax.text(0.5, 0.5, "no active steps", transform=ax.transAxes,
                    ha="center", va="center", color="gray"); continue
        h_mean = np.nanmean(r["h_mat"], axis=0)
        mean   = np.nanmean(r["prof"], axis=0)
        lo_o, hi_o = np.nanpercentile(r["prof"], band_outer, axis=0)
        lo_i, hi_i = np.nanpercentile(r["prof"], band_inner, axis=0)
        ax.fill_betweenx(h_mean, lo_o, hi_o, color=r["color"], alpha=0.15)
        ax.fill_betweenx(h_mean, lo_i, hi_i, color=r["color"], alpha=0.30)
        ax.plot(mean, h_mean, color=r["color"], lw=2.2, marker="o", ms=5,
                label="Event mean", zorder=5)
        qs = r["q_salt"]
        if len(qs) and np.any(np.isfinite(qs)):
            qs_mean = np.nanmean(qs)
            ax.plot([qs_mean, mean[0]], [H_SALT, h_mean[0]],
                    color=r["color"], lw=2.2, ls="--", zorder=5)
            ax.plot(qs_mean, H_SALT, marker="D", ms=8, color=r["color"],
                    markeredgecolor="black", markeredgewidth=0.5, zorder=6,
                    label=f"Salt BC @ {H_SALT:.2f} m")
        if len(r["spc_obs"]):
            sv = r["spc_obs"]
            lo_err = max(np.mean(sv) - np.percentile(sv, 25), 0.0)
            hi_err = max(np.percentile(sv, 75) - np.mean(sv), 0.0)
            ax.errorbar(np.mean(sv), r["spc_z"],
                        xerr=[[lo_err], [hi_err]],
                        fmt="*", ms=14, color='green', markeredgecolor="black",
                        markeredgewidth=0.5, zorder=9, label=f"SPC @ {r['spc_z']:.2f} m")
        if len(r["fc_obs"]):
            ax.vlines(np.mean(r["fc_obs"]), r["fc_lo"], r["fc_hi"], color='blueviolet',
                      lw=2.2, zorder=4, label=f"FC {r['fc_lo']:.2f}-{r['fc_hi']:.2f} m")
        ax.set_yscale("log")
        ax.set_ylim(H_SALT * 0.75, h_mean[-1] * 1.4)
        ax.set_xlim(left=0)
        ax.set_xlabel(f"Mass flux [{FLUX_UNIT}]", fontsize=LABEL_FS)
        ax.set_title(r["label"], fontsize=TICK_FS + 1)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, which="both", color=GRID_COLOR, lw=GRID_LW, alpha=0.4)
        ax.legend(fontsize=TICK_FS - 2, loc="upper right", framealpha=0.92)
    axes[0][0].set_ylabel("Height above snow surface [m]", fontsize=LABEL_FS)
    fig.suptitle("Event-mean mass-flux profile", fontsize=LABEL_FS + 1,
                 fontweight="bold", y=1.005)
    plt.tight_layout()
    savefig(fig, f"drift_flux_mean_profile_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 3) Blowing-snow validation  (blowing_snow_validation.ipynb)
# ═════════════════════════════════════════════════════════════════════════════

def build_bs_event(sim_key, domain=DEFAULT_DOMAIN):
    """Aligned FC / SPC fluxes (original + saltation BC) for one event."""
    sim   = SIMULATIONS[sim_key]
    h_cfg = INSTRUMENT_HEIGHTS[sim_key]
    spc_z, fc_lo, fc_hi = h_cfg["spc_z"], h_cfg["fc_z_lo"], h_cfg["fc_z_hi"]
    fc_r  = fc_hi - fc_lo
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])

    bs_times, drift = fiv.interp_bs_drift_profile(files)
    _, sd_a  = fro.interp_surface(files, "SNOW_DRIFT")
    _, ust_a = fro.interp_surface(files, "UST")
    q_s = salt_flux_vec(np.array(sd_a), np.array(ust_a))

    with _NC(files[0]) as nc0:
        la, lo = fiv._get_latlon(nc0)
        h_ref = read_sn_c_height_at_t(nc0, la, lo, 0)
    h_aug     = np.concatenate([[H_SALT], h_ref])
    drift_aug = np.concatenate([q_s[:, None], drift], axis=1)

    idx = pd.DatetimeIndex(bs_times).tz_localize(None)
    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    wrf_cols = {
        "Q_fc_salt_wrf":  fiv.integrate_drift(drift_aug, h_aug, fc_lo, fc_hi) / fc_r,
        "q_spc_salt_wrf": fiv.extract_at_height(drift_aug, h_aug, spc_z),
    }

    # Model mean particle radius [µm] at the SPC height, from SFC_MEANR (same
    # 8 drift levels as BS_QI_DRIFT, stored in metres). Optional: only if the
    # variable is present in the output.
    if any("SFC_MEANR" in _NC(f).variables for f in files[:1]):
        _, meanr_prof = fiv.interp_sfc_meanr_profile(files)
        wrf_cols["meanr_wrf_um"] = fiv.extract_at_height(meanr_prof, h_ref, spc_z) * 1e6
    else:
        wrf_cols["meanr_wrf_um"] = np.nan

    wrf = pd.DataFrame(wrf_cols, index=idx).sort_index().loc[t0:t1]

    fc_raw  = fiv.load_flowcapt(sim["aws_file"])
    spc_raw = load_spc_events(sim["spc_file"])
    spc_raw["mean_r_um"] = fiv.spc_mean_radius(spc_raw, SPC_BIN_DIAMETERS)
    fc_h  = fiv.align_obs(fc_raw,  wrf, window_min=ALIGN_WINDOW_MIN)
    spc_h = fiv.align_obs(spc_raw, wrf, window_min=ALIGN_WINDOW_MIN)

    obs_spc   = spc_h[SPC_FLUX_COL]   * 1e3
    mod_spc   = wrf["q_spc_salt_wrf"] * 1e3
    obs_meanr = spc_h.get("mean_r_um", pd.Series(np.nan, index=spc_h.index))
    mod_meanr = wrf["meanr_wrf_um"]
    # The mean radius is meaningful only when snow is actually in transport, so
    # it is kept only where BOTH the mass flux at the SPC height is > 0 AND the
    # reported radius is > 0 (the model sets the radius to 0 to flag "no
    # particles", which can occur even at a small positive flux; without the
    # radius > 0 condition those spurious zeros would remain).
    obs_meanr = obs_meanr.where((obs_spc > 0) & (obs_meanr > 0))
    mod_meanr = mod_meanr.where((mod_spc > 0) & (mod_meanr > 0))

    return {
        "sim_key": sim_key, "label": sim["label"],
        "spc_z": spc_z, "fc_lo": fc_lo, "fc_hi": fc_hi,
        "obs_fc":       fc_h["PF_FC4"],
        "mod_fc_salt":  wrf["Q_fc_salt_wrf"] * 1e3,
        "obs_spc":      obs_spc,
        "mod_spc_salt": mod_spc,
        "obs_meanr":    obs_meanr,
        "mod_meanr":    mod_meanr,
    }


def print_bs_metrics(ev):
    """Console MAE/Bias/r table for one blowing-snow event dict."""
    print(f"  {'Quantity':36s}  {'N':>4s}  {'MAE':>8s}  {'Bias':>8s}  {'r':>6s}")
    rows = [
        ("FC",                           ev["obs_fc"],  ev["mod_fc_salt"]),
        (f"SPC (@ {ev['spc_z']:.2f} m)", ev["obs_spc"], ev["mod_spc_salt"]),
    ]
    for lbl, o_s, m_s in rows:
        a, b = _finite_pair(o_s, m_s)
        n = len(a)
        mae_v  = float(np.nanmean(np.abs(a - b))) if n else np.nan
        bias_v = float(np.nanmean(b - a)) if n else np.nan
        r_v    = pearsonr(a, b)[0] if n >= 3 else np.nan
        print(f"  {lbl:36s}  {n:4d}  {mae_v:8.3f}  {bias_v:+8.3f}  {r_v:6.3f}")


def plot_bs_timeseries(ev, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Time series + PDF for FC, SPC, and mean radius (one event)."""
    pairs = [
        (ev["obs_fc"],  ev["mod_fc_salt"],
         f"FC flux @ {ev['fc_lo']:.2f}-{ev['fc_hi']:.2f} m \n [{FLUX_UNIT}]"),
        (ev["obs_spc"], ev["mod_spc_salt"],
         f"SPC flux @ {ev['spc_z']:.2f} m \n [{FLUX_UNIT}]"),
        (ev["obs_meanr"], ev["mod_meanr"],
         "Mean particle \n radius [μm]"),
    ]
    fig = plt.figure(figsize=_fs(16, 10))
    gs  = gridspec.GridSpec(3, 2, width_ratios=[3.5, 1], hspace=0.18, wspace=0.04,
                            top=0.90, bottom=0.09, left=0.09, right=0.97)
    ax_ts  = [fig.add_subplot(gs[r, 0]) for r in range(3)]
    ax_pdf = [fig.add_subplot(gs[r, 1]) for r in range(3)]
    for i in range(1, 3):
        ax_ts[i].sharex(ax_ts[0])
    for r in range(3):
        ax_pdf[r].sharey(ax_ts[r])
    mod_color = event_color(ev["sim_key"])
    for r, (obs_s, mod_s, ylabel) in enumerate(pairs):
        ax = ax_ts[r]
        ax.plot(obs_s.index, obs_s.values, color=COLOR_OBS, lw=1.2, label="Obs")
        ax.plot(mod_s.index, mod_s.values, color=mod_color, lw=1.2, label="CRYOWRF")
        ax.set_ylabel(ylabel, fontsize=TICK_FS); ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID_COLOR, lw=GRID_LW, ls="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        if r < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax_p = ax_pdf[r]
        for data, color in [(obs_s, COLOR_OBS), (mod_s, mod_color)]:
            x, y = kde_curve(data)
            if len(x):
                ax_p.fill_betweenx(x, y, alpha=ALPHA_PDF, color=color)
                ax_p.plot(y, x, color=color, lw=1.2)
        ax_p.grid(True, color=GRID_COLOR, lw=GRID_LW, ls="--")
        ax_p.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
        if r < 2:
            ax_p.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax_p.tick_params(axis="x", labelsize=9)
            ax_p.set_xlabel("Probability density [-]")
    ax_ts[0].legend(fontsize=TICK_FS, loc="upper right")
    fig.suptitle(f"Mass fluxes: CRYOWRF vs observations - {ev['label']}",
                 fontsize=13, fontweight="bold", y=0.998)
    plt.tight_layout()
    savefig(fig, f"bs_timeseries_{domain}_{ev['sim_key']}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_bs_scatter(ev, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Two-panel FC / SPC obs-vs-model scatter (one event)."""
    fig, axes = plt.subplots(1, 3, figsize=_fs(15, 5))
    specs = [
        (ev["obs_fc"],  ev["mod_fc_salt"],
         f"FC {ev['fc_lo']:.2f}-{ev['fc_hi']:.2f} m [{FLUX_UNIT}]", "a"),
        (ev["obs_spc"], ev["mod_spc_salt"],
         f"SPC @ {ev['spc_z']:.2f} m [{FLUX_UNIT}]", "b"),
        (ev["obs_meanr"], ev["mod_meanr"],
         "Mean particle radius [μm]", "c"),
    ]
    for ax, (obs_s, mod_s, label, letter) in zip(axes, specs):
        x, y = _finite_pair(obs_s, mod_s)
        ax.scatter(x, y, color=event_color(ev["sim_key"]), s=SC_SIZE, alpha=ALPHA_SC, linewidths=0)
        _add_one_to_one(ax, x, y, fs=11)
        ax.set_xlabel("Observed"); ax.set_ylabel("Modelled")
        ax.set_title(f"({letter}) {label}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", lw=0.6, zorder=0)
    fig.suptitle(f"CRYOWRF vs PEA - {ev['label']}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, f"bs_scatter_{domain}_{ev['sim_key']}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def bs_combined_pairs(events):
    """Reshape a dict of event dicts into {quantity: {event: (obs, mod)}}."""
    out = {}
    for key, ev in events.items():
        out.setdefault(f"FC {ev['fc_lo']:.2f}-{ev['fc_hi']:.2f} m [{FLUX_UNIT}]", {})[key] = \
            (ev["obs_fc"], ev["mod_fc_salt"])
        out.setdefault(f"SPC @ {ev['spc_z']:.2f} m [{FLUX_UNIT}]", {})[key] = \
            (ev["obs_spc"], ev["mod_spc_salt"])
    return out


def plot_bs_scatter_combined(events, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """All-event overlaid scatter for the blowing-snow fluxes."""
    pairs = bs_combined_pairs(events)
    qtys = list(pairs.keys())
    fig, axes = plt.subplots(1, len(qtys), figsize=_fs(5.6 * len(qtys), 4.5), squeeze=False)
    for ax, qty in zip(axes[0], qtys):
        all_x, all_y = [], []
        for key, (o_s, m_s) in pairs[qty].items():
            x, y = _finite_pair(o_s, m_s)
            all_x.append(x); all_y.append(y)
            ax.scatter(x, y, color=SIMULATIONS[key]["color"], s=SC_SIZE,
                       alpha=ALPHA_SC, linewidths=0, label=SIMULATIONS[key]["label"])
        if all_x:
            _add_one_to_one(ax, np.concatenate(all_x), np.concatenate(all_y), fs=13)
        ax.set_xlabel("Observed"); ax.set_ylabel("Modelled")
        ax.set_title(qty, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#dddddd", lw=0.6, zorder=0)
        ax.tick_params(labelsize=TICK_FS)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), fontsize=13, framealpha=0.9)
    fig.suptitle("CRYOWRF vs PEA - blowing snow, all events",
                 fontsize=16, fontweight="bold", y=1.06)
    plt.tight_layout()
    savefig(fig, f"bs_scatter_combined_all_events_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_bs_flux_scatter_all(events, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Single scatter pooling every mass-flux value (SPC + FC, both heights)
    across all events. Colour = event, marker = instrument."""
    fig, ax = plt.subplots(figsize=_fs(6.5, 6.5))
    all_x, all_y = [], []
    for key, ev in events.items():
        col = SIMULATIONS[key]["color"]
        for obs_s, mod_s, marker in [
            (ev["obs_fc"],  ev["mod_fc_salt"],  "d"),   # FlowCapt
            (ev["obs_spc"], ev["mod_spc_salt"], "o"),   # SPC
        ]:
            x, y = _finite_pair(obs_s, mod_s)
            if not len(x):
                continue
            all_x.append(x); all_y.append(y)
            ax.scatter(x, y, color=col, marker=marker, s=SC_SIZE,
                       alpha=ALPHA_SC, linewidths=0)
    if all_x:
        _add_one_to_one(ax, np.concatenate(all_x), np.concatenate(all_y), fs=13)
    ax.set_xlabel(f"Observed mass flux [{FLUX_UNIT}]", fontsize=LABEL_FS)
    ax.set_ylabel(f"Modelled mass flux [{FLUX_UNIT}]", fontsize=LABEL_FS)
    ax.set_title("Blowing-snow mass flux - all events, all instruments",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#dddddd", lw=0.6, zorder=0)
    ax.tick_params(labelsize=TICK_FS)

    event_handles = [Line2D([0], [0], marker="s", color="w",
                            markerfacecolor=SIMULATIONS[k]["color"], markersize=9,
                            label=SIMULATIONS[k]["label"]) for k in events]
    instr_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=9, label="SPC"),
        Line2D([0], [0], marker="d", color="w", markerfacecolor="gray", markersize=9, label="FlowCapt"),
    ]
    ax.legend(handles=event_handles + instr_handles, fontsize=TICK_FS - 1,
              loc="upper left", framealpha=0.9)
    plt.tight_layout()
    savefig(fig, f"bs_flux_scatter_all_events_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_bs_meanr_scatter_all(events, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Single mean-radius scatter pooling the events (SPC was at different
    heights per event, but values are combined here). Colour = event."""
    fig, ax = plt.subplots(figsize=_fs(6.5, 6.5))
    all_x, all_y = [], []
    for key, ev in events.items():
        x, y = _finite_pair(ev["obs_meanr"], ev["mod_meanr"])
        if not len(x):
            continue
        all_x.append(x); all_y.append(y)
        ax.scatter(x, y, color=SIMULATIONS[key]["color"], s=SC_SIZE,
                   alpha=ALPHA_SC, linewidths=0, label=SIMULATIONS[key]["label"])
    if all_x:
        _add_one_to_one(ax, np.concatenate(all_x), np.concatenate(all_y), fs=13)
    ax.set_xlabel("Observed mean radius [μm]", fontsize=LABEL_FS)
    ax.set_ylabel("Modelled mean radius [μm]", fontsize=LABEL_FS)
    ax.set_title("Blowing-snow mean particle radius - all events",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#dddddd", lw=0.6, zorder=0)
    ax.tick_params(labelsize=TICK_FS)
    if all_x:
        ax.legend(fontsize=TICK_FS - 1, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    savefig(fig, f"bs_meanr_scatter_all_events_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def bs_stats_tables(events):
    """Per-event + OVERALL MAE/Bias/r for the combined blowing-snow fluxes."""
    pairs = bs_combined_pairs(events)
    groups = ([(SIMULATIONS[k]["label"], [k]) for k in events]
              + [("OVERALL", list(events))])
    tables = {}
    for qty, per_event in pairs.items():
        rows = []
        for glabel, keys in groups:
            o_all, m_all = [], []
            for key in keys:
                if key not in per_event:
                    continue
                x, y = _finite_pair(*per_event[key])
                o_all.append(x); m_all.append(y)
            if not o_all:
                continue
            o = np.concatenate(o_all); m = np.concatenate(m_all)
            rows.append({"Event": glabel, "N": len(o),
                         "MAE": float(np.nanmean(np.abs(o - m))),
                         "Bias": float(np.nanmean(m - o)),
                         "PBIAS %": pbias(o, m),
                         "NRMSE %": nrmse(o, m),
                         "r": pearsonr(o, m)[0] if len(o) >= 3 else float("nan")})
        tables[qty] = pd.DataFrame(rows).round(
            {"MAE": 3, "Bias": 3, "PBIAS %": 1, "NRMSE %": 1, "r": 3})
    return tables


def bs_meanr_stats(events):
    """Per-event + OVERALL N/MAE/Bias/r for the blowing-snow mean particle
    radius (SPC vs SFC_MEANR), using the flux>0 & radius>0 masked pairs."""
    groups = ([(SIMULATIONS[k]["label"], [k]) for k in events]
              + [("OVERALL", list(events))])
    rows = []
    for glabel, keys in groups:
        o_all, m_all = [], []
        for key in keys:
            ev = events.get(key)
            if ev is None:
                continue
            x, y = _finite_pair(ev["obs_meanr"], ev["mod_meanr"])
            o_all.append(x); m_all.append(y)
        if not o_all:
            continue
        o = np.concatenate(o_all); m = np.concatenate(m_all)
        if not len(o):
            continue
        rows.append({"Event": glabel, "N": len(o),
                     "MAE": float(np.nanmean(np.abs(m - o))),
                     "Bias": float(np.nanmean(m - o)),
                     "r": pearsonr(o, m)[0] if len(o) >= 3 else float("nan")})
    return {"Mean particle radius [µm]":
            pd.DataFrame(rows).round({"MAE": 1, "Bias": 1, "r": 3})}


# ═════════════════════════════════════════════════════════════════════════════
#  Time-lag / cross-correlation analysis of the blowing-snow fluxes
#
#  The modelled peaks appear to arrive slightly AFTER the measured ones. Obs and
#  model live on the same regular WRF output grid (obs are aligned onto it in
#  ``build_bs_event``), so a lag of *k* steps is applied by shifting the model
#  column by -k rows. Sign convention: a POSITIVE lag means the model has to be
#  looked at *k* steps LATER to match the current observation, i.e. the modelled
#  peak arrives ``k * dt`` after the measured one (the model is late).
# ═════════════════════════════════════════════════════════════════════════════

# Flux quantities each event carries, as (obs-key, model-key, short label).
# The saltation-BC model is used to match the time-series / scatter figures.
BS_LAG_QUANTITIES = [
    ("obs_fc",  "mod_fc_salt",  "FC"),
    ("obs_spc", "mod_spc_salt", "SPC"),
]


def _series_timestep_min(idx):
    """Median spacing of a DatetimeIndex, in minutes (nan if undefined)."""
    d = pd.Series(pd.DatetimeIndex(idx).sort_values()).diff().dropna()
    if not len(d):
        return np.nan
    return d.median().total_seconds() / 60.0


def bs_lag_pair(obs_s, mod_s, step):
    """Finite (obs, model) arrays with the model shifted by *step* grid steps.

    ``step`` > 0 pairs obs(t) with model(t + step*dt); ``step`` < 0 pairs
    obs(t) with an earlier model value.
    """
    df = pd.concat([obs_s.rename("obs"), mod_s.rename("mod")], axis=1)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    shifted = df["mod"].shift(-int(step))
    pair = pd.concat([df["obs"].rename("o"), shifted.rename("m")], axis=1).dropna()
    o = pair["o"].to_numpy(float)
    m = pair["m"].to_numpy(float)
    mask = np.isfinite(o) & np.isfinite(m)
    return o[mask], m[mask]


def bs_lag_correlation(obs_s, mod_s, max_steps=6):
    """Pearson r between obs(t) and model(t + lag) over integer step lags.

    Returns a dict with arrays: ``steps`` (grid steps), ``minutes`` (lag in
    minutes), ``r`` (Pearson r, nan where n < 3), ``n`` (sample size), and the
    scalar ``dt_min`` (grid spacing in minutes).
    """
    dt_min = _series_timestep_min(
        obs_s.index.union(mod_s.index) if len(mod_s) else obs_s.index)
    steps = np.arange(-int(max_steps), int(max_steps) + 1)
    r_vals, n_vals = [], []
    for k in steps:
        o, m = bs_lag_pair(obs_s, mod_s, k)
        n_vals.append(len(o))
        r_vals.append(pearsonr(o, m)[0] if len(o) >= 3 else np.nan)
    return {"steps": steps, "minutes": steps * dt_min,
            "r": np.array(r_vals), "n": np.array(n_vals), "dt_min": dt_min}


def bs_best_lag(obs_s, mod_s, max_steps=6):
    """Lag (dict: step, minutes, r, n, dt_min) that maximises Pearson r."""
    c = bs_lag_correlation(obs_s, mod_s, max_steps)
    r = c["r"]
    if np.all(np.isnan(r)):
        return {"step": 0, "minutes": 0.0, "r": np.nan, "n": 0,
                "dt_min": c["dt_min"], "r0": np.nan}
    i = int(np.nanargmax(r))
    i0 = int(np.where(c["steps"] == 0)[0][0])
    return {"step": int(c["steps"][i]), "minutes": float(c["minutes"][i]),
            "r": float(r[i]), "n": int(c["n"][i]), "dt_min": c["dt_min"],
            "r0": float(r[i0])}


def bs_lag_tables(events, max_steps=6):
    """Per-event best-lag summary for FC and SPC, as {label: DataFrame}."""
    tables = {}
    for _, mod_key, qlabel in BS_LAG_QUANTITIES:
        obs_key = "obs_fc" if qlabel == "FC" else "obs_spc"
        rows = []
        for key, ev in events.items():
            b = bs_best_lag(ev[obs_key], ev[mod_key], max_steps)
            rows.append({"Event": SIMULATIONS[key]["label"],
                         "dt [min]": round(b["dt_min"], 1),
                         "r (lag 0)": round(b["r0"], 3),
                         "best lag [min]": round(b["minutes"], 1),
                         "best lag [steps]": b["step"],
                         "r (best)": round(b["r"], 3),
                         "N": b["n"]})
        tables[f"{qlabel} flux - best time lag"] = pd.DataFrame(rows)
    return tables


def plot_bs_lag_curves(events, domain=DEFAULT_DOMAIN, max_steps=6,
                       save=True, output_dir=OUTPUT_DIR):
    """Pearson r as a function of applied time lag, one line per event.

    One panel per flux quantity (FC, SPC). A filled marker flags the lag that
    maximises r for each event. Positive lag = modelled peak arrives late.
    """
    fig, axes = plt.subplots(1, len(BS_LAG_QUANTITIES),
                             figsize=_fs(6.4 * len(BS_LAG_QUANTITIES), 5.2),
                             squeeze=False)
    for ax, (_, mod_key, qlabel) in zip(axes[0], BS_LAG_QUANTITIES):
        obs_key = "obs_fc" if qlabel == "FC" else "obs_spc"
        for key, ev in events.items():
            c = bs_lag_correlation(ev[obs_key], ev[mod_key], max_steps)
            col = SIMULATIONS[key]["color"]
            ax.plot(c["minutes"], c["r"], color=col, lw=1.4, marker="o",
                    ms=4, label=SIMULATIONS[key]["label"])
            if not np.all(np.isnan(c["r"])):
                i = int(np.nanargmax(c["r"]))
                ax.plot(c["minutes"][i], c["r"][i], color=col, marker="o",
                        ms=11, mfc=col, mec="k", mew=1.2, zorder=5)
        ax.axvline(0, color="#888888", lw=1.0, ls="--")
        ax.axhline(0, color="#dddddd", lw=0.8)
        ax.set_xlabel("Applied time lag [min]  (+ = model late)", fontsize=TICK_FS)
        ax.set_ylabel("Pearson r", fontsize=TICK_FS)
        ax.set_title(f"{qlabel} flux", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID_COLOR, lw=GRID_LW, ls="--", zorder=0)
        ax.tick_params(labelsize=TICK_FS)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(events),
               bbox_to_anchor=(0.5, 1.04), fontsize=TICK_FS, framealpha=0.9)
    fig.suptitle("Blowing-snow flux: lagged cross-correlation",
                 fontsize=14, fontweight="bold", y=1.10)
    plt.tight_layout()
    savefig(fig, f"bs_lag_correlation_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_bs_lag_scatter(events, domain=DEFAULT_DOMAIN, max_steps=6,
                        save=True, output_dir=OUTPUT_DIR):
    """Obs-vs-model scatter at each event's best lag (rows = FC/SPC quantity,
    columns = events). Grey points show the zero-lag pairing for reference."""
    keys = list(events.keys())
    nrow, ncol = len(BS_LAG_QUANTITIES), len(keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=_fs(4.6 * ncol, 4.6 * nrow),
                             squeeze=False)
    for r, (_, mod_key, qlabel) in enumerate(BS_LAG_QUANTITIES):
        obs_key = "obs_fc" if qlabel == "FC" else "obs_spc"
        for c, key in enumerate(keys):
            ax = axes[r][c]
            ev = events[key]
            b = bs_best_lag(ev[obs_key], ev[mod_key], max_steps)
            x0, y0 = bs_lag_pair(ev[obs_key], ev[mod_key], 0)
            xb, yb = bs_lag_pair(ev[obs_key], ev[mod_key], b["step"])
            ax.scatter(x0, y0, color="#bbbbbb", s=SC_SIZE, alpha=0.55,
                       linewidths=0, label="lag 0", zorder=2)
            ax.scatter(xb, yb, color=SIMULATIONS[key]["color"], s=SC_SIZE,
                       alpha=ALPHA_SC, linewidths=0, label="best lag", zorder=3)
            allx = np.concatenate([x0, xb]) if len(x0) or len(xb) else np.array([])
            ally = np.concatenate([y0, yb]) if len(y0) or len(yb) else np.array([])
            _add_one_to_one(ax, allx, ally, fs=10)
            ax.set_title(f"{SIMULATIONS[key]['label']}\n"
                         f"lag {b['minutes']:+.0f} min  "
                         f"(r {b['r0']:.2f} → {b['r']:.2f})", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{qlabel}: modelled [{FLUX_UNIT}]", fontsize=10)
            if r == nrow - 1:
                ax.set_xlabel(f"Observed [{FLUX_UNIT}]", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#dddddd", lw=0.6, zorder=0)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle("Blowing-snow flux: obs vs model at the best time lag",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    savefig(fig, f"bs_lag_scatter_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def print_bs_lag_summary(events, max_steps=6):
    """Console best-lag table for FC and SPC fluxes (one row per event)."""
    for _, mod_key, qlabel in BS_LAG_QUANTITIES:
        obs_key = "obs_fc" if qlabel == "FC" else "obs_spc"
        print(f"\n  {qlabel} flux - best time lag "
              f"(+ = modelled peak arrives after the measured one)")
        print(f"  {'Event':28s}  {'dt[min]':>7s}  {'r@0':>6s}  "
              f"{'lag[min]':>8s}  {'r@lag':>6s}  {'N':>3s}")
        for key, ev in events.items():
            b = bs_best_lag(ev[obs_key], ev[mod_key], max_steps)
            print(f"  {SIMULATIONS[key]['label']:28s}  {b['dt_min']:7.1f}  "
                  f"{b['r0']:6.3f}  {b['minutes']:+8.1f}  {b['r']:6.3f}  {b['n']:3d}")


def build_ceilometer(sim_key, domain=DEFAULT_DOMAIN,
                     blsn_dir=r"C:\Users\lory2\Desktop\tesi\BLSN\good_notebooks",
                     ceilo_root=r"D:\data\ceilo"):
    """WRF column blowing-snow vs ceilometer in-cloud backscatter (one event)."""
    sim = SIMULATIONS[sim_key]
    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    yr = t0.year
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])
    tt, qbs = fro.interp_column_sum(files, "bs_qi")
    wrf_s = pd.Series(qbs, index=pd.DatetimeIndex(tt).tz_localize(None)).sort_index().loc[t0:t1]
    try:
        blsn_df = fro.load_blsn_csv(os.path.join(blsn_dir, f"blsn_{yr}_v1505_jump_25.csv"))
        ceilo_s = fro.integrated_blsn_backscatter(os.path.join(ceilo_root, str(yr)),
                                                  blsn_df, t0, t1)
    except Exception as e:
        print(f"{sim_key}: ceilo load failed: {e}")
        ceilo_s = pd.Series([], dtype=float)
    wdf = pd.DataFrame({"qbs": wrf_s})
    ceilo_h = (fiv.align_obs(pd.DataFrame({"c": ceilo_s}), wdf,
                             window_min=ALIGN_WINDOW_MIN)["c"]
               if len(ceilo_s) else pd.Series(np.nan, index=wdf.index))
    return {"label": sim["label"], "color": SIMULATIONS[sim_key]["color"],
            "wrf": wrf_s, "obs_h": ceilo_h}


def build_wind(sim_key, domain=DEFAULT_DOMAIN):
    """Fine-mesh wind vs AWS WS2 at the anemometer height (one event)."""
    sim   = SIMULATIONS[sim_key]
    ws_z  = INSTRUMENT_HEIGHTS[sim_key]["ws_z"]
    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])
    tt, ws_fm, h_fm = interp_wind_sfc(files, target_z=ws_z)
    wdf = _ws_series(tt, ws_fm).to_frame("ws_sfc").loc[t0:t1]
    try:
        aws_h = fiv.align_aws(fiv.load_aws(sim["aws_file"]), wdf, window_min=ALIGN_WINDOW_MIN)
        wdf["ws_obs"] = aws_h["WS2_Avg"]
    except Exception as e:
        print(f"{sim_key}: AWS wind failed: {e}")
        wdf["ws_obs"] = np.nan
    return {"label": sim["label"], "color": SIMULATIONS[sim_key]["color"],
            "h_actual": float(np.nanmean(h_fm)), "ws_z": ws_z, "df": wdf}


def plot_wind(wind_by_event, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    """Stacked wind-speed time series, one panel per event."""
    keys = list(wind_by_event.keys())
    fig, axes = plt.subplots(len(keys), 1, figsize=_fs(13, 3.3 * len(keys)), squeeze=False)
    for ax, key in zip(axes[:, 0], keys):
        r = wind_by_event[key]; d = r["df"]
        ax.plot(d.index, d["ws_sfc"], color=r["color"], lw=2.0,
                label=f"WRF VW_SFC @ {r['h_actual']:.2f} m")
        ax.plot(d.index, d["ws_obs"], color=COLOR_OBS, lw=1.4, marker="o", ms=3,
                label=f"AWS WS2 @ {r['ws_z']:.2f} m")
        ax.set_ylabel(f"Wind speed [{WS_UNIT}]")
        ax.set_title(r["label"])
        ax.grid(True, alpha=0.4); ax.legend(fontsize=9, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Fine-mesh wind speed vs AWS", fontweight="bold", y=1.01)
    plt.tight_layout()
    savefig(fig, f"bs_finemesh_wind_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def build_flux_vs_wind(sim_key, domain=DEFAULT_DOMAIN):
    """Mass flux vs fine-mesh wind speed, obs and model (one event)."""
    sim   = SIMULATIONS[sim_key]
    h_cfg = INSTRUMENT_HEIGHTS[sim_key]
    ws_z, spc_z, fc_lo, fc_hi = h_cfg["ws_z"], h_cfg["spc_z"], h_cfg["fc_z_lo"], h_cfg["fc_z_hi"]
    fc_r  = fc_hi - fc_lo
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])

    bs_times, drift = fiv.interp_bs_drift_profile(files)
    _, sd_a  = fro.interp_surface(files, "SNOW_DRIFT")
    _, ust_a = fro.interp_surface(files, "UST")
    q_s = salt_flux_vec(np.array(sd_a), np.array(ust_a))
    with _NC(files[0]) as nc0:
        la, lo = fiv._get_latlon(nc0)
        h_ref = read_sn_c_height_at_t(nc0, la, lo, 0)
    h_aug     = np.concatenate([[H_SALT], h_ref])
    drift_aug = np.concatenate([q_s[:, None], drift], axis=1)
    tt, ws_fm, _ = interp_wind_sfc(files, target_z=ws_z)

    idx = pd.DatetimeIndex(bs_times).tz_localize(None)
    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    wrf = pd.DataFrame({
        "q_spc": fiv.extract_at_height(drift_aug, h_aug, spc_z) * 1e3,
        "Q_fc":  fiv.integrate_drift(drift_aug, h_aug, fc_lo, fc_hi) / fc_r * 1e3,
        "ws_mod": _ws_series(tt, ws_fm).reindex(idx),
    }, index=idx).sort_index().loc[t0:t1]
    wt = pd.DataFrame(index=wrf.index)
    spc_h = fiv.align_obs(load_spc_events(sim["spc_file"]),      wt, window_min=ALIGN_WINDOW_MIN)
    fc_h  = fiv.align_obs(fiv.load_flowcapt(sim["aws_file"]), wt, window_min=ALIGN_WINDOW_MIN)
    aws_h = fiv.align_aws(fiv.load_aws(sim["aws_file"]),      wt, window_min=ALIGN_WINDOW_MIN)
    return {"sim_key": sim_key, "label": sim["label"], "ws_z": ws_z,
            "ws_obs": aws_h["WS2_Avg"], "spc_obs": spc_h[SPC_FLUX_COL] * 1e3,
            "fc_obs": fc_h["PF_FC4"], "ws_mod": wrf["ws_mod"],
            "q_spc": wrf["q_spc"], "Q_fc": wrf["Q_fc"]}


def build_flux_vs_ustar(sim_key, domain=DEFAULT_DOMAIN):
    """Mass flux vs friction velocity, obs and model (one event).

    Section 8: uses the WRF UST diagnostic directly and is unaffected by the
    fine-mesh wind corrections.
    """
    sim   = SIMULATIONS[sim_key]
    h_cfg = INSTRUMENT_HEIGHTS[sim_key]
    spc_z, fc_lo, fc_hi = h_cfg["spc_z"], h_cfg["fc_z_lo"], h_cfg["fc_z_hi"]
    fc_r  = fc_hi - fc_lo
    t0, t1 = pd.Timestamp(sim["start"]), pd.Timestamp(sim["end"])
    yr = t0.year
    files = fiv.get_wrf_files(domain=domain, input_dir=sim["sim_dir"])

    bs_times, drift = fiv.interp_bs_drift_profile(files)
    _, sd_a  = fro.interp_surface(files, "SNOW_DRIFT")
    _, ust_a = fro.interp_surface(files, "UST")
    _, thr_a = fro.interp_surface(files, "THRESHOLD_USTAR")
    q_s = salt_flux_vec(np.array(sd_a), np.array(ust_a))
    with _NC(files[0]) as nc0:
        la, lo = fiv._get_latlon(nc0)
        h_ref = read_sn_c_height_at_t(nc0, la, lo, 0)
    h_aug     = np.concatenate([[H_SALT], h_ref])
    drift_aug = np.concatenate([q_s[:, None], drift], axis=1)

    idx = pd.DatetimeIndex(bs_times).tz_localize(None)
    wrf = pd.DataFrame({
        "ust": ust_a, "thr": thr_a,
        "q_spc": fiv.extract_at_height(drift_aug, h_aug, spc_z) * 1e3,
        "Q_fc":  fiv.integrate_drift(drift_aug, h_aug, fc_lo, fc_hi) / fc_r * 1e3,
    }, index=idx).sort_index().loc[t0:t1]
    wt = pd.DataFrame(index=wrf.index)
    spc_h = fiv.align_obs(load_spc_events(sim["spc_file"]),      wt, window_min=ALIGN_WINDOW_MIN)
    fc_h  = fiv.align_obs(fiv.load_flowcapt(sim["aws_file"]), wt, window_min=ALIGN_WINDOW_MIN)
    ust_ep = fiv.align_obs(pd.DataFrame({"u": fro.load_eddypro_ustar(EDDY_USTAR_FILES[yr])}),
                           wt, window_min=ALIGN_WINDOW_MIN)["u"]
    return {"sim_key": sim_key, "label": sim["label"], "wrf": wrf,
            "spc_obs": spc_h[SPC_FLUX_COL] * 1e3, "fc_obs": fc_h["PF_FC4"],
            "ust_obs": ust_ep}


def _flux_legend(model_color=COLOR_WRF):
    # Observations are black, the model is shown in the event colour; markers
    # distinguish the instrument (circle = SPC, diamond = FlowCapt).
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_OBS, markersize=7, label="SPC obs"),
        Line2D([0], [0], marker="d", color="w", markerfacecolor=COLOR_OBS, markersize=7, label="FC obs"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_color, markersize=7, label="SPC WRF"),
        Line2D([0], [0], marker="d", color="w", markerfacecolor=model_color, markersize=7, label="FC WRF"),
    ]


def plot_flux_vs_wind(fw_by_event, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    keys = list(fw_by_event.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=_fs(4.0 * len(keys), 4), squeeze=False)
    for j, key in enumerate(keys):
        r = fw_by_event[key]; ax = axes[0, j]
        mc = event_color(r["sim_key"])
        ax.scatter(r["ws_obs"], r["spc_obs"], c=COLOR_OBS, marker="o", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.scatter(r["ws_obs"], r["fc_obs"],  c=COLOR_OBS, marker="d", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.scatter(r["ws_mod"], r["q_spc"],   c=mc, marker="o", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.scatter(r["ws_mod"], r["Q_fc"],    c=mc, marker="d", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.set_title(r["label"], fontsize=TICK_FS + 1)
        ax.set_xlabel(f"Wind speed at {r['ws_z']:.2f} m [{WS_UNIT}]", fontsize=LABEL_FS)
        ax.grid(True, color=GRID_COLOR, lw=GRID_LW, alpha=0.5)
        ax.tick_params(labelsize=TICK_FS)
        if j == 0:
            ax.set_ylabel(f"Mass flux [{FLUX_UNIT}]", fontsize=LABEL_FS)
            ax.legend(handles=_flux_legend(mc), fontsize=TICK_FS - 2, loc="upper left")
    fig.suptitle("Mass flux vs wind speed", fontsize=LABEL_FS + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, f"flux_vs_wind_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


def plot_flux_vs_ustar(fu_by_event, domain=DEFAULT_DOMAIN, save=True, output_dir=OUTPUT_DIR):
    keys = list(fu_by_event.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=_fs(4.0 * len(keys), 4), squeeze=False)
    for j, key in enumerate(keys):
        r = fu_by_event[key]; d = r["wrf"]; ax = axes[0, j]
        mc = event_color(r["sim_key"])
        ax.scatter(r["ust_obs"], r["spc_obs"], c=COLOR_OBS, marker="o", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.scatter(r["ust_obs"], r["fc_obs"],  c=COLOR_OBS, marker="d", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.scatter(d["ust"], d["q_spc"], c=mc, marker="o", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.scatter(d["ust"], d["Q_fc"],  c=mc, marker="d", s=SC_SIZE, alpha=0.6, edgecolors="none")
        ax.axvline(d["thr"].median(), color=mc, ls="--", lw=1.2)
        ax.set_title(r["label"], fontsize=TICK_FS + 1)
        ax.set_xlabel(f"Friction velocity u* [{WS_UNIT}]", fontsize=LABEL_FS)
        ax.grid(True, color=GRID_COLOR, lw=GRID_LW, alpha=0.5)
        ax.tick_params(labelsize=TICK_FS)
        if j == 0:
            ax.set_ylabel(f"Mass flux [{FLUX_UNIT}]", fontsize=LABEL_FS)
            legend = _flux_legend(mc) + [
                Line2D([0], [0], color=mc, ls="--", lw=1.2, label="u*_thr")]
            ax.legend(handles=legend, fontsize=TICK_FS - 3, loc="upper left")
    fig.suptitle("Snow mass flux vs friction velocity",
                 fontsize=LABEL_FS + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    savefig(fig, f"flux_vs_ustar_{domain}.png", save, output_dir)
    plt.show()
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
#  Time-lag analysis of the WIND DIRECTION (circular)
#
#  The AWS direction (WD2) and the WRF 10 m diagnostic (wd10_deg) already live on
#  the same merged time grid built in ``build_merged_event``. To test whether a
#  temporal offset improves the (weak) agreement, the model is sampled at
#  ``obs_time + lag`` and the Jammalamadaka-SenGupta CIRCULAR correlation
#  (``circ_corr``) is recomputed for each lag.
#
#  Sign convention (matches the blowing-snow lag section): a POSITIVE lag means
#  the model has to be read *later* to match the current observation, i.e. the
#  modelled feature arrives ``lag`` minutes AFTER the measured one (model late).
#
#  The lag axis is continuous (default 30-min resolution), independent of the
#  native output spacing: the model direction is interpolated in time on its
#  cos/sin components (so the 0/360 wrap is handled correctly) and sampled at the
#  shifted observation times. The observations themselves are NEVER interpolated,
#  so N stays equal to the number of real obs whose shifted time still falls
#  inside the model coverage.
# ═════════════════════════════════════════════════════════════════════════════

def _circ_model_sampler(mod_s):
    """Return a function t -> direction[deg] interpolating *mod_s* in time.

    Interpolation is done on cos/sin so directions wrapping across 0/360 are
    handled correctly; times outside the model coverage return NaN.
    """
    m = mod_s.dropna()
    m = m[~m.index.duplicated(keep="first")].sort_index()
    xt = m.index.view("int64").astype(float)      # ns since epoch
    ang = np.deg2rad(m.to_numpy(float))
    mc, ms = np.cos(ang), np.sin(ang)

    def sample(times):
        x = pd.DatetimeIndex(times).view("int64").astype(float)
        c = np.interp(x, xt, mc, left=np.nan, right=np.nan)
        s = np.interp(x, xt, ms, left=np.nan, right=np.nan)
        return np.rad2deg(np.arctan2(s, c)) % 360.0

    return sample


def wd_lag_circ_correlation(merged, obs_col="WD2", mod_col="wd10_deg",
                            step_min=30, max_lag_min=240):
    """Circular correlation of wind direction vs applied time lag.

    Parameters
    ----------
    merged : DataFrame with *obs_col* (AWS) and *mod_col* (WRF) directions [deg].
    step_min : lag resolution in minutes (default 30).
    max_lag_min : largest |lag| scanned in minutes (default 240 = 4 h).

    Returns a DataFrame with columns ``lag_min``, ``r`` (circular), ``N``.
    Positive lag => model late.
    """
    o = merged[obs_col].dropna()
    o = o[~o.index.duplicated(keep="first")].sort_index()
    sample = _circ_model_sampler(merged[mod_col])

    lags = np.arange(-int(max_lag_min), int(max_lag_min) + 1, int(step_min))
    rows = []
    for L in lags:
        mv = sample(o.index + pd.Timedelta(minutes=int(L)))
        ov = o.to_numpy(float)
        mask = np.isfinite(ov) & np.isfinite(mv)
        r = circ_corr(ov[mask], mv[mask]) if mask.sum() >= 3 else np.nan
        rows.append({"lag_min": int(L), "r": r, "N": int(mask.sum())})
    return pd.DataFrame(rows)


def wd_best_lag(merged, obs_col="WD2", mod_col="wd10_deg",
                step_min=30, max_lag_min=240):
    """Best-lag summary dict for the wind-direction circular correlation."""
    c = wd_lag_circ_correlation(merged, obs_col, mod_col, step_min, max_lag_min)
    r0 = float(c.loc[c["lag_min"] == 0, "r"].iloc[0])
    valid = c.dropna(subset=["r"])
    if valid.empty:
        return {"lag_min": 0, "r": np.nan, "N": 0, "r0": r0, "table": c}
    best = valid.loc[valid["r"].idxmax()]
    return {"lag_min": int(best["lag_min"]), "r": float(best["r"]),
            "N": int(best["N"]), "r0": r0, "table": c}


def plot_wd_lag(merged, obs_col="WD2", mod_col="wd10_deg",
                step_min=30, max_lag_min=240, title=None,
                domain=DEFAULT_DOMAIN, save=False, output_dir=OUTPUT_DIR,
                fname=None):
    """Plot circular r vs applied time lag for the wind direction; mark best."""
    b = wd_best_lag(merged, obs_col, mod_col, step_min, max_lag_min)
    c = b["table"]
    fig, ax = plt.subplots(figsize=_fs(8, 5))
    sim_key = merged.attrs.get("sim_key")
    col = event_color(sim_key) if sim_key in SIMULATIONS else COLOR_WRF
    ax.plot(c["lag_min"], c["r"], color=col, lw=1.6, marker="o", ms=4)
    ax.plot(b["lag_min"], b["r"], color=col, marker="o", ms=12, mfc=col,
            mec="k", mew=1.4, zorder=5,
            label=f"best: {b['lag_min']:+d} min, r={b['r']:.3f}")
    ax.axvline(0, color="#888888", lw=1.0, ls="--")
    ax.axhline(0, color="#dddddd", lw=0.8)
    ax.plot(0, b["r0"], marker="s", ms=9, color="#444444", zorder=6,
            label=f"lag 0: r={b['r0']:.3f}")
    ax.set_xlabel("Applied time lag [min]  (+ = model late)", fontsize=LABEL_FS)
    ax.set_ylabel("Circular correlation r", fontsize=LABEL_FS)
    ax.set_title(title or "Wind direction: lagged circular correlation",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID_COLOR, lw=GRID_LW, ls="--", zorder=0)
    ax.legend(fontsize=TICK_FS, loc="best", framealpha=0.9)
    plt.tight_layout()
    if fname:
        savefig(fig, fname, save, output_dir)
    plt.show()
    plt.close(fig)
    return b


# ═════════════════════════════════════════════════════════════════════════════
#  Time-lag analysis of the LINEAR surface variables (wind speed, RH, ...)
#
#  Same idea and sign convention as the wind-direction section, but for
#  non-circular variables: the model is interpolated LINEARLY in time and paired
#  with the AWS observations at ``obs_time + lag``; the score is the ordinary
#  Pearson r. Positive lag => model late. Observations are never interpolated.
#  Works for any event in ``merged_by_event`` regardless of native dt (each
#  event has its own output spacing).
# ═════════════════════════════════════════════════════════════════════════════

def _lin_model_sampler(mod_s):
    """Return a function t -> value, linearly interpolating *mod_s* in time.

    Times outside the model coverage return NaN.
    """
    m = mod_s.dropna()
    m = m[~m.index.duplicated(keep="first")].sort_index()
    xt = m.index.view("int64").astype(float)      # ns since epoch
    mv = m.to_numpy(float)

    def sample(times):
        x = pd.DatetimeIndex(times).view("int64").astype(float)
        return np.interp(x, xt, mv, left=np.nan, right=np.nan)

    return sample


def var_lag_correlation(merged, obs_col, mod_col, step_min=30, max_lag_min=240,
                        circular=False):
    """Correlation of *obs_col* vs *mod_col* as a function of applied time lag.

    circular=False -> Pearson r with linear model interpolation (wind speed, RH).
    circular=True  -> circular r with cos/sin interpolation (wind direction).

    Returns a DataFrame with columns ``lag_min``, ``r``, ``N``.
    Positive lag => model late.
    """
    o = merged[obs_col].dropna()
    o = o[~o.index.duplicated(keep="first")].sort_index()
    sample = (_circ_model_sampler(merged[mod_col]) if circular
              else _lin_model_sampler(merged[mod_col]))

    lags = np.arange(-int(max_lag_min), int(max_lag_min) + 1, int(step_min))
    rows = []
    for L in lags:
        mv = sample(o.index + pd.Timedelta(minutes=int(L)))
        ov = o.to_numpy(float)
        mask = np.isfinite(ov) & np.isfinite(mv)
        if mask.sum() >= 3:
            r = (circ_corr(ov[mask], mv[mask]) if circular
                 else float(pearsonr(ov[mask], mv[mask])[0]))
        else:
            r = np.nan
        rows.append({"lag_min": int(L), "r": r, "N": int(mask.sum())})
    return pd.DataFrame(rows)


def var_best_lag(merged, obs_col, mod_col, step_min=30, max_lag_min=240,
                 circular=False):
    """Best-lag summary dict (lag_min, r, N, r0, table) for one variable."""
    c = var_lag_correlation(merged, obs_col, mod_col, step_min, max_lag_min, circular)
    r0 = float(c.loc[c["lag_min"] == 0, "r"].iloc[0])
    valid = c.dropna(subset=["r"])
    if valid.empty:
        return {"lag_min": 0, "r": np.nan, "N": 0, "r0": r0, "table": c}
    best = valid.loc[valid["r"].idxmax()]
    return {"lag_min": int(best["lag_min"]), "r": float(best["r"]),
            "N": int(best["N"]), "r0": r0, "table": c}


# (obs_col, mod_col, label, unit, circular)
LAG_VARS = [
    ("WS2_Avg", "ws_sfc_ms", "Wind speed",        WS_UNIT, False),
    ("rhw_pct", "rhw2_pct",  "Relative humidity", "%",     False),
]


def lag_summary_tables(merged_by_event, step_min=30, max_lag_min=240,
                       variables=LAG_VARS):
    """Per-event best-lag summary for each variable, as {title: DataFrame}."""
    tables = {}
    for obs_col, mod_col, label, unit, circ in variables:
        rows = []
        for key, mdf in merged_by_event.items():
            if obs_col not in mdf or mod_col not in mdf:
                continue
            b = var_best_lag(mdf, obs_col, mod_col, step_min, max_lag_min, circ)
            rows.append({"Event": SIMULATIONS[key]["label"],
                         "r (lag 0)": round(b["r0"], 3),
                         "best lag [min]": b["lag_min"],
                         "best lag [h]": round(b["lag_min"] / 60.0, 2),
                         "r (best)": round(b["r"], 3),
                         "Δr": round(b["r"] - b["r0"], 3),
                         "N": b["N"]})
        tables[f"{label} [{unit}] – best time lag"] = pd.DataFrame(rows)
    return tables


def plot_lag_curves(merged_by_event, step_min=30, max_lag_min=240,
                    variables=LAG_VARS, domain=DEFAULT_DOMAIN,
                    save=False, output_dir=OUTPUT_DIR, fname=None):
    """Correlation vs applied time lag, one panel per variable, one line/event.

    A filled marker flags each event's best lag. Positive lag = model late.
    """
    fig, axes = plt.subplots(1, len(variables),
                             figsize=_fs(6.4 * len(variables), 5.2), squeeze=False)
    for ax, (obs_col, mod_col, label, unit, circ) in zip(axes[0], variables):
        for key, mdf in merged_by_event.items():
            if obs_col not in mdf or mod_col not in mdf:
                continue
            c = var_lag_correlation(mdf, obs_col, mod_col, step_min, max_lag_min, circ)
            col = SIMULATIONS[key]["color"]
            ax.plot(c["lag_min"], c["r"], color=col, lw=1.5, marker="o", ms=3,
                    label=SIMULATIONS[key]["label"])
            v = c.dropna(subset=["r"])
            if not v.empty:
                bi = v.loc[v["r"].idxmax()]
                ax.plot(bi["lag_min"], bi["r"], color=col, marker="o", ms=11,
                        mfc=col, mec="k", mew=1.2, zorder=5)
        ax.axvline(0, color="#888888", lw=1.0, ls="--")
        ax.axhline(0, color="#dddddd", lw=0.8)
        ax.set_xlabel("Applied time lag [min]  (+ = model late)", fontsize=TICK_FS)
        ax.set_ylabel("Pearson r", fontsize=TICK_FS)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID_COLOR, lw=GRID_LW, ls="--", zorder=0)
        ax.tick_params(labelsize=TICK_FS)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(merged_by_event),
               bbox_to_anchor=(0.5, 1.06), fontsize=TICK_FS, framealpha=0.9)
    _vt = ", ".join(v[2] for v in variables)
    fig.suptitle(f"Lagged cross-correlation: {_vt}",
                 fontsize=14, fontweight="bold", y=1.13)
    plt.tight_layout()
    if fname:
        savefig(fig, fname, save, output_dir)
    plt.show()
    plt.close(fig)


def _shifted_model_series(merged, obs_col, mod_col, lag_min, circular=False):
    """Model sampled at ``obs_time + lag`` and returned on the observation grid.

    This is the model curve *aligned* to the observations by the applied lag, so
    it can be overplotted directly against ``merged[obs_col]``.
    """
    o = merged[obs_col].dropna()
    o = o[~o.index.duplicated(keep="first")].sort_index()
    sample = (_circ_model_sampler(merged[mod_col]) if circular
              else _lin_model_sampler(merged[mod_col]))
    vals = sample(o.index + pd.Timedelta(minutes=int(lag_min)))
    return pd.Series(vals, index=o.index)


def plot_lag_timeseries(merged_by_event, variables=LAG_VARS, step_min=30,
                        max_lag_min=240, domain=DEFAULT_DOMAIN,
                        save=False, output_dir=OUTPUT_DIR, fname=None):
    """Time series at each event's best lag (rows = variables, cols = events).

    Each panel shows the AWS observations, the raw model (lag 0, dashed) and the
    model shifted to the best lag (solid). Titles report r0 → r_best.
    """
    keys = list(merged_by_event.keys())
    nrow, ncol = len(variables), len(keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=_fs(5.4 * ncol, 3.2 * nrow),
                             squeeze=False, sharex="col")
    for r, (obs_col, mod_col, label, unit, circ) in enumerate(variables):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            mdf = merged_by_event[key]
            if obs_col not in mdf or mod_col not in mdf:
                ax.axis("off"); continue
            b = var_best_lag(mdf, obs_col, mod_col, step_min, max_lag_min, circ)
            col = SIMULATIONS[key]["color"]
            o, m = mdf[obs_col], mdf[mod_col]
            sh = _shifted_model_series(mdf, obs_col, mod_col, b["lag_min"], circ)
            ov = mask_direction_wraps(o.values) if circ else o.values
            mv = mask_direction_wraps(m.values) if circ else m.values
            shv = mask_direction_wraps(sh.values) if circ else sh.values
            ax.plot(o.index, ov, color=COLOR_OBS, lw=1.3, label="AWS")
            ax.plot(m.index, mv, color=col, lw=1.1, ls="--", alpha=0.55,
                    label="CRYOWRF (lag 0)")
            ax.plot(sh.index, shv, color=col, lw=1.7,
                    label=f"CRYOWRF (lag {b['lag_min']:+d} min)")
            ax.set_title(f"{SIMULATIONS[key]['label']}\n"
                         f"r {b['r0']:.2f} → {b['r']:.2f}  "
                         f"({b['lag_min']:+d} min)", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{label} [{unit}]", fontsize=10)
            ax.grid(True, color=GRID_COLOR, lw=GRID_LW, ls="--", alpha=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
            ax.tick_params(labelsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle("Time series at the best time lag (dashed = lag 0)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if fname:
        savefig(fig, fname, save, output_dir)
    plt.show()
    plt.close(fig)


def show_tables(tables):
    """Display a dict of {title: DataFrame} in a notebook or plain console."""
    try:
        from IPython.display import display as _disp
    except Exception:
        _disp = None
    for title, df in tables.items():
        print(f"\n-- {title} --")
        if _disp is not None:
            _disp(df)
        else:
            print(df.to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
#  DEFINITIVE TIMING-vs-ACCURACY DIAGNOSTICS
#
#  Goal: decide, for each event, whether a poor model score is a *timing/phase*
#  error (the right feature at the wrong time -> a lag fixes it) or a genuine
#  *accuracy* error (wrong values at every lag). The verdict rests on four
#  independent tests, none of which is conclusive alone:
#
#   1. Cross-variable lag COHERENCE. A real phase shift advects the whole
#      weather system together, so pressure, temperature, wind speed and wind
#      direction should share ONE lag (same sign, similar magnitude). If each
#      variable prefers a different lag (opposite signs included), there is no
#      coherent shift -> accuracy, not timing.
#   2. Pressure / temperature REFERENCE. These are the well-resolved synoptic
#      variables. If their best lag is ~0 and their lag-0 r is already high, the
#      synoptic timing is correct by construction, so any wind/RH lag cannot be
#      blamed on synoptic phase error.
#   3. SIGNIFICANCE of the improvement. With N ~ 16-24 a correlation is very
#      noisy. Two guards: (a) the Fisher 95% CIs of r(lag0) and r(best) must be
#      DISJOINT for the improvement to be meaningful; (b) the best lag must be
#      INTERIOR, not pinned at the +/-max_lag window edge (an edge optimum means
#      the search never found a peak -> unresolved / spurious). A window-widening
#      check confirms whether interior lags stay put.
#   4. BOOTSTRAP stability. A circular moving-block bootstrap over the paired
#      samples gives the sampling spread of the best lag and of
#      dr = r(best) - r(lag0), plus P(dr > 0). A timing signal is stable: narrow
#      lag spread and P(dr>0) ~ 1. A spurious one is not.
# ═════════════════════════════════════════════════════════════════════════════

# All five validation variables in lag form: (obs, mod, label, unit, circular)
ALL_LAG_VARS = [
    ("psfc_hPa_aws", "psfc_hPa_wrf", "Surface pressure",      "hPa",   False),
    ("theta_C",      "theta2_C",     "Potential temperature", "deg C", False),
    ("WS2_Avg",      "ws_sfc_ms",    "Wind speed",            WS_UNIT, False),
    ("WD2",          "wd10_deg",     "Wind direction",        "deg",   True),
    ("rhw_pct",      "rhw2_pct",     "Relative humidity",     "%",     False),
]

# Variables that carry the synoptic *timing* signal (the coherence test).
# Humidity is excluded: it is dominated by local moisture/cloud errors, not
# advection, so it is not a reliable phase tracer.
DYN_LAG_LABELS = ["Surface pressure", "Potential temperature",
                  "Wind speed", "Wind direction"]


def fisher_ci(r, n, alpha=0.05):
    """Two-sided (1-alpha) CI for a Pearson correlation via Fisher z."""
    if r is None or not np.isfinite(r) or n is None or n < 4 or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = _norm.ppf(1.0 - alpha / 2.0)
    return (float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se)))


def _fmt_ci(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "-"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def _lag_pair_matrix(merged, obs_col, mod_col, step_min=30, max_lag_min=240,
                     circular=False):
    """(ov, M, lags): obs values ov[i] and model matrix M[i,k] = model sampled
    at obs_time_i + lags[k]. NaN where the shifted time is outside coverage."""
    o = merged[obs_col].dropna()
    o = o[~o.index.duplicated(keep="first")].sort_index()
    ov = o.to_numpy(float)
    sample = (_circ_model_sampler(merged[mod_col]) if circular
              else _lin_model_sampler(merged[mod_col]))
    lags = np.arange(-int(max_lag_min), int(max_lag_min) + 1, int(step_min))
    cols = [sample(o.index + pd.Timedelta(minutes=int(L))) for L in lags]
    M = np.column_stack(cols) if cols else np.empty((len(ov), 0))
    return ov, M, lags


def _r_vec(ov, mv, circular=False):
    """Correlation of aligned vectors, ignoring NaNs. Returns (r, N)."""
    mask = np.isfinite(ov) & np.isfinite(mv)
    n = int(mask.sum())
    if n < 3:
        return np.nan, n
    if circular:
        return float(circ_corr(ov[mask], mv[mask])), n
    return float(pearsonr(ov[mask], mv[mask])[0]), n


def best_lag_record(merged, obs_col, mod_col, label, unit, circular,
                    step_min=30, max_lag_min=240):
    """One-variable diagnostic row: r0, best lag, r_best, Fisher CIs, edge flag,
    and whether the CIs of r0 and r_best are disjoint (significant improvement).
    """
    ov, M, lags = _lag_pair_matrix(merged, obs_col, mod_col, step_min,
                                   max_lag_min, circular)
    if M.shape[1] == 0:
        return {"Variable": label, "r0": np.nan, "CI0": "-",
                "best lag [min]": 0, "best lag [h]": 0.0, "r_best": np.nan,
                "CI_best": "-", "Δr": np.nan, "N": 0, "edge?": "",
                "sig Δr?": "-"}
    rs = np.array([_r_vec(ov, M[:, k], circular)[0] for k in range(len(lags))])
    ns = np.array([_r_vec(ov, M[:, k], circular)[1] for k in range(len(lags))])
    i0 = int(np.where(lags == 0)[0][0])
    r0, n0 = rs[i0], int(ns[i0])
    if not np.any(np.isfinite(rs)):
        kbest, r_best, n_best, lag_best = i0, np.nan, n0, 0
    else:
        kbest = int(np.nanargmax(rs))
        r_best, n_best, lag_best = float(rs[kbest]), int(ns[kbest]), int(lags[kbest])
    lo0, hi0 = fisher_ci(r0, n0) if not circular else (np.nan, np.nan)
    lob, hib = fisher_ci(r_best, n_best) if not circular else (np.nan, np.nan)
    at_edge = abs(lag_best) >= int(max_lag_min)
    sig = (not circular and np.isfinite(lob) and np.isfinite(hi0) and lob > hi0)
    return {"Variable": label,
            "r0": round(r0, 3), "CI0": _fmt_ci(lo0, hi0),
            "best lag [min]": lag_best, "best lag [h]": round(lag_best / 60.0, 2),
            "r_best": round(r_best, 3), "CI_best": _fmt_ci(lob, hib),
            "Δr": round(r_best - r0, 3), "N": n_best,
            "edge?": "EDGE" if at_edge else "",
            "sig Δr?": ("yes" if sig else "no") if not circular else "-"}


def timing_diagnostics_tables(merged_by_event, step_min=30, max_lag_min=240,
                              variables=ALL_LAG_VARS):
    """Per-event table over ALL variables. Returns {event_label: DataFrame}."""
    tables = {}
    for key, mdf in merged_by_event.items():
        rows = [best_lag_record(mdf, oc, mc, lb, un, ci, step_min, max_lag_min)
                for oc, mc, lb, un, ci in variables
                if oc in mdf and mc in mdf]
        tables[f"{SIMULATIONS[key]['label']} – lag diagnostics"] = pd.DataFrame(rows)
    return tables


def block_bootstrap_lag(merged, obs_col, mod_col, circular=False,
                        step_min=30, max_lag_min=240, n_boot=1000,
                        block=3, seed=0):
    """Circular moving-block bootstrap of the best-lag search.

    Resamples blocks of consecutive observation points (preserving short-range
    autocorrelation), re-runs the full lag scan on each resample and records the
    argmax lag and dr = r(best)-r(lag0). Returns the observed best lag/r plus the
    bootstrap 95% interval on the lag, its IQR, P(dr>0) and the median dr.
    """
    ov, M, lags = _lag_pair_matrix(merged, obs_col, mod_col, step_min,
                                   max_lag_min, circular)
    n = len(ov)
    if n < 4 or M.shape[1] == 0:
        return {"best_lag": 0, "best_r": np.nan, "r0": np.nan,
                "lag_CI": (np.nan, np.nan), "lag_IQR": (np.nan, np.nan),
                "P(Δr>0)": np.nan, "Δr_median": np.nan, "N": n}
    i0 = int(np.where(lags == 0)[0][0])
    rs = np.array([_r_vec(ov, M[:, k], circular)[0] for k in range(len(lags))])
    r0 = rs[i0]
    kbest = int(np.nanargmax(rs)) if np.any(np.isfinite(rs)) else i0
    best_lag, best_r = int(lags[kbest]), float(rs[kbest])
    rng = np.random.default_rng(seed)
    nblk = max(1, int(np.ceil(n / block)))
    blag, bdr = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=nblk)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        ovb, Mb = ov[idx], M[idx]
        rsb = np.array([_r_vec(ovb, Mb[:, k], circular)[0]
                        for k in range(len(lags))])
        if not np.any(np.isfinite(rsb)):
            continue
        kb = int(np.nanargmax(rsb))
        blag.append(int(lags[kb]))
        bdr.append(float(rsb[kb] - rsb[i0]))
    blag, bdr = np.asarray(blag, float), np.asarray(bdr, float)
    if blag.size == 0:
        return {"best_lag": best_lag, "best_r": best_r, "r0": float(r0),
                "lag_CI": (np.nan, np.nan), "lag_IQR": (np.nan, np.nan),
                "P(Δr>0)": np.nan, "Δr_median": np.nan, "N": n}
    return {"best_lag": best_lag, "best_r": best_r, "r0": float(r0),
            "lag_CI": (float(np.percentile(blag, 2.5)),
                       float(np.percentile(blag, 97.5))),
            "lag_IQR": (float(np.percentile(blag, 25)),
                        float(np.percentile(blag, 75))),
            "P(Δr>0)": float(np.mean(bdr > 0)),
            "Δr_median": float(np.median(bdr)), "N": n}


def bootstrap_lag_table(merged_by_event, variables=None, step_min=30,
                        max_lag_min=240, n_boot=1000, block=3, seed=0):
    """Bootstrap best-lag stability for the dynamical variables, one row per
    (event, variable). Returns a single DataFrame."""
    if variables is None:
        variables = [v for v in ALL_LAG_VARS if v[2] in DYN_LAG_LABELS]
    rows = []
    for key, mdf in merged_by_event.items():
        for oc, mc, lb, un, ci in variables:
            if oc not in mdf or mc not in mdf:
                continue
            b = block_bootstrap_lag(mdf, oc, mc, ci, step_min, max_lag_min,
                                    n_boot, block, seed)
            rows.append({"Event": SIMULATIONS[key]["label"], "Variable": lb,
                         "best lag [min]": b["best_lag"],
                         "lag 95% CI [min]": f"[{b['lag_CI'][0]:.0f}, "
                                             f"{b['lag_CI'][1]:.0f}]",
                         "lag IQR [min]": f"[{b['lag_IQR'][0]:.0f}, "
                                          f"{b['lag_IQR'][1]:.0f}]",
                         "P(Δr>0)": round(b["P(Δr>0)"], 2),
                         "Δr median": round(b["Δr_median"], 3),
                         "N": b["N"]})
    return pd.DataFrame(rows)


def timing_verdict(merged_by_event, step_min=30, max_lag_min=240,
                   n_min=25, coherence_tol_min=90, dr_min=0.15,
                   variables=ALL_LAG_VARS):
    """Combine the four tests into a transparent per-event verdict.

    Rules (evidence is printed alongside the label, not hidden):
      * N < n_min                         -> INCONCLUSIVE (undersampled).
      * >=2 dynamical vars with a SIGNIFICANT (disjoint Fisher CI), INTERIOR
        improvement (Δr>=dr_min) whose best lags agree in sign and within
        coherence_tol_min                 -> TIMING (model lag ~ <shared lag>).
      * otherwise, if best lags are edge-pinned / incoherent / not significant
                                          -> ACCURACY (no coherent phase shift).
    Returns a summary DataFrame and prints a short reasoned verdict per event.
    """
    label2col = {lb: (oc, mc, ci) for oc, mc, lb, un, ci in variables}
    rows = []
    for key, mdf in merged_by_event.items():
        recs = {lb: best_lag_record(mdf, oc, mc, lb, "", ci, step_min, max_lag_min)
                for lb, (oc, mc, ci) in label2col.items()
                if oc in mdf and mc in mdf}
        Ns = [r["N"] for r in recs.values() if r["N"]]
        Nmin = min(Ns) if Ns else 0
        # dynamical variables that show a significant, interior improvement
        good = [lb for lb in DYN_LAG_LABELS
                if lb in recs and recs[lb]["sig Δr?"] == "yes"
                and recs[lb]["edge?"] != "EDGE"
                and np.isfinite(recs[lb]["Δr"]) and recs[lb]["Δr"] >= dr_min]
        good_lags = [recs[lb]["best lag [min]"] for lb in good]
        coherent = (len(good) >= 2 and (max(good_lags) - min(good_lags))
                    <= coherence_tol_min
                    and len({np.sign(x) for x in good_lags if x != 0}) <= 1)
        n_edge = sum(1 for lb in DYN_LAG_LABELS
                     if lb in recs and recs[lb]["edge?"] == "EDGE")
        p_rec = recs.get("Surface pressure", {})
        p_lag = p_rec.get("best lag [min]", np.nan)
        if Nmin < n_min:
            verdict = f"INCONCLUSIVE (N={Nmin} < {n_min})"
        elif coherent:
            shared = int(np.median(good_lags))
            verdict = (f"TIMING (coherent lag ~{shared:+d} min across "
                       f"{', '.join(good)})")
        else:
            verdict = "ACCURACY (no coherent, significant phase shift)"
        rows.append({"Event": SIMULATIONS[key]["label"], "N (min)": Nmin,
                     "pressure lag [min]": p_lag,
                     "sig+interior dyn vars": ", ".join(good) if good else "none",
                     "edge-pinned dyn vars": n_edge, "Verdict": verdict})
    summary = pd.DataFrame(rows)
    for _, r in summary.iterrows():
        print(f"• {r['Event']}: {r['Verdict']}")
        print(f"    N={r['N (min)']}, pressure best lag={r['pressure lag [min]']} "
              f"min, significant+interior dynamical vars: "
              f"{r['sig+interior dyn vars']}, "
              f"edge-pinned: {r['edge-pinned dyn vars']}\n")
    return summary
