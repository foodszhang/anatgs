"""Respiratory surrogate decomposition utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, hilbert


def normalize_signal(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def interpolate_trace(trace_time: np.ndarray, trace: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    trace_time = np.asarray(trace_time, dtype=np.float32).reshape(-1)
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    target_time = np.asarray(target_time, dtype=np.float32).reshape(-1)
    if trace_time.size != trace.size:
        raise ValueError("trace_time and trace length mismatch")
    order = np.argsort(trace_time)
    return np.interp(target_time, trace_time[order], trace[order]).astype(np.float32)


def _estimate_period_seconds(timestamps: np.ndarray, phase: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    up = np.unwrap(np.asarray(phase, dtype=np.float32).reshape(-1))
    peaks, _ = find_peaks(up)
    if peaks.size < 2:
        return np.full_like(t, fill_value=max(float(np.median(np.diff(t))), 1e-3), dtype=np.float32)
    peak_times = t[peaks]
    periods = np.diff(peak_times)
    centers = 0.5 * (peak_times[:-1] + peak_times[1:])
    out = np.interp(t, centers, periods, left=periods[0], right=periods[-1])
    return np.asarray(out, dtype=np.float32)


def _rolling_peak_to_trough(sig: np.ndarray, window: int) -> np.ndarray:
    s = np.asarray(sig, dtype=np.float32).reshape(-1)
    n = s.size
    w = max(int(window), 3)
    half = w // 2
    out = np.zeros_like(s)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        cur = s[lo:hi]
        out[i] = float(np.max(cur) - np.min(cur))
    return out.astype(np.float32)


def decompose_surrogate(
    trace: np.ndarray,
    timestamps: np.ndarray,
    window_seconds: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - features [N,5] = [sin(phi), cos(phi), a, T, d]
      - scalar normalized surrogate [N]
    """
    r = np.asarray(trace, dtype=np.float32).reshape(-1)
    t = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    if r.size != t.size:
        raise ValueError("trace and timestamps length mismatch")
    if r.size < 4:
        raise ValueError("trace too short for decomposition")

    r_norm = normalize_signal(r)
    analytic = hilbert(r_norm.astype(np.float64))
    phase = np.angle(analytic).astype(np.float32)
    sin_phi = np.sin(phase).astype(np.float32)
    cos_phi = np.cos(phase).astype(np.float32)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    window = max(int(round(float(window_seconds) / max(dt, 1e-4))), 3)
    amp = normalize_signal(_rolling_peak_to_trough(r_norm, window=window))
    period = normalize_signal(_estimate_period_seconds(t, phase))
    grad = np.gradient(r_norm, edge_order=1)
    direction = np.where(grad >= 0.0, 1.0, -1.0).astype(np.float32)

    features = np.stack([sin_phi, cos_phi, amp, period, direction], axis=-1).astype(np.float32)
    return features, r_norm.astype(np.float32)


def phase_only_from_timestamps(timestamps: np.ndarray, n_cycles: float = 1.0) -> np.ndarray:
    t = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    phi = 2.0 * np.pi * (t * float(n_cycles))
    return np.stack([np.sin(phi), np.cos(phi)], axis=-1).astype(np.float32)

