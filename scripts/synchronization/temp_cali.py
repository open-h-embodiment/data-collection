#!/usr/bin/env python

"""
Estimate temporal offset between two periodic signals using sine wave fitting.

This script provides a function `estimate_offset` to determine the time offset between two
noisy but periodic data streams (e.g., signals from two sensors) by fitting each to a sine wave
model and comparing their phase difference. This is particularly useful when synchronizing
unsynchronized periodic signals in time-critical applications like robotics or sensor fusion.

How It Works:
-------------
- Fits both y1 and y2 to sine wave models of the form A·sin(ω·t + φ).
- Estimates the phase difference between the fitted sine waves.
- Converts phase difference to time offset using: Δt = Δφ / ω.
- Returns both the principal solution and its phase-shifted alternative.

Notes:
------
- Requires the signals to be reasonably well-approximated by sinusoids of the same frequency.
- Works best when noise is low and frequency ω is known and consistent.
- Useful in temporal calibration of periodic motion sensors (e.g., robot joints, images, etc.).

"""

import numpy as np
from scipy.optimize import curve_fit


def estimate_offset(t1, y1, t2, y2, w0, a1=1, a2=1):
    '''
    Estimate the temporal offset between two periodic data streams using sine fitting.
    Parameters:
    -----------
    t1 : np.ndarray of shape (N,) Time stamps of the first data stream.
    y1 : np.ndarray of shape (N,) Signal values corresponding to t1 
    t2 : np.ndarray of shape (N,) Time stamps of the second data stream.
    y2 : np.ndarray of shape (N,) Signal values corresponding to t2.
    w0 : float Known angular frequency (rad/s) of the periodic motion 
    a1 : float, optional (default=1) Initial amplitude guess for y1.
    a2 : float, optional (default=1) Initial amplitude guess for y2. 
    Returns:
    --------
    estimated_offset_1 : float
        Primary estimated temporal offset between the two signals (in seconds).
        A positive value indicates that y2 is ahead of y1, and negative means delayed.
    estimated_offset_2 : float
        Alternate solution due to the periodic nature of sine wave fitting,
        offset by half a period (π/w) from the first solution.
    Notes:
    ------
    - The function fits both signals to sine curves using non-linear least squares.
    - It computes the phase difference between the two fitted curves.
    - The temporal offset is inferred from the phase difference: Δt = Δφ / ω.
    - Since sine waves are periodic, two plausible offsets are returned.
    '''
    # --- Define sine model to be fitted: A * sin(w * t + φ) ---
    def sine_model(t, A, w, phi):
        return A * np.sin(w * t + phi)
    
    # --- Fit the first signal ---
    t0 = t1[0] # Shift time origin for numerical stability
    params1, _ = curve_fit(sine_model, t1-t0, y1, p0=[a1, w0, 0])
    A1, w1, phi1 = params1

    # --- Fit the second signal ---
    params2, _ = curve_fit(sine_model, t2-t0, y2, p0=[a2, w0, 0])
    A2, w2, phi2 = params2

    # --- Estimate time offset from phase difference ---
    # Δφ = w * Δt  => Δt = Δφ / w
    phase_diff = (phi2 - phi1)
    estimated_w = (w1+w2)/2

    # Normalize the phase difference to the range [-π, π] to avoid wrap-around issues
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    estimated_offset_1 = phase_diff / estimated_w

    # Due to sine periodicity, there's a second possible solution offset by ±π/w
    if estimated_offset_1 > 0:
        estimated_offset_2 = estimated_offset_1-np.pi/estimated_w
    else:
        estimated_offset_2 = estimated_offset_1+np.pi/estimated_w

    print(f"Estimated time offset:  {estimated_offset_1:.4f} or {estimated_offset_2:.4f} seconds")

    return estimated_offset_1, estimated_offset_2