import sys
from argparse import ArgumentParser
from glob import glob
import json
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from astropy.constants import c

from utils.image_handling import getallfluxes
from utils.RM_functions import functionRMdepol, function_synch_simple
from pol_phase_rot import PhaseRotate

plt.rcParams.update({
    "font.family": "Serif",
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18
})

def phase_rot(ms_in: str = None, h5_out: str = None, intercept: float = None, rm: float = None):
    """
    Make template h5 with default values (phase=0 and amplitude=1) and convert to phase rotate matrix
    for polarization alignment between different observations.
    Args:
        h5_in: Input h5parm
        h5_out: Output h5parm
        intercept: Intercept value
        rm: RM value
    """

    phaserot = PhaseRotate(ms_in=ms_in, h5_out=h5_out)
    phaserot.make_template(polrot=True)
    phaserot.rotate(intercept=intercept, rotation_measure=rm)
    phaserot.h5.close()


def fit_RM(i_fits: list = None, u_fits: list = None, q_fits: list = None, regionfile: str = None):
    """
    Find Rotation Measure and intercept for the polarisation angle

    Args:
        i_fits: Stokes I FITS channel images
        u_fits: Stokes U FITS channel images
        q_fits: Stokes Q FITS channel images
        regionfile: Region file
        h5_in: h5parm input
        ref_RM: reference RM
        ref_chi0: reference intercept for polarisation angle

    Return: RM, chi0, lambda ref, L-number
    """

    i_fits = sorted(i_fits)
    u_fits = sorted(u_fits)
    q_fits = sorted(q_fits)

    if len(i_fits) == 0 and len(u_fits) == 0 and len(q_fits) == 0:
        sys.exit("ERROR: No images selected/found")

    freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U = getallfluxes(i_fits, q_fits, u_fits, regionfile)

    freqvec_MHz = freqvec / 1e6

    # Filter bad images
    mask = np.isfinite(Iflux) & np.isfinite(Qflux) & np.isfinite(Uflux)
    sort_idx = np.argsort(freqvec[mask])
    arrays = [freqvec, freqvec_MHz, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U]
    freqvec, freqvec_MHz, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U = [a[mask][sort_idx] for a in arrays]

    lambda2 = (c.value / freqvec) ** 2

    lambdaref2 = np.median(lambda2)
    freqref = np.median(freqvec)

    print(f"lambdaref2 = {lambdaref2:.4f} m^2")

    # Fit Stokes I
    fitI, pcov_I = scipy.optimize.curve_fit(
        lambda freq, norm, alpha: function_synch_simple(freq, norm, alpha, freq_ref=freqref),
        freqvec,
        Iflux,
        p0=[np.nanmedian(Iflux), -0.7],
        sigma=sigma_I,
        maxfev=100000
    )
    Imodel = function_synch_simple(freqvec, *fitI, freq_ref=freqref)

    # Fractional polarisation
    q = Qflux / Imodel
    u = Uflux / Imodel
    P = q + 1j * u

    # Peak RM search
    rm_grid = np.arange(-2000, 2000, 0.1)
    fdf = np.array([
        np.abs(np.sum(P * np.exp(-2j * rm * lambda2)))
        for rm in rm_grid
    ])
    peak_idx = np.argmax(fdf)

    if 0 < peak_idx < len(rm_grid) - 1:
        y0, y1, y2 = fdf[peak_idx - 1], fdf[peak_idx], fdf[peak_idx + 1]
        dm = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        rm_init = rm_grid[peak_idx] + dm * (rm_grid[1] - rm_grid[0])
    else:
        rm_init = rm_grid[peak_idx]

    print(f"RM synthesis peak: {rm_init:.3f} rad/m^2")

    # Derotate P to lambdaref2 and take angle
    P_derotated = P * np.exp(-2j * rm_init * (lambda2 - lambdaref2))
    chi0_init = 0.5 * np.angle(np.sum(P_derotated))
    print(f"chi0 initial estimate: {chi0_init:.3f} rad")

    p0 = np.median(np.abs(P))

    x0_QU_depol = np.array([p0, rm_init, chi0_init, 0.03])

    rm_window = 5.0  # rad/m^2 — widen if needed
    bounds_lo = [0,       rm_init - rm_window, -np.pi, 0.0]
    bounds_hi = [np.inf,  rm_init + rm_window,  np.pi, 10.0]

    functionRMdepol_ref = partial(functionRMdepol, lambdaref2=lambdaref2)

    fitQU_depol, pcov_QU_depol = scipy.optimize.curve_fit(
        functionRMdepol_ref,
        lambda2,
        np.append(q, u),
        p0=x0_QU_depol,
        sigma=np.append(sigma_Q / Imodel, sigma_U / Imodel),
        bounds=(bounds_lo, bounds_hi),
        maxfev=300000
    )

    err = np.sqrt(np.diag(pcov_QU_depol))

    if fitQU_depol[0] <= 0:
        sys.exit("WARNING: negative polarization fraction - fitting may be unstable")

    fitstr = (
        f"fit: RM={fitQU_depol[1]:.3f} +/- {err[1]:.3f} rad m^-2; "
        f"chi0={fitQU_depol[2]:.3f} +/- {err[2]:.3f} rad; "
        f"sigmaRM2={fitQU_depol[3]:.3f} +/- {err[3]:.3f} rad^2 m^-4; "
        f"p0={fitQU_depol[0]:.3f} +/- {err[0]:.3f}"
    )

    print(fitstr)

    ##### PLOTTING #####

    lam2 = lambda2
    pol_model = Imodel * fitQU_depol[0] * np.exp(-2 * fitQU_depol[3] * lam2 ** 2)
    phase = 2 * (fitQU_depol[1] * (lam2 - lambdaref2) + fitQU_depol[2])
    Qmodel = pol_model * np.cos(phase)
    Umodel = pol_model * np.sin(phase)

    ##### PLOTTING #####

    # --- Plot Stokes I, Q, U ---
    panels = [
        ("Stokes I", Iflux, sigma_I, function_synch_simple(freqvec_MHz, *fitI, freq_ref=150.)),
        ("Stokes Q", Qflux, sigma_Q, Qmodel),
        ("Stokes U", Uflux, sigma_U, Umodel),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11.25))
    for ax, (title, flux, sigma, model) in zip(axes, panels):
        ax.errorbar(lam2, flux, yerr=sigma, linestyle="", marker="s",
                    color='black', markersize=5)
        ax.plot(lam2, model, color='darkred', linestyle='--', label=f'{title} fit')
        ax.set_xlabel(r'$\lambda^2$ [m$^2$]')
        ax.set_ylabel('Flux [Jy]')
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig("StokesIQU_wav2.png", dpi=150)
    plt.close()

    # --- Plot Polarization Angle ---
    sort_lam = np.argsort(lam2)
    lam2_s = lam2[sort_lam]
    polangle = 0.5 * np.arctan2(Uflux[sort_lam], Qflux[sort_lam])
    polangle_sigma = 0.5 * np.sqrt(
        (sigma_U[sort_lam] ** 2 * Qflux[sort_lam] ** 2 +
         sigma_Q[sort_lam] ** 2 * Uflux[sort_lam] ** 2)
        / (Uflux[sort_lam] ** 2 + Qflux[sort_lam] ** 2) ** 2
    )
    polangle_model = 0.5 * np.arctan2(Umodel[sort_lam], Qmodel[sort_lam])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(lam2_s, polangle, yerr=polangle_sigma,
                linestyle="", marker="o", color='black', label='Data')
    ax.plot(lam2_s, polangle_model,
            color='darkred', linestyle='--', label='Model')
    ax.set_xlabel(r'$\lambda^2$ [m$^2$]')
    ax.set_ylabel('Polarisation angle [rad]')
    ax.legend()
    plt.tight_layout()
    plt.savefig("polangle.png", dpi=150)
    plt.close()

    # --- Plot Polarization Percentage ---
    P_plot = (Qflux + 1j * Uflux) / Iflux
    P_amp = np.abs(P_plot)
    sigma_P = np.sqrt((sigma_Q ** 2 + sigma_U ** 2) / Iflux ** 2)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(freqvec_MHz, 100 * P_amp, yerr=100 * sigma_P,
                linestyle="", marker="s", color='black', label='Polarisation fraction')
    ax.plot(freqvec_MHz,
            100 * np.sqrt(Umodel ** 2 + Qmodel ** 2) / Imodel,
            color='darkred', linestyle='--', label='Model')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Polarisation percentage [%]')
    ax.legend()
    plt.tight_layout()
    plt.savefig("polfrac.png", dpi=150)
    plt.close()

    ####################

    return fitQU_depol[1], fitQU_depol[2], lambdaref2


def get_phase_rot(RM, chi0, input_ms=None, ref_RM=None, ref_chi0=None, lambdaref2=4.5):
    """
    Get polarization phase rotation solution file.

    Args
    ----------
    RM : float
        Rotation measure of the source [rad/m^2].
    chi0 : float
        Intrinsic polarization angle of the source [rad].
    input_ms : str, optional
        Path to the input MeasurementSet.
    ref_RM : float
        Reference rotation measure to subtract [rad/m^2].
    ref_chi0 : float
        Reference polarization angle to subtract [rad].
    lambdaref2 : float, optional
        Reference wavelength squared [m^2]. Default is 4.5.
    """

    delta_RM = RM - ref_RM
    delta_chi0 = chi0 - ref_chi0
    print(RM - ref_RM)
    print(chi0 - ref_chi0)

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('-----INPUT pol_phase_rot.py------')
    print('RM:       ', 2. * delta_RM)
    print('Chi0:', 2. * (delta_chi0 - (delta_RM * lambdaref2)))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    if input_ms is not None:
        h5_out = 'polrot.h5'
        phase_rot(ms_in=input_ms,
                  h5_out=h5_out,
                  intercept=2. * (delta_chi0 - (delta_RM * lambdaref2)),
                  rm=2. * delta_RM)


def parse_args():
    """Argument parser"""

    parser = ArgumentParser(description='Perform polarisation alignment for deep observations')
    parser.add_argument('--input_directory', help='Directory with Stokes Q and U images', type=str, default='./')
    parser.add_argument('--output_directory', help='Output image directory', type=str, default='./')
    parser.add_argument('--region', help='DS9 region file on linear polarised signal', type=str, required=True)

    # For getting alignment h5parm
    parser.add_argument('--msin', help='Input MeasurementSet with the desired time and frequency axis', type=str)
    parser.add_argument('--ref_RM', help='Reference RM for alignment', type=float)
    parser.add_argument('--ref_chi0', help='Reference intercept for polarisation angle', type=float)

    return parser.parse_args()


def main():
    args = parse_args()

    # Get input images
    i_fits = glob(args.input_directory+"/*-0???-I-image.fits")
    u_fits = glob(args.input_directory+"/*-0???-U-image.fits")
    q_fits = glob(args.input_directory+"/*-0???-Q-image.fits")

    RM, chi0, lambdaref2 = fit_RM(i_fits, u_fits, q_fits, args.region)

    json_filename = "rm_alignment.json"
    with open(json_filename, 'w') as file:
        json.dump({"RM": RM, "chi0": chi0}, file, indent=4)

    if args.ref_RM is not None and args.ref_chi0 is not None and args.msin is not None:
        get_phase_rot(RM, chi0, args.msin, args.ref_RM, args.ref_chi0, lambdaref2)

if __name__ == "__main__":
    main()