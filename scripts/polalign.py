import sys
from argparse import ArgumentParser
from glob import glob
import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from astropy.constants import c

from utils.image_handling import getallfluxes, get_lambda2
from utils.RM_functions import functionRM, functionRMdepol, function_synch_simple, make_P
from pol_phase_rot import PhaseRotate

# Define your desired font
font_name = "Serif"  # Change this to your desired font name

# Update Matplotlib font configuration
plt.rcParams['font.family'] = font_name


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

    # Sort on name
    i_fits = sorted(i_fits)
    u_fits = sorted(u_fits)
    q_fits = sorted(q_fits)

    # Future update?
    lambdaref2 = get_lambda2(i_fits)
    x0_QU_depol = np.array([0.05, 6.0, 0.9, 0.03])  # initial guess QU fitting with depol
    x0_I = np.array([0.24, -1.1])  # initial guess Stokes I function_synch_simple

    if len(i_fits)==0 and len(u_fits)==0 and len(q_fits)==0:
        sys.exit("ERROR: No images selected/found")

    freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U = getallfluxes(i_fits, q_fits, u_fits, regionfile)
    # Frequency vector in MHz
    freqvec_MHz = freqvec / 1e6

    # Filter bad images
    fitI, pcov_I = scipy.optimize.curve_fit(function_synch_simple, freqvec, Iflux, p0=x0_I, sigma=sigma_I)
    chisq = (Iflux - function_synch_simple(freqvec, *fitI)) ** 2 / sigma_I ** 2
    mask = (chisq <= 2.5 * np.std(chisq)) & np.isfinite(Iflux) & np.isfinite(Qflux) & np.isfinite(Uflux)
    sort_idx = np.argsort(freqvec[mask])
    arrays = [freqvec, freqvec_MHz, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U]
    freqvec, freqvec_MHz, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U = [a[mask][sort_idx] for a in arrays]
    wav = c.value / freqvec

    # fit I
    fitI, pcov_I = scipy.optimize.curve_fit(function_synch_simple,
                                            freqvec, Iflux,
                                            p0=x0_I,
                                            sigma=sigma_I)

    Imodel = function_synch_simple(freqvec, fitI[0], fitI[1])

    # Fit Q and U
    fitQU_depol, pcov_QU_depol = scipy.optimize.curve_fit(functionRMdepol,
                                                          np.append(wav, wav),
                                                          np.append(Qflux / Imodel, Uflux / Imodel),
                                                          p0=x0_QU_depol,
                                                          sigma=np.append(sigma_Q / Imodel, sigma_U / Imodel))

    if fitQU_depol[0] <= 0:
        print('WARNING: negative polarization fraction')
        x0_QU_depol_tmp = np.copy(x0_QU_depol)
        x0_QU_depol_tmp[0] = np.abs(fitQU_depol[0])
        x0_QU_depol_tmp[1] = fitQU_depol[1]
        x0_QU_depol_tmp[2] = x0_QU_depol[2] - 0.5 * np.pi
        x0_QU_depol_tmp[3] = fitQU_depol[3]
        fitQU_depol, pcov_QU_depol = scipy.optimize.curve_fit(functionRMdepol,
                                                              np.append(wav, wav),
                                                              np.append(Qflux / Imodel, Uflux / Imodel),
                                                              p0=x0_QU_depol_tmp,
                                                              sigma=np.append(sigma_Q / Imodel, sigma_U / Imodel))
        fitQU_depol_err = np.sqrt(np.diag(pcov_QU_depol))
        print('fitQU_depol', fitQU_depol)
        print('fitQU_depol_err', fitQU_depol_err)
        if fitQU_depol[0] <= 0:
            print('STUCK: negative polarization fraction')
            sys.exit()

    fitstr = r'fit:  RM=' + str(round(fitQU_depol[1], 3)) + r'$\pm$' + str(
        round(fitQU_depol_err[1], 3)) + r' [rad m$^{-2}$];  $\chi_{ref}=$' + str(
        round(fitQU_depol[2], 3)) + r'$\pm$' + str(round(fitQU_depol_err[2], 3)) + r' [rad];\n $\sigma_{RM}^{2}=$' + str(
        round(fitQU_depol[3], 3)) + r'$\pm$' + str(
        round(fitQU_depol_err[3], 3)) + r' [rad$^{2}$ m$^{-4}$];  $p_0=$' + str(
        round(fitQU_depol[0], 3)) + r'$\pm$' + str(round(fitQU_depol_err[0], 3))

    ##### PLOTTING #####

    # --- Plot Stokes I, Q, U ---
    lam2 = wav ** 2
    pol_model = Imodel * fitQU_depol[0] * np.exp(-2 * fitQU_depol[3] * lam2 ** 2)
    phase = 2 * (fitQU_depol[1] * (lam2 - lambdaref2) + fitQU_depol[2])
    Qmodel = pol_model * np.cos(phase)
    Umodel = pol_model * np.sin(phase)
    panels = [
        ("Stokes I", Iflux, sigma_I, function_synch_simple(freqvec_MHz, *fitI, freq_ref=150.)),
        ("Stokes Q", Qflux, sigma_Q, Qmodel),
        ("Stokes U", Uflux, sigma_U, Umodel),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11.25))
    for ax, (title, flux, sigma, model) in zip(axes, panels):
        ax.errorbar(lam2, flux, yerr=sigma, linestyle="", marker="s", color='black', markersize=5)
        ax.plot(lam2, model, color='darkred', linestyle='--', label=f'{title} fit')
        ax.set_xlabel(r'$\lambda^2$ [m$^2$]')
        ax.set_ylabel('Flux [Jy]')
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"StokesIQU_wav2.png")
    plt.close()

    # --- Plot Polarization Angle ---
    polangle = 0.5 * np.arctan2(Uflux, Qflux)
    polangle_sigma = 0.5 * np.sqrt(
        ((sigma_U ** 2) * (Qflux ** 2) + (sigma_Q ** 2) * (Uflux ** 2)) / (Uflux ** 2 + Qflux ** 2) ** 2)

    fig, ax = plt.subplots(figsize=(8 * 1.5, 6 * 1.25))
    ax.errorbar(lam2, polangle, yerr=polangle_sigma, linestyle="", marker="o", color='black',
                label='Polarization angle')
    ax.plot(lam2, 0.5 * np.arctan2(Umodel, Qmodel), color='darkred', linestyle='--', label='Model fit')
    ax.set_xlabel(r'$\lambda^2$ [m$^2$]')
    ax.set_ylabel('Polarization angle [rad]')
    ax.set_ylim(-0.5 * np.pi, 0.5 * np.pi)
    ax.legend()
    plt.tight_layout()
    plt.savefig('polangle.png')
    plt.close()

    # --- Plot Polarization Percentage ---
    P = make_P(Qflux, Uflux, sigma_Q, sigma_U) / Iflux
    pfracion_sigma = np.sqrt(
        (sigma_I ** 2) * (Qflux ** 2 + Uflux ** 2) ** 2 +
        (Iflux ** 2) * ((sigma_Q ** 2) * (Qflux ** 2) + (sigma_U ** 2) * (Uflux ** 2))
    ) / (Iflux ** 2 * np.sqrt(Uflux ** 2 + Qflux ** 2))

    fig, ax = plt.subplots(figsize=(8 * 1.5, 6 * 1.25))
    ax.errorbar(freqvec[P > 0] / 1e6, 100. * P[P > 0], yerr=100. * pfracion_sigma[P > 0],
                linestyle="", marker="s", color='black', label='Polarization fraction')
    ax.errorbar(freqvec[P == 0] / 1e6, 100. * P[P == 0], yerr=100. * pfracion_sigma[P == 0],
                linestyle="", marker="s", color='black', lolims=True)
    ax.plot(freqvec_MHz, 100 * np.sqrt(Umodel ** 2 + Qmodel ** 2) / Imodel, color='darkred', linestyle='--',
            label='Model fit')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Polarization percentage')
    ax.set_ylim(-1, 10)
    ax.legend()
    plt.tight_layout()
    plt.savefig('polfrac.png')
    plt.close()

    # RM , chi0, lambda ref, L-number
    print(fitstr)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f"RM: {fitQU_depol[1]}")
    print(f"chi0: {fitQU_depol[2]}")
    print(r"$\lambda$ (reference): "+str(round(lambdaref2, 2)))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

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
    parser.add_argument('--region_file', help='DS9 region file on linear polarised signal', type=str, required=True)

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

    RM, chi0, lambdaref2 = fit_RM(i_fits, u_fits, q_fits, args.region_file)

    json_filename = "rm_alignment.json"
    with open(json_filename, 'w') as file:
        json.dump({"RM": RM, "chi0": chi0}, file, indent=4)

    if args.ref_RM is not None and args.ref_chi0 is not None and args.msin is not None:
        get_phase_rot(RM, chi0, args.msin, args.ref_RM, args.ref_chi0, lambdaref2)

if __name__ == "__main__":
    main()