import sys
import csv
from argparse import ArgumentParser
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd

from utils.image_handling import getallfluxes
from utils.RM_functions import functionRM, functionRMdepol, function_synch_simple, make_P
from utils.parsing import extract_l_number
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


def find_RMandoffets(i_fits: list = None, u_fits: list = None, q_fits: list = None, regionfile: str = None):
    """
    Find Rotation Measure offsets for combining multiple observations for deep imaging

    Args:
        i_fits: Stokes I FITS channel images
        u_fits: Stokes U FITS channel images
        q_fits: Stokes Q FITS channel images
        regionfile: Region file
        h5_in: h5parm input
        ref_RM: reference RM
        ref_offset: reference offset

    Return: RM, offset, lambda ref, L-number
    """

    # Sort on name
    i_fits = sorted(i_fits)
    u_fits = sorted(u_fits)
    q_fits = sorted(q_fits)

    # Future update?
    lambdaref2 = 4.5  # (np.mean(wav**2))
    x0_QU = np.array([0.2, 6.0, 0.9])  # initial guess QU fitting
    x0_QU_depol = np.array([0.05, 6.0, 0.9, 0.03])  # initial guess QU fitting with depol
    x0_I = np.array([0.24, -1.1])  # initial guess Stokes I function_synch_simple

    if len(i_fits)==0 and len(u_fits)==0 and len(q_fits)==0:
        sys.exit("ERROR: No images selected/found")

    L = extract_l_number(i_fits[0])

    freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U = getallfluxes(i_fits, q_fits, u_fits, regionfile)

    fitI, pcov_I = scipy.optimize.curve_fit(function_synch_simple, freqvec, Iflux, p0=x0_I, sigma=sigma_I)
    fitI_err = np.sqrt(np.diag(pcov_I))
    print('fitI', fitI)
    print('fitI_err', fitI_err)

    chisq = (Iflux - function_synch_simple(freqvec, fitI[0], fitI[1])) ** 2 / sigma_I ** 2
    idx = np.where(chisq > 2.5 * np.std(chisq))  # use emperical noise on chisq distribution
    idx_incl = np.where(chisq <= 2.5 * np.std(chisq))

    plt.plot(freqvec[idx] / 1e6, Iflux[idx], linestyle="", marker="x", markersize=12., label='removed: bad Stokes I', color='red')
    plt.errorbar(freqvec / 1e6, Iflux, yerr=sigma_I, linestyle="", marker="o", label='Stokes I', color='black')

    # keep only good values, remove flagged ones based on bad chisq Stokes I fit
    freqvec = freqvec[idx_incl]
    Iflux = Iflux[idx_incl]
    Qflux = Qflux[idx_incl]
    Uflux = Uflux[idx_incl]
    sigma_I = sigma_I[idx_incl]
    sigma_Q = sigma_Q[idx_incl]
    sigma_U = sigma_U[idx_incl]

    # Compute polarization fraction
    pol_frac = make_P(Qflux, Uflux, sigma_Q, sigma_U) / Iflux

    # Indices where polarization fraction <= 0.06
    idx_incl = np.where(pol_frac <= 0.06)[0]

    # Create a mask for NaN values in Iflux, Qflux, or Uflux
    nan_mask = ~(np.isnan(Iflux) | np.isnan(Qflux) | np.isnan(Uflux))

    # Combine both conditions
    final_mask = np.intersect1d(idx_incl, np.where(nan_mask)[0])

    # Apply filtering
    freqvec = freqvec[final_mask]
    Iflux = Iflux[final_mask]
    Qflux = Qflux[final_mask]
    Uflux = Uflux[final_mask]
    sigma_I = sigma_I[final_mask]
    sigma_Q = sigma_Q[final_mask]
    sigma_U = sigma_U[final_mask]

    # Sort based on freqvec
    sort_idx = np.argsort(freqvec)

    freqvec = freqvec[sort_idx]
    Iflux = Iflux[sort_idx]
    Qflux = Qflux[sort_idx]
    Uflux = Uflux[sort_idx]
    sigma_I = sigma_I[sort_idx]
    sigma_Q = sigma_Q[sort_idx]
    sigma_U = sigma_U[sort_idx]

    # fit I again but now with cleaned data
    fitI, pcov_I = scipy.optimize.curve_fit(function_synch_simple, freqvec, Iflux, p0=x0_I, sigma=sigma_I)
    # fitI, pcov_I = scipy.optimize.curve_fit(function_synch, freqvec, Iflux, p0=x0_I, sigma=sigma_I)
    fitI_err = np.sqrt(np.diag(pcov_I))
    print('fitI', fitI)
    print('fitI_err', fitI_err)
    Imodel = function_synch_simple(freqvec, fitI[0], fitI[1])

    wav = 299792458. / freqvec
    fitQU, pcov_QU = scipy.optimize.curve_fit(functionRM, np.append(wav, wav), np.append(Qflux, Uflux), p0=x0_QU,
                                              sigma=np.append(sigma_Q, sigma_U))
    fitQU_err = np.sqrt(np.diag(pcov_QU))
    print('fitQU', fitQU)
    print('fitQU_err', fitQU_err)

    fitQU_depol, pcov_QU_depol = scipy.optimize.curve_fit(functionRMdepol, np.append(wav, wav),
                                                          np.append(Qflux / Imodel, Uflux / Imodel), p0=x0_QU_depol,
                                                          sigma=np.append(sigma_Q / Imodel, sigma_U / Imodel))
    fitQU_depol_err = np.sqrt(np.diag(pcov_QU_depol))
    print('fitQU_depol', fitQU_depol)
    print('fitQU_depol_err', fitQU_depol_err)

    if fitQU_depol[0] <= 0:
        print('WARNING: negative polarization fraction')
        x0_QU_depol_tmp = np.copy(x0_QU_depol)
        x0_QU_depol_tmp[0] = np.abs(fitQU_depol[0])
        x0_QU_depol_tmp[1] = fitQU_depol[1]
        x0_QU_depol_tmp[2] = x0_QU_depol[2] - 0.5 * np.pi
        x0_QU_depol_tmp[3] = fitQU_depol[3]
        fitQU_depol, pcov_QU_depol = scipy.optimize.curve_fit(functionRMdepol, np.append(wav, wav),
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

    plt.plot(freqvec / 1e6, function_synch_simple(freqvec / 1e6, fitI[0], fitI[1], freq_ref=150.), color='black', label='Stokes I fit')
    plt.errorbar(freqvec / 1e6, Qflux, yerr=sigma_Q, linestyle="", marker="o", label='Stokes Q', color='blue')
    plt.errorbar(freqvec / 1e6, Uflux, yerr=sigma_U, linestyle="", marker="o", label='Stokes U', color='purple')
    plt.plot(freqvec / 1e6, make_P(Qflux, Uflux, sigma_Q, sigma_U), linestyle="", marker="o", label='Stokes P', color='grey')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Flux [Jy]')
    plt.title(L)
    plt.legend()
    plt.savefig(L + '_StokesIQU_freq.png')
    plt.close()

    # --- Plot Stokes I ---
    plt.figure(figsize=(12, 11.25))
    plt.subplot(3, 1, 1)
    plt.errorbar(wav ** 2, Iflux, yerr=sigma_I, linestyle="", marker="s", label='Stokes I', color='black')
    plt.plot(wav ** 2, function_synch_simple(freqvec / 1e6, fitI[0], fitI[1], freq_ref=150.), color='darkred',
             linestyle='--', label='Stokes I fit')
    plt.xlabel(r'$\lambda^2$ [m$^2$]')
    plt.ylabel('Flux [Jy]')
    plt.title('Stokes I')
    # plt.title(L + ' ' + fitstr)
    plt.tight_layout()

    # --- Plot Stokes Q ---
    plt.subplot(3, 1, 2)
    plt.errorbar(wav ** 2, Qflux, yerr=sigma_Q, linestyle="", marker="s", label='Stokes Q', color='black', markersize=5)
    plt.plot(wav ** 2,
        Imodel * fitQU_depol[0] *
        np.cos(2 * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) *
        np.exp(-2 * fitQU_depol[3] * wav ** 4),
        label='Stokes Q fit', color='darkred', linestyle='--')
    plt.xlabel(r'$\lambda^2$ [m$^2$]')
    plt.ylabel('Flux [Jy]')
    plt.title('Stokes Q')

    # --- Plot Stokes U ---
    plt.subplot(3, 1, 3)
    plt.errorbar(wav ** 2, Uflux, yerr=sigma_U, linestyle="", marker="s", label='Stokes U', color='black', markersize=5)
    plt.plot(wav ** 2,
        Imodel * fitQU_depol[0] *
        np.sin(2 * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) *
        np.exp(-2 * fitQU_depol[3] * wav ** 4),
        label='Stokes U fit', color='darkred', linestyle='--')
    plt.xlabel(r'$\lambda^2$ [m$^2$]')
    plt.ylabel('Flux [Jy]')
    plt.title('Stokes U')

    # Finalise plot
    plt.tight_layout()
    plt.savefig(f"{L}_StokesIQU_wav2.png")
    plt.close()

    # --- Plot Polarization Angle ---
    plt.figure(figsize=(8 * 1.5, 6 * 1.25))
    plt.subplot(2, 1, 1)
    polangle_sigma = 0.5 * np.sqrt((((sigma_U ** 2) * (Qflux ** 2)) + ((sigma_Q ** 2) * (Uflux ** 2))) / ((Uflux ** 2 + Qflux ** 2) ** 2))
    polangle = 0.5 * np.arctan2(Uflux, Qflux)
    plt.errorbar(wav ** 2, polangle, yerr=polangle_sigma, linestyle="", marker="o", color='black', label='polarization angle')
    print(fitQU_depol[0], fitQU_depol[1], fitQU_depol[2], fitQU_depol[3])

    Qmodel = Imodel * fitQU_depol[0] * np.cos(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4)
    Umodel = Imodel * fitQU_depol[0] * np.sin(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4)

    plt.plot(wav ** 2, 0.5 * np.arctan2(Umodel, Qmodel), label='model fit', color='darkred', linestyle='--')

    plt.xlabel(r'$\lambda^2$ [m$^2$]')
    plt.ylabel('Polarization angle [rad]')
    plt.ylim(-0.5 * np.pi, 0.5 * np.pi)

    pfracion_sigma_p1 = (sigma_I ** 2) * (Qflux ** 2 + Uflux ** 2) ** 2
    pfracion_sigma_p2 = (Iflux ** 2) * (((sigma_Q ** 2) * (Qflux ** 2)) + ((sigma_U ** 2) * (Uflux ** 2)))
    pfracion_sigma_p3 = (Iflux ** 4) * (Uflux ** 2 + Qflux ** 2)
    pfracion_sigma = np.sqrt((pfracion_sigma_p1 + pfracion_sigma_p2) / pfracion_sigma_p3)
    P = make_P(Qflux, Uflux, sigma_Q, sigma_U) / Iflux

    # --- Plot Polarization Percentage ---
    plt.subplot(2, 1, 2)
    plt.errorbar(freqvec[P > 0] / 1e6, 100. * P[P > 0], yerr=100. * pfracion_sigma[P > 0], linestyle="", marker="s",
                 color='black', label='polarization fraction')
    plt.errorbar(freqvec[P == 0] / 1e6, 100. * P[P == 0], yerr=100. * pfracion_sigma[P == 0], linestyle="", marker="s",
                 color='black', lolims=True)
    plt.plot(freqvec / 1e6, 100 * np.sqrt(Umodel ** 2 + Qmodel ** 2) / Imodel, label='model fit', color='darkred', linestyle='--')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Polarization percentage')
    plt.savefig(L + '_polangle_frac.png')
    plt.ylim(-1, 10)
    plt.tight_layout()

    plt.close()

    # RM , offset, lambda ref, L-number
    print(fitstr)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f"RM: {fitQU_depol[1]}")
    print(f"offset: {fitQU_depol[2]}")
    print(f"$\lambda$ (reference): {lambdaref2}")
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    return fitQU_depol[1], fitQU_depol[2], lambdaref2, L


def get_phase_rot(RM, offset, input_ms=None, ref_RM=None, ref_offset=None, lambdaref2=None, L=None):
    """
    Get phase rotation h5parm
    """

    delta_RM = RM - ref_RM
    delta_offset = offset - ref_offset
    print(RM - ref_RM)
    print(offset - ref_offset)

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('-----INPUT pol_phase_rot.py------')
    print('RM:       ', 2. * delta_RM)
    print('intercept:', 2. * (delta_offset - (delta_RM * lambdaref2)))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    if input_ms is not None:
        h5_out = L.split('_')[0] + '_polrot.h5'
        phase_rot(ms_in=input_ms,
                  h5_out=h5_out,
                  intercept=2. * (delta_offset - (delta_RM * lambdaref2)),
                  rm=2. * delta_RM)
    return h5_out


def parse_args():
    """Argument parser"""

    parser = ArgumentParser(description='Perform polarisation alignment for deep observations')
    parser.add_argument('--input_directory', help='Directory with Stokes Q and U images', type=str, default='./')
    parser.add_argument('--output_directory', help='Output image directory', type=str, default='./')
    parser.add_argument('--region_file', help='DS9 region file', type=str, required=True)
    parser.add_argument('--msin', help='Input h5parm file', type=str)
    parser.add_argument('--ref_RM', help='Reference RM', type=float)
    parser.add_argument('--ref_offset', help='Reference offset', type=float)
    parser.add_argument('--RM_offset_csv', help='Input CSV with RM and offset from reference observation (instead of --ref_offset or --ref_RM)')
    parser.add_argument('--applycal', action='store_true', help='Apply corrections to MS')

    return parser.parse_args()


def main():
    args = parse_args()

    # Get input images
    i_fits = glob(args.input_directory+"/*-0???-I-image.fits")
    u_fits = glob(args.input_directory+"/*-0???-U-image.fits")
    q_fits = glob(args.input_directory+"/*-0???-Q-image.fits")

    RM, offset, lambdaref2, L = find_RMandoffets(i_fits, u_fits, q_fits, args.region_file)

    # File name
    csv_filename = "rm_offset_data.csv"

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["RM", "Offset"])  # Write header
        writer.writerow([RM, offset])  # Write data

    if ((args.ref_RM is not None and args.ref_offset is not None) or args.RM_offset_csv is not None
            and args.msin is not None):
        if args.RM_offset_csv is not None:
            try:
                df = pd.read_csv(args.RM_offset_csv)
            except FileNotFoundError:
                df = pd.read_csv(glob(args.RM_offset_csv)[0])
            ref_RM = df['RM'][0]
            ref_offset = df['Offset'][0]
        else:
            ref_RM = args.ref_RM
            ref_offset = args.ref_offset
        h5_out = get_phase_rot(RM, offset, args.msin, ref_RM, ref_offset, lambdaref2, L)

    if args.applycal:
        from applycal import ApplyCal
        Ac = ApplyCal(msin=args.msin, h5=h5_out, msout='polaligned_'+args.msin.split("/")[-1])
        Ac.print_cmd()
        Ac.run()


if __name__ == "__main__":
    main()