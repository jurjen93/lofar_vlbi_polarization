import matplotlib
import numpy as np
import sys
from glob import glob
from astropy.io import fits
from astropy import units as u
import pyregion
import matplotlib.pyplot as plt
import scipy.optimize
from utils.fits_handling import flatten, make_freq_vec, make_noise_vec
from utils.RM_functions import functionRM, functionRMdepol, function_synch_simple, make_P
from utils.parsing import extract_l_number
from argparse import ArgumentParser
from pol_phase_rot import PhaseRotate

matplotlib.use('QtAgg')


def get_nbeams_region(filename, ds9region):
    """
    Get number of beams in region file
    Args:
        filename: Input file name
        ds9region: ds9 region file

    Returns: Number of beams
    """

    with fits.open(filename) as hdu:
        hduflat = flatten(hdu)
        header = hduflat.header

        r = pyregion.open(ds9region)
        manualmask = r.get_mask(hdu=hduflat)
        image = np.copy(hdu[0].data[0][0])
        image[np.where(manualmask == False)] = 0.0
        image[np.where(manualmask == True)] = 1.0
        # print('number of pixels', np.sum(image))
        n_beam = np.sum(image) / ((header['BMAJ'] / header['CDELT2']) * (header['BMIN'] / header['CDELT2']) * np.pi / 4.)
        print('n_beam', n_beam)
        return n_beam


def getflux(filename, ds9region):
    """
    Gef flux from fits file
    Args:
        filename: Input file name
        ds9region: DS9 region file

    Returns: Flux density
    """

    with fits.open(filename) as hdu:
        hduflat = flatten(hdu)

        header = hduflat.header

        bmaj = header['BMAJ'] * u.deg  # Major axis of the beam in degrees
        bmin = header['BMIN'] * u.deg  # Minor axis of the beam in degrees
        beam_area = (bmaj * bmin * np.pi / (4 * np.log(2))).to(u.steradian)
        cdelt1 = abs(header['CDELT1']) * u.deg  # Pixel size in degrees along axis 1
        cdelt2 = abs(header['CDELT2']) * u.deg  # Pixel size in degrees along axis 2
        pixel_area = (cdelt1 * cdelt2).to(u.steradian)
        conversion_factor = beam_area / pixel_area

        r = pyregion.open(ds9region)
        manualmask = r.get_mask(hdu=hduflat)
        hdu[0].data[0][0][np.where(manualmask == False)] = 0.0

        return np.sum(hdu[0].data) / conversion_factor


def getallfluxes(Ifiles, Qfiles, Ufiles, ds9region):
    """
    Gef fluxes from Stokes files
    Args:
        Ifiles: Stokes I file names
        Qfiles: Stokes Q file names
        Ufiles: Stokes U file names
        ds9region: DS9 region file

    Returns: Frequency vector, Stokes I flux, Stokes Q flux, Stokes U flux, Sigma Stokes I, Sigma Stokes Q, Sigma Stokes U
    """

    n_beams = get_nbeams_region(Qfiles[0], ds9region)
    freqvec = make_freq_vec(Qfiles)
    Iflux = np.zeros((len(Qfiles)))
    Qflux = np.zeros((len(Qfiles)))
    Uflux = np.zeros((len(Qfiles)))

    sigma_I = make_noise_vec(Ifiles)
    sigma_Q = make_noise_vec(Qfiles)
    sigma_U = make_noise_vec(Ufiles)

    for image_idx, image in enumerate(Ifiles):
        Iflux[image_idx] = getflux(image, ds9region)
        print('I Flux:', Iflux[image_idx])
    for image_idx, image in enumerate(Qfiles):
        Qflux[image_idx] = getflux(image, ds9region)
        print('Q Flux:', Qflux[image_idx])
    for image_idx, image in enumerate(Ufiles):
        Uflux[image_idx] = getflux(image, ds9region)
        print('U Flux:', Uflux[image_idx])

    freqvec = freqvec[~np.isnan(Iflux)]
    Qflux = Qflux[~np.isnan(Iflux)]
    Uflux = Uflux[~np.isnan(Iflux)]

    sigma_I = sigma_I[~np.isnan(Iflux)]
    sigma_Q = sigma_Q[~np.isnan(Iflux)]
    sigma_U = sigma_U[~np.isnan(Iflux)]
    Iflux = Iflux[~np.isnan(Iflux)]  # do this last, otherwise previous steps go wrong

    sigma_I = sigma_I * np.sqrt(n_beams)
    sigma_Q = sigma_Q * np.sqrt(n_beams)
    sigma_U = sigma_U * np.sqrt(n_beams)

    return freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U


def phase_rot(h5_in, h5_out, intercept, rm):
    """
    Make template h5 with default values (phase=0 and amplitude=1) and convert to phase rotate matrix
    for polarization alignment between different observations.
    Args:
        h5_in: Input h5parm
        h5_out: Output h5parm
        intercept: Intercept value
        rm: RM value
    """

    phaserot = PhaseRotate(h5_in, h5_out)
    phaserot.make_template(polrot=True)
    phaserot.rotate(intercept=intercept, rotation_measure=rm)
    phaserot.h5.close()


def find_RMandoffets(i_fits, u_fits, q_fits, regionfile, h5_in=None, ref_RM=6.30423201, ref_offset=0.8701224377970226):
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
    """

    if len(i_fits)==0 and len(u_fits)==0 and len(q_fits)==0:
        sys.exit("ERROR: No images selected/found")

    L = extract_l_number(i_fits[0])

    freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U = getallfluxes(i_fits, q_fits, u_fits, regionfile)

    lambdaref2 = 4.5  # (np.mean(wav**2))
    x0_QU = np.array([0.2, 6.0, 0.9])  # initial guess QU fitting
    x0_QU_depol = np.array([0.05, 6.0, 0.9, 0.03])  # initial guess QU fitting with depol
    x0_I = np.array([0.24, -1.1])  # initial guess Stokes I function_synch_simple

    fitI, pcov_I = scipy.optimize.curve_fit(function_synch_simple, freqvec, Iflux, p0=x0_I, sigma=sigma_I)
    fitI_err = np.sqrt(np.diag(pcov_I))
    print('fitI', fitI)
    print('fitI_err', fitI_err)

    chisq = (Iflux - function_synch_simple(freqvec, fitI[0], fitI[1])) ** 2 / sigma_I ** 2
    idx = np.where(chisq > 2.5 * np.std(chisq))  # use emperical noise on chisq distribution
    idx_incl = np.where(chisq <= 2.5 * np.std(chisq))

    plt.plot(freqvec[idx] / 1e6, Iflux[idx], linestyle="", marker="x", markersize=12., label='removed: bad Stokes I',
             color='red')
    plt.errorbar(freqvec / 1e6, Iflux, yerr=sigma_I, linestyle="", marker="o", label='Stokes I', color='black')

    # keep only good values, remove flagged ones based on bad chisq Stokes I fit
    freqvec = freqvec[idx_incl]
    Iflux = Iflux[idx_incl]
    Qflux = Qflux[idx_incl]
    Uflux = Uflux[idx_incl]
    sigma_I = sigma_I[idx_incl]
    sigma_Q = sigma_Q[idx_incl]
    sigma_U = sigma_U[idx_incl]

    idx = np.where(make_P(Qflux, Uflux, sigma_Q, sigma_U) / Iflux > 0.06)  # filter out pol fraction
    idx_incl = np.where(
        make_P(Qflux, Uflux, sigma_Q, sigma_U) / Iflux <= 0.06)  # filter out pol fraction larger than 5%
    # keep only good values, remove flagged ones based on bad chisq Stokes I fit
    freqvec = freqvec[idx_incl]
    Iflux = Iflux[idx_incl]
    Qflux = Qflux[idx_incl]
    Uflux = Uflux[idx_incl]
    sigma_I = sigma_I[idx_incl]
    sigma_Q = sigma_Q[idx_incl]
    sigma_U = sigma_U[idx_incl]

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

    fitstr = 'fit:  RM=' + str(round(fitQU_depol[1], 3)) + '$\pm$' + str(
        round(fitQU_depol_err[1], 3)) + ' [rad m$^{-2}$];  $\chi_{ref}=$' + str(
        round(fitQU_depol[2], 3)) + '$\pm$' + str(round(fitQU_depol_err[2], 3)) + ' [rad];\n $\sigma_{RM}^{2}=$' + str(
        round(fitQU_depol[3], 3)) + '$\pm$' + str(
        round(fitQU_depol_err[3], 3)) + ' [rad$^{2}$ m$^{-4}$];  $p_0=$' + str(
        round(fitQU_depol[0], 3)) + '$\pm$' + str(round(fitQU_depol_err[0], 3))

    plt.plot(freqvec / 1e6, function_synch_simple(freqvec / 1e6, fitI[0], fitI[1], freq_ref=150.), color='black',
             label='Stokes I fit')
    plt.errorbar(freqvec / 1e6, Qflux, yerr=sigma_Q, linestyle="", marker="o", label='Stokes Q', color='blue')
    plt.errorbar(freqvec / 1e6, Uflux, yerr=sigma_U, linestyle="", marker="o", label='Stokes U', color='purple')
    plt.plot(freqvec / 1e6, make_P(Qflux, Uflux, sigma_Q, sigma_U), linestyle="", marker="o", label='Stokes P',
             color='grey')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Flux [Jy]')
    plt.title(L)
    plt.legend()
    plt.savefig(L + '_StokesIQU_freq.png')
    plt.close()

    plt.figure(figsize=(8 * 1.5, 6 * 1.25))
    plt.subplot(2, 1, 1)
    plt.errorbar(wav ** 2, Iflux, yerr=sigma_I, linestyle="", marker="o", label='Stokes I', color='black')
    plt.plot(wav ** 2, function_synch_simple(freqvec / 1e6, fitI[0], fitI[1], freq_ref=150.), color='black',
             label='Stokes I fit')
    plt.errorbar(wav ** 2, Qflux, yerr=sigma_Q, linestyle="", marker="o", label='Stokes Q', color='blue')
    plt.errorbar(wav ** 2, Uflux, yerr=sigma_U, linestyle="", marker="o", label='Stokes U', color='purple')

    # RM only
    # plt.plot(wav**2, fitQU[0]*np.cos(2.*(fitQU[1]*(wav**2-lambdaref2) + fitQU[2])), label='Stokes U fit', color='blue')
    # plt.plot(wav**2, fitQU[0]*np.sin(2.*(fitQU[1]*(wav**2-lambdaref2) + fitQU[2])), label='Stokes Q fit', color='purple')
    # with depol
    plt.plot(wav ** 2, Imodel * fitQU_depol[0] * np.cos(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4),
             label='Stokes Q fit', color='blue')
    plt.plot(wav ** 2, Imodel * fitQU_depol[0] * np.sin(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4),
             label='Stokes U fit', color='purple')

    # plt.plot(wav**2, np.sqrt(Uflux**2 + Qflux**2), linestyle="",marker="o", label='Stokes P')
    plt.xlabel('$\lambda^2$ [m$^2$]')
    plt.ylabel('Flux [Jy]')
    plt.legend(loc='upper right')
    plt.title(L + ' ' + fitstr)

    plt.subplot(2, 1, 2)
    plt.errorbar(wav ** 2, Qflux, yerr=sigma_Q, linestyle="", marker="o", label='Stokes Q', color='blue')
    plt.errorbar(wav ** 2, Uflux, yerr=sigma_U, linestyle="", marker="o", label='Stokes U', color='purple')
    plt.plot(wav ** 2, Imodel * fitQU_depol[0] * np.cos(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4),
             label='Stokes Q fit', color='blue')
    plt.plot(wav ** 2, Imodel * fitQU_depol[0] * np.sin(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4),
             label='Stokes U fit', color='purple')
    plt.xlabel('$\lambda^2$ [m$^2$]')
    plt.ylabel('Flux [Jy]')
    plt.legend(loc='upper right')

    plt.savefig(L + '_StokesIQU_wav2.png')
    plt.tight_layout()
    plt.close()

    plt.figure(figsize=(8 * 1.5, 6 * 1.25))
    plt.subplot(2, 1, 1)
    polangle_sigma = 0.5 * np.sqrt((((sigma_U ** 2) * (Qflux ** 2)) + ((sigma_Q ** 2) * (Uflux ** 2))) / (
                (Uflux ** 2 + Qflux ** 2) ** 2))  # https://astro.subhashbose.com/tools/error-propagation-calculator
    polangle = 0.5 * np.arctan2(Uflux, Qflux)
    plt.errorbar(wav ** 2, polangle, yerr=polangle_sigma, linestyle="", marker="o", color='black',
                 label='polarization angle')
    print(fitQU_depol[0], fitQU_depol[1], fitQU_depol[2], fitQU_depol[3])

    Qmodel = Imodel * fitQU_depol[0] * np.cos(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4)
    Umodel = Imodel * fitQU_depol[0] * np.sin(
        2. * (fitQU_depol[1] * (wav ** 2 - lambdaref2) + fitQU_depol[2])) * np.exp(-2. * fitQU_depol[3] * wav ** 4)

    plt.plot(wav ** 2, 0.5 * np.arctan2(Umodel, Qmodel), label='model fit', color='black')

    plt.xlabel('$\lambda^2$ [m$^2$]')
    plt.ylabel('Polarization angle [rad]')
    plt.title(L + ' ' + fitstr)
    # plt.xlim(3.2, 6.85)
    plt.ylim(-0.5 * np.pi, 0.5 * np.pi)
    plt.legend(loc='upper right')

    pfracion_sigma_p1 = (sigma_I ** 2) * (Qflux ** 2 + Uflux ** 2) ** 2
    pfracion_sigma_p2 = (Iflux ** 2) * (((sigma_Q ** 2) * (Qflux ** 2)) + ((sigma_U ** 2) * (Uflux ** 2)))
    pfracion_sigma_p3 = (Iflux ** 4) * (Uflux ** 2 + Qflux ** 2)
    pfracion_sigma = np.sqrt((pfracion_sigma_p1 + pfracion_sigma_p2) / pfracion_sigma_p3)
    P = make_P(Qflux, Uflux, sigma_Q, sigma_U) / Iflux

    plt.subplot(2, 1, 2)
    plt.errorbar(freqvec[P > 0] / 1e6, 100. * P[P > 0], yerr=100. * pfracion_sigma[P > 0], linestyle="", marker="o",
                 color='black', label='polarization fraction')
    plt.errorbar(freqvec[P == 0] / 1e6, 100. * P[P == 0], yerr=100. * pfracion_sigma[P == 0], linestyle="", marker="o",
                 color='black', lolims=True)
    plt.plot(freqvec / 1e6, 100 * np.sqrt(Umodel ** 2 + Qmodel ** 2) / Imodel, label='model fit', color='black')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Polarization percentage')
    plt.legend(loc='upper right')
    # plt.title(L + ' ' + fitstr)
    plt.savefig(L + '_polangle_frac.png')
    plt.ylim(-1, 10)
    plt.tight_layout()

    plt.close()

    delta_RM = fitQU_depol[1] - ref_RM
    delta_offset = fitQU_depol[2] - ref_offset
    print(fitQU_depol[1] - ref_RM)
    print(fitQU_depol[2] - ref_offset)

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('-----INPUT pol_phase_rot.py------')
    print('RM:       ', 2. * delta_RM)
    print('intercept:', 2. * (delta_offset - (delta_RM * lambdaref2)))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    if h5_in is not None:
        h5_out = L.split('_')[0] + '_polrot.h5'
        phase_rot(h5_in, h5_out, 2. * (delta_offset - (delta_RM * lambdaref2)), 2. * delta_RM)
    return


def parse_args():
    """Argument parser"""

    parser = ArgumentParser(description='Perform polarisation alignment for deep observations')
    parser.add_argument('--input_directory', help='Directory with Stokes Q and U images', type=str, default='./')
    parser.add_argument('--output_directory', help='Output image directory', type=str, default='./')
    parser.add_argument('--region_file', help='DS9 region file', type=str, required=True)
    parser.add_argument('--input_h5', help='Input h5parm file', type=str)
    parser.add_argument('--ref_RM', help='Reference RM', type=float, default=6.30423201)
    parser.add_argument('--ref_offset', help='Reference offset', type=float, default=0.8701224377970226)
    return parser.parse_args()


def main():
    args = parse_args()

    # Get input images
    i_fits = glob(args.input_directory+"/*-0???-I-image.fits")
    u_fits = glob(args.input_directory+"/*-0???-U-image.fits")
    q_fits = glob(args.input_directory+"/*-0???-Q-image.fits")

    find_RMandoffets(i_fits, u_fits, q_fits, args.region_file, args.input_h5, args.ref_RM, args.ref_offset)


if __name__ == "__main__":
    main()