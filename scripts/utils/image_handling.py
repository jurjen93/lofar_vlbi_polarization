import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.constants import c
import pyregion

from .fits_handling import flatten, make_freq_vec, make_noise_vec

import warnings

warnings.filterwarnings("ignore")


def clipped_median(data, sigma_clip=3.0, max_iter=5):
    """
    Compute the sigma-clipped median of a one-dimensional array.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional array containing the data values.

    sigma_clip : float, optional
        Clipping threshold in units of the standard deviation.
    max_iter : int, optional
        Maximum number of sigma-clipping iterations. Default is 5.

    Returns
    -------
    float
        Sigma-clipped median of the input data.
    """
    data = data[np.isfinite(data)]  # Remove NaN and Inf values
    for _ in range(max_iter):
        median = np.median(data)
        std_dev = np.std(data)
        mask = np.abs(data - median) < sigma_clip * std_dev
        new_data = data[mask]
        if len(new_data) == len(data):
            break
        data = new_data
    return np.median(data)


def get_nbeams_region(filename: str = None, ds9region: str = None):
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
        n_beam = np.sum(image) / ((header['BMAJ'] / header['CDELT2']) * (header['BMIN'] / header['CDELT2']) * np.pi / 4.)
        print('n_beam', n_beam)
        return n_beam


def get_lambda2(fitsfiles):
    """
    Get lambda^2 from input FITS files

    Args:
        fitsfiles: FITS files from channel images

    Returns: lambda^2
    """
    freqs = []
    for fitsfile in fitsfiles:
        with fits.open(fitsfile) as f:
            freqs.append(f[0].header["CRVAL3"])
    return (c.value/np.median(freqs))**2


def getflux(filename: str = None, ds9region: str = None):
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


def getallfluxes(Ifiles: list = None, Qfiles: list = None, Ufiles: list = None, ds9region: str = None):
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
        print('I Flux -->', image, Iflux[image_idx])
    for image_idx, image in enumerate(Qfiles):
        Qflux[image_idx] = getflux(image, ds9region)
        print('Q Flux -->', image, Qflux[image_idx])
    for image_idx, image in enumerate(Ufiles):
        Uflux[image_idx] = getflux(image, ds9region)
        print('U Flux -->', image, Uflux[image_idx])

    mask = ~np.isnan(Iflux)
    freqvec = np.array(freqvec)[mask]
    Qflux = Qflux[mask]
    Uflux = Uflux[mask]
    sigma_I = sigma_I[mask]
    sigma_Q = sigma_Q[mask]
    sigma_U = sigma_U[mask]
    Iflux = Iflux[mask]

    sigma_I = sigma_I * np.sqrt(n_beams)
    sigma_Q = sigma_Q * np.sqrt(n_beams)
    sigma_U = sigma_U * np.sqrt(n_beams)

    return freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U