import numpy as np
from astropy.io import fits
from astropy import units as u
import pyregion
from fits_handling import flatten, make_freq_vec, make_noise_vec



def findrms(mIn, maskSup=1e-7):
    """
    find the rms of an array, from Cycil Tasse/kMS
    """
    try:
        m = mIn[np.abs(mIn) > maskSup]
        rmsold = np.std(m)
        diff = 1e-1
        cut = 3.
        bins = np.arange(np.min(m), np.max(m), (np.max(m) - np.min(m)) / 30.)
        med = np.median(m)
        for i in range(10):
            ind = np.where(np.abs(m - med) < rmsold * cut)[0]
            rms = np.std(m[ind])
            if np.abs((rms - rmsold) / rmsold) < diff: break
            rmsold = rms
        return rms
    except ValueError:
        return np.nan


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
        # print('number of pixels', np.sum(image))
        n_beam = np.sum(image) / ((header['BMAJ'] / header['CDELT2']) * (header['BMIN'] / header['CDELT2']) * np.pi / 4.)
        print('n_beam', n_beam)
        return n_beam


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

    if ds9region is not None:
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
    Iflux = Iflux[~np.isnan(Iflux)]

    sigma_I = sigma_I * np.sqrt(n_beams)
    sigma_Q = sigma_Q * np.sqrt(n_beams)
    sigma_U = sigma_U * np.sqrt(n_beams)

    return freqvec, Iflux, Qflux, Uflux, sigma_I, sigma_Q, sigma_U