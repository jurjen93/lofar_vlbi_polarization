from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from .image_handling import findrms


def get_header(fits_file):
    """
    Get FITS header
    Args:
        fits_file: FITS file name

    Returns: FITS header
    """

    with fits.open(fits_file) as hdul:
        return hdul[0].header


def make_freq_vec(fits_files):
    """
    Make frequency vector
    Args:
        Ifits_files: Stokes I fits files

    Returns: frequency vector
    """

    print('Number of freqs', len(fits_files))
    freqvec = np.zeros((len(fits_files)))
    for image_idx, image in enumerate(fits_files):
        with fits.open(image) as hdul:
            header = hdul[0].header
            freqvec[image_idx] = header['CRVAL3']
    print('Frequencies:', freqvec)
    return freqvec


def make_image_cube(fits_files, return_noise=False):
    """
    Make image cube
    Args:
        fits_files: FITS files
        noise: Return noise array

    Returns: Cube, noise_array (optional)
    """

    with fits.open(fits_files[0]) as hdul:
        template = np.squeeze(hdul[0].data)

    cube = np.zeros((len(fits_files), template.shape[0], template.shape[0]))  # freq axis first (RMtools wants this)
    noise_array = np.zeros((len(fits_files)))
    print(cube.shape, noise_array.shape)

    for image_idx, image in enumerate(fits_files):
        print(image)
        with fits.open(image) as hdul:
            cube[image_idx, :, :] = np.squeeze(hdul[0].data)
            if return_noise:
                noise_array[image_idx] = findrms(np.squeeze(hdul[0].data))
    if return_noise:
        return cube, noise_array
    else:
        return cube, None


def flatten(f):
    """
    Flatten a FITS image cube to a two-dimensional image.

    Parameters
    ----------
    f : astropy.io.fits.HDUList
        Open FITS file containing the image data to be flattened.

    Returns
    -------
    astropy.io.fits.PrimaryHDU
        A new 2D FITS primary HDU containing the flattened image and a
        corresponding 2D WCS header.
    """

    naxis = f[0].header['NAXIS']
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header, data=f[0].data)

    w = WCS(f[0].header)
    wn = WCS(naxis=2)

    wn.wcs.crpix[0] = w.wcs.crpix[0]
    wn.wcs.crpix[1] = w.wcs.crpix[1]
    wn.wcs.cdelt = w.wcs.cdelt[0:2]
    wn.wcs.crval = w.wcs.crval[0:2]
    wn.wcs.ctype[0] = w.wcs.ctype[0]
    wn.wcs.ctype[1] = w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"] = 2
    copy = ('EQUINOX', 'EPOCH', 'BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r = f[0].header.get(k)
        if r is not None:
            header[k] = r

    slice = []
    for i in range(naxis, 0, -1):
        if i <= 2:
            slice.append(np.s_[:], )
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header, data=f[0].data[tuple(slice)])
    return hdu


def make_noise_vec(fits_files):
    """
    Make noise vector

    Args:
        fits_files: FITS files

    Returns: Noise array

    """
    noise_array = np.zeros((len(fits_files)))
    for image_idx, image in enumerate(fits_files):
        print(image)
        hdul = fits.open(image)
        if np.isfinite(np.mean(np.squeeze(hdul[0].data))):
            noise_array[image_idx] = findrms(np.squeeze(hdul[0].data))
        else:
            noise_array[image_idx] = np.nan
        hdul.close()
    return noise_array


def remove_bad_fits(fits_files):
    """
    Filter a list of FITS files, keeping only those whose primary image
    data contain finite values.

    Parameters
    ----------
    fits_files : list of str
        List of paths to FITS files to validate.

    Returns
    -------
    list of str
        List of FITS file paths that contain only finite values in the
        primary HDU data.
    """
    goodfits = []

    for fitsfile in fits_files:
        try:
            with fits.open(fitsfile, memmap=True) as hdul:
                if np.isfinite(hdul[0].data).all():
                    goodfits.append(fitsfile)
                else:
                    print(f"Removing bad FITS file: {fitsfile}")
        except Exception as e:
            print(f"Removing unreadable FITS file: {fitsfile} ({e})")
