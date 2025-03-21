from astropy.io import fits
import numpy as np
from .image_handling import findrms
from astropy.wcs import WCS


def get_header(fits_file):
    """
    Get FITS header
    Args:
        fits_file: FITS file name

    Returns: FITS header
    """

    with fits.open(fits_file) as hdul:
        return hdul[0].header


def make_freq_vec(Ifits_files):
    """
    Make frequency vector
    Args:
        Ifits_files: Stokes I fits files

    Returns: frequency vector
    """

    print('Number of freqs', len(Ifits_files))
    freqvec = np.zeros((len(Ifits_files)))
    for image_idx, image in enumerate(Ifits_files):
        with fits.open(image) as hdul:
            header = hdul[0].header
            freqvec[image_idx] = header['CRVAL3']
    print('Frequencies:', freqvec)
    return freqvec


import numpy as np
from astropy.io import fits


def make_image_cube(Ifits_files, return_noise=False):
    """
    Make image cube
    Args:
        Ifits_files: List of Stokes I FITS files
        return_noise: Whether to return noise array

    Returns: Cube, noise_array (optional)
    """

    with fits.open(Ifits_files[0]) as hdul:
        template = np.squeeze(hdul[0].data)
        if template.size == 0:
            raise ValueError(f"Error: First FITS file {Ifits_files[0]} contains no data.")

    cube = np.zeros((len(Ifits_files), *template.shape))
    noise_array = np.zeros(len(Ifits_files))

    for image_idx, image in enumerate(Ifits_files):
        with fits.open(image) as hdul:
            data = np.squeeze(hdul[0].data)
            if data.size == 0:
                raise ValueError(f"Error: FITS file {image} contains no data.")
            cube[image_idx, :, :] = data
            if return_noise:
                noise_array[image_idx] = findrms(data)

    return (cube, noise_array) if return_noise else (cube, None)


def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

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


def make_noise_vec(Ifits_files):
    """
    Make noise vector

    Args:
        Ifits_files: Stokes I fits files

    Returns: Noise array

    """
    noise_array = np.zeros((len(Ifits_files)))
    for image_idx, image in enumerate(Ifits_files):
        print(image)
        hdul = fits.open(image)
        if np.isfinite(np.mean(np.squeeze(hdul[0].data))):
            noise_array[image_idx] = findrms(np.squeeze(hdul[0].data))
        else:
            noise_array[image_idx] = np.nan
        hdul.close()
    return noise_array
