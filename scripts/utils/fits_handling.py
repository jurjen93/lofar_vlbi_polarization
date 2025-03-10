from astropy.io import fits
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


def make_image_cube(Ifits_files, return_noise=False):
    """
    Make image cube
    Args:
        Ifits_files: Stokes I fits files
        noise: Return noise array

    Returns: Cube, noise_array (optional)
    """

    with fits.open(Ifits_files[0]) as hdul:
        template = np.squeeze(hdul[0].data)

    cube = np.zeros((len(Ifits_files), template.shape[0], template.shape[0]))  # freq axis first (RMtools wants this)
    noise_array = np.zeros((len(Ifits_files)))
    print(cube.shape, noise_array.shape)

    for image_idx, image in enumerate(Ifits_files):
        print(image)
        with fits.open(image) as hdul:
            cube[image_idx, :, :] = np.squeeze(hdul[0].data)
            if return_noise:
                noise_array[image_idx] = findrms(np.squeeze(hdul[0].data))
    if return_noise:
        return cube, noise_array
    else:
        return cube, None
