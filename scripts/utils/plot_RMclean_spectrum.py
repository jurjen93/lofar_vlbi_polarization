import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import pyregion
import matplotlib.pyplot as plt
from .fits_handling import flatten
from .image_handling import clipped_median


def extract_RM_spectrum(fitscube, ds9_regionfile):
    hdu = fits.open(fitscube)
    hduflat = flatten(hdu)
    w = WCS(hduflat.header)
    r = pyregion.open(ds9_regionfile)
    manualmask = r.get_mask(hdu=hduflat)

    # convert Jy/beam to Jy per pixel
    bmaj = hdu[0].header['BMAJ']  # in degrees
    bmin = hdu[0].header['BMIN']  # in degrees
    pixscale_x = abs(hdu[0].header['CDELT1'])  # in degrees
    pixscale_y = abs(hdu[0].header['CDELT2'])  # in degrees
    beam_area = (np.pi * bmaj * bmin) / (4 * np.log(2))  # in deg^2
    pixel_area = pixscale_x * pixscale_y  # in deg^2
    conversion_factor = pixel_area / beam_area
    hdu[0].data *= conversion_factor  # now in Jy per pixel

    # Get the pixel coordinates of the masked region
    y_indices, x_indices = np.where(manualmask)
    print(f"Number of pixels in masked region: {len(x_indices)}")

    print(hdu[0].data.shape)

    rm_spectra_I = np.zeros(hdu[0].data.shape[1])
    count = 0
    # Extract the RM spectrum for each pixel in the masked region

    for y, x in zip(y_indices, x_indices):
        # count how many pixels are being added
        count += 1
        rm_spectrum_I = hdu[0].data[0, :, y, x]
        rm_spectra_I = rm_spectra_I + rm_spectrum_I

    print(f"Total number of pixels added: {count}")
    # create RM-axis from FITS header
    delta_RM = hdu[0].header['CDELT3']
    ref_RM = hdu[0].header['CRVAL3']
    ref_pixel = hdu[0].header['CRPIX3']
    num_RM_channels = hdu[0].header['NAXIS3']
    rm_axis = ref_RM + (np.arange(num_RM_channels) + 1 - ref_pixel) * delta_RM
    return rm_spectra_I, rm_axis


def plot_RMclean_spectrum(region_files: list[str],
                          FDF_clean_tot: str,
                          output_file: str = 'RMclean_spectrum.pdf'):
    """
    Plot RM-clean spectra for special sources from multiple DS9 regions.

    Parameters
    ----------
    region_files : list[str]
        DS9 region files.
    FDF_clean_tot : str
        FITS cube file path.
    output_file : str
        Output PDF filename.
    """
    plt.figure(figsize=(10, 6))

    legend = []

    for region_file in region_files:
        rm_spectra, rm_axis = extract_RM_spectrum(FDF_clean_tot, region_file)

        # Subtract median and convert to mJy
        flux = 1e3 * (rm_spectra - clipped_median(rm_spectra))
        plt.plot(rm_axis, flux)
        legend.append(region_file.replace('.reg', ''))

    plt.xlim(-20, 20)
    plt.xlabel('Faraday depth [rad m$^{-2}$]', fontsize=12)
    plt.ylabel('Faraday dispersion function [mJy  RMSF$^{-1}$]', fontsize=12)
    plt.legend(fontsize=12)
    plt.legend(legend)
    plt.savefig(output_file, dpi=144)
    plt.close()

