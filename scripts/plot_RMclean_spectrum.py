from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pyregion
from astropy.io import fits

from utils.fits_handling import flatten
from utils.image_handling import clipped_median

plt.rcParams.update({
    "font.family": "Serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12
})


def extract_rm_spectrum(fitscube, ds9_regionfile):
    """
    Extract and sum the Faraday depth (RM) spectrum within a DS9-defined
    region from an RM synthesis FITS cube.

    Parameters
    ----------
    fitscube : str
        Path to the RM synthesis FITS cube.

    ds9_regionfile : str
        Path to a DS9 region file defining the source region from which
        the integrated RM spectrum will be extracted.

    Returns
    -------
    rm_spectra_I : numpy.ndarray
        Integrated RM spectrum obtained by summing the spectra of all
        pixels within the masked region. Units are Jy.

    rm_axis : numpy.ndarray
        Faraday depth axis (rad m^-2) corresponding to the extracted
        RM spectrum.
    """
    hdu = fits.open(fitscube)
    hduflat = flatten(hdu)
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
    hdu[0].data *= conversion_factor  # in Jy per pixel
    print(conversion_factor)
    # Get the pixel coordinates of the masked region
    y_indices, x_indices = np.where(manualmask)
    print(f"Number of pixels in masked region: {len(x_indices)}")

    print(hdu[0].data.shape)

    rm_spectra_I = np.zeros(hdu[0].data.shape[1])
    count = 0

    for y, x in zip(y_indices, x_indices):
        # count how many pixels are being added
        count += 1
        rm_spectrum_I = hdu[0].data[0, :, y, x]
        rm_spectra_I = rm_spectra_I + rm_spectrum_I

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

    plt.figure(figsize=(6, 4))
    legend = []
    colors = ['darkgreen', 'darkred', 'darkblue','orange','gray']
    linestyles = ['dashed', 'dotted', '-','-.','--']

    for i in range(len(region_files)):
        region_file = region_files[i]
        rm_spectra, rm_axis = extract_rm_spectrum(FDF_clean_tot, region_file)

        # Subtract median and convert to mJy
        flux = 1e3 * (rm_spectra - clipped_median(rm_spectra))
        plt.plot(rm_axis, flux, color=colors[i], linestyle=linestyles[i])
        legend.append(region_file.replace('.reg', '').replace("_", " ").title())
        print(f"RM peak --> {region_file.replace('.reg', '').replace('_', ' ').title()}: {rm_axis[np.argmax(flux)]}")

    plt.xlim(min(rm_axis), max(rm_axis))
    plt.xlabel('Faraday depth (rad m$^{-2}$)')
    plt.ylabel('FDF (mJy  RMSF$^{-1}$)')
    if len(region_files) > 1:
        plt.legend()
        plt.legend(legend)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    """Argument parser"""

    parser = ArgumentParser(description='Make RM clean spectrum from RM-synthesis and RM-clean output.')
    parser.add_argument('--rm_clean_fits', help='RM-synthesis total clean FITS file', type=str, required=True)
    parser.add_argument('--region', help='DS9 region file (up to 5 region files at once)', nargs='+')
    parser.add_argument('--spectrum_png_name', help='PNG name for output spectrum', type=str, default='RMclean_spectrum.png')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.region is None:
        region_files = glob("*.reg")[0:5]
    else:
        region_files = args.region

    plot_RMclean_spectrum(region_files,
                          args.rm_clean_fits,
                          'RMclean_spectrum_tot.png')


if __name__ == "__main__":
    main()