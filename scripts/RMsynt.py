from argparse import ArgumentParser
from glob import glob
import numpy as np

from RMtools_3D.do_RMsynth_3D import run_rmsynth, writefits
from RMtools_3D.do_RMclean_3D import run_rmclean, writefits as writefits_clean
from utils.fits_handling import get_header, make_freq_vec, make_image_cube

import multiprocessing
pool = multiprocessing.Pool()

def do_RMsynt(i_images: list = None,
              q_images: list = None,
              u_images: list = None,
              output_prefix: str = 'RMsynth',
              dphi: float = None,
              phi_max: float = None,
              clean_threshold: float = None,
              do_rmclean: bool = False,):
    """
    Perform RM synthesis
    Args:
        i_images: Stokes I fits images (Currently not used)
        q_images: Stokes Q fits images
        u_images: Stokes U fits images
        output_prefix: Output prefix
        dphi: delta phi
        phi_max: phi max
        clean_threshold: cleaning threshold in Jy/beam
        do_rmclean: perform RM cleaning
    """

    q_header = get_header(q_images[0])

    # Make frequency array:
    freq_array = make_freq_vec(u_images)

    # Make image cubes
    i_data = i_images # TODO: Add I-model?
    u_data, rms_u = make_image_cube(sorted(u_images), return_noise=True)
    q_data, rms_q = make_image_cube(sorted(q_images), return_noise=True)

    # NaN filtering
    valid = ~np.any(np.isnan(u_data), axis=(1, 2)) & ~np.any(np.isnan(q_data), axis=(1, 2))
    u_data = u_data[valid]
    q_data = q_data[valid]
    rms_u = rms_u[valid]
    rms_q = rms_q[valid]
    freq_array = np.array(freq_array)[valid]

    # Noise vector (take average of Q and U noises)
    rms = 0.5 * (rms_u + rms_q)
    if clean_threshold is None:
        clean_threshold = np.nanmedian(rms)/np.sqrt(len(rms))
        print(f"Calculated clean_threshold: {clean_threshold} Jy/beam")

    data_arr = run_rmsynth(q_data,
                          u_data,
                          freq_array,
                          dataI=i_data,
                          rmsArr=rms,
                          phiMax_radm2=phi_max,
                          dPhi_radm2=dphi,
                          nSamples=None,
                          weightType="variance",
                          fitRMSF=False,
                          nBits=32,
                          verbose=True,
                          not_rmsf=False)

    writefits(data_arr,
              headtemplate=q_header,
              fitRMSF=False,
              prefixOut=output_prefix,
              outDir='./',
              write_seperate_FDF=True,
              not_rmsf=False,
              nBits=32,
              verbose=False)

    if do_rmclean:
        clean_fdf, cc_arr, iter_count_arr, resid_fdf, header = run_rmclean(
            output_prefix + 'FDF_tot_dirty.fits',
            output_prefix + 'RMSF_tot.fits',
            clean_threshold,
            gain=0.1,
            maxIter=3000,
            nBits=32,
            pool=pool,
            chunksize=100,
            verbose=True,
            log=print,
            window=np.nan,
        )

        writefits_clean(clean_fdf,
             cc_arr,
             iter_count_arr,
             resid_fdf,
             header,
             prefixOut=output_prefix,
             outDir='./',
             write_separate_FDF=True,
             nBits=32,
             verbose=False)


def parse_args():
    """Argument parser"""

    parser = ArgumentParser(description='Perform RM synthesis and cleaning.')
    parser.add_argument('--input_directory', help='Directory with Stokes Q and U images', type=str, default='./')
    parser.add_argument('--dphi', help='Delta phi', type=float, default=0.3)
    parser.add_argument('--phi_max', help='Phi maximum', type=float, default=60.)
    parser.add_argument('--prefix', help='Output prefix', type=str, default="RMsynth")
    parser.add_argument('--clean_threshold', help='Cleaning threshold', type=float)
    parser.add_argument('--do_rmclean', action='store_true', help="Do RM Cleaning")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get input images
    i_fits = None #glob(args.input_directory+"/*-0???-I-model.fits")
    u_fits = glob(args.input_directory+"/*-0???-U-image.fits")
    q_fits = glob(args.input_directory+"/*-0???-Q-image.fits")

    # Perform RM synthesis
    do_RMsynt(i_fits,
              q_fits,
              u_fits,
              args.prefix,
              args.dphi,
              args.phi_max,
              args.clean_threshold,
              args.do_rmclean)


if __name__ == "__main__":
    main()
