from argparse import ArgumentParser
from glob import glob
from numpy import nan, median

from RMtools_3D.do_RMsynth_3D import run_rmsynth, writefits
from RMtools_3D.do_RMclean_3D import run_rmclean, writefits as writefits_clean
from utils.fits_handling import get_header, make_freq_vec, make_image_cube

import multiprocessing
pool = multiprocessing.Pool()

def do_RMsynt(q_images: list = None,
              u_images: list = None,
              output_prefix: str = 'prefix',
              output_directory: str = './',
              dphi: float = None,
              phi_max: float = None,
              n_samples: int = None):
    """
    Perform RM synthesis
    Args:
        q_images: Stokes Q fits images
        u_images: Stokes U fits images
        output_prefix: Output prefix
        output_directory: Output directory
        dphi: delta phi
        phi_max: phi max
        n_samples: number of samples
    """

    q_header = get_header(q_images[0])

    if dphi is not None:
        n_samples = None  # ignored when dPhi is given

    # Make frequency array:
    freq_array = make_freq_vec(u_images)

    # Make image cubes
    # i_data, rms_i = make_image_cube(i_images, return_noise=True)
    u_data, rms_u = make_image_cube(u_images, return_noise=True)
    q_data, rms_q = make_image_cube(q_images, return_noise=True)

    # Noise vector (take average of Q and U noises)
    rms = 0.5 * (rms_u + rms_q)

    dataArr = run_rmsynth(q_data,
              u_data,
              freq_array,
              dataI=None,
              rmsArr=rms,
              phiMax_radm2=phi_max,
              dPhi_radm2=dphi,
              nSamples=n_samples,
              weightType="variance",
              fitRMSF=False,
              nBits=32,
              verbose=True,
              not_rmsf=False)

    writefits(dataArr,
              headtemplate=q_header,
              fitRMSF=False,
              prefixOut=output_prefix,
              outDir=output_directory,
              write_seperate_FDF=True,
              not_rmsf=False,
              nBits=32,
              verbose=False)

    cleanFDF, ccArr, iterCountArr, residFDF, header = run_rmclean(
            output_prefix + 'FDF_tot_dirty.fits',
            output_prefix + 'RMSF_tot.fits',
            median(rms),
            maxIter=1000,
            gain=0.1,
            nBits=32,
            pool=pool,
            chunksize=100,
            verbose=True,
            log=print,
            window=nan,
        )

    writefits_clean(cleanFDF,
             ccArr,
             iterCountArr,
             residFDF,
             header,
             prefixOut=output_prefix,
             outDir=output_directory,
             write_separate_FDF=True,
             nBits=32,
             verbose=False)


def parse_args():
    """Argument parser"""

    parser = ArgumentParser(description='Perform RMS synthesis')
    parser.add_argument('--input_directory', help='Directory with Stokes Q and U images', type=str, required=True)
    parser.add_argument('--output_directory', help='Output image directory', type=str, default='./')
    parser.add_argument('--dphi', help='Delta phi', type=float, default=0.3)
    parser.add_argument('--phi_max', help='Phi maximum', type=float, default=30.)
    parser.add_argument('--prefix', help='Output prefix', type=str, default="RMsynth")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get input images
    u_fits = glob(args.input_directory+"/*-0???-U-image.fits")
    q_fits = glob(args.input_directory+"/*-0???-Q-image.fits")

    # Perform RM synthesis
    do_RMsynt(sorted(q_fits), sorted(u_fits), args.prefix, args.output_directory, args.dphi, args.phi_max)


if __name__ == "__main__":
    main()
