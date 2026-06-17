#!/bin/bash

# Input MS
MS=$1

# Stokes IV imaging
wsclean \
-no-update-model-required \
-minuv-l 80.0 \
-size 256 256 \
-reorder \
-weight briggs 0.2 \
-parallel-reordering 4 \
-mgain 0.6 \
-data-column DATA \
-channels-out 250 \
-parallel-deconvolution 1024 \
-parallel-gridding 4 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-multiscale \
-multiscale-scale-bias 0.7 \
-multiscale-max-scales 8 \
-pol iv \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-name 1.2arcsec \
-scale 0.3arcsec \
-join-channels \
-nmiter 12 \
-niter 50000 \
-beam-size 1.2 \
-taper-gaussian 1.0asec \
${MS}

breizorro --make-binary --fill-holes --threshold=3 --restored-image=1.2arcsec-MFS-I-image.fits --boxsize=30 --outfile=pol.mask.fits

# Stokes QU imaging
wsclean \
-no-update-model-required \
-minuv-l 80.0 \
-size 256 256 \
-reorder \
-weight briggs 0.2 \
-parallel-reordering 4 \
-mgain 0.6 \
-data-column DATA \
-channels-out 250 \
-parallel-deconvolution 1024 \
-parallel-gridding 4 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-fits-mask pol.mask.fits \
-pol qu \
-no-mf-weighting \
-fit-rm \
-beam-size 1.2 \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-name 1.2arcsec \
-scale 0.3arcsec \
-join-polarizations \
-join-channels \
-squared-channel-joining \
-nmiter 12 \
-niter 10000 \
-taper-gaussian 1.0asec \
${MS}

# Cleanup
rm *-0???-*residual*.fits
rm *-0???-*psf*.fits
rm *-0???-*model*.fits
rm *-0???-*dirty*.fits
