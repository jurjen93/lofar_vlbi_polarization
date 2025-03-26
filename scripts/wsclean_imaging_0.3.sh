#!/bin/bash

# Input MS
MS=$1

# Stokes IV imaging
wsclean \
-no-update-model-required \
-minuv-l 80.0 \
-size 800 800 \
-reorder \
-weight briggs -1.5 \
-parallel-reordering 4 \
-mgain 0.8 \
-data-column DATA \
-channels-out 96 \
-parallel-deconvolution 1024 \
-parallel-gridding 6 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-multiscale \
-multiscale-scale-bias 0.75 \
-multiscale-max-scales 8 \
-pol iv \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-name pol \
-scale 0.075arcsec \
-join-channels \
-nmiter 12 \
-niter 100000 \
${MS}

breizorro --make-binary --fill-holes --threshold=4 --restored-image=pol-MFS-I-image.fits --boxsize=30 --outfile=pol.mask.fits

# Stokes QU imaging
wsclean \
-no-update-model-required \
-minuv-l 80.0 \
-size 800 800 \
-reorder \
-weight briggs -1.5 \
-parallel-reordering 4 \
-mgain 0.8 \
-data-column DATA \
-channels-out 96 \
-parallel-deconvolution 1024 \
-parallel-gridding 6 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-multiscale \
-multiscale-scale-bias 0.75 \
-multiscale-max-scales 8 \
-pol qu \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-name pol \
-scale 0.075arcsec \
-join-polarizations \
-join-channels \
-squared-channel-joining \
-nmiter 12 \
-niter 30000 \
${MS}

# Cleanup
rm *-0???-*residual*.fits
rm *-0???-*psf*.fits
rm *-0???-*model*.fits
rm *-0???-*dirty*.fits
