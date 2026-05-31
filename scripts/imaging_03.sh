#!/bin/bash

# Input MS
MS=$1

# Stokes IV imaging
wsclean \
-no-update-model-required \
-minuv-l 80.0 \
-size 1024 1024 \
-reorder \
-weight briggs -0.5 \
-parallel-reordering 6 \
-mgain 0.6 \
-data-column DATA \
-channels-out 250 \
-parallel-deconvolution 1024 \
-parallel-gridding 6 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-multiscale \
-multiscale-scale-bias 0.7 \
-multiscale-max-scales 8 \
-pol iv \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-name 0.3arcsec \
-scale 0.075arcsec \
-join-channels \
-nmiter 12 \
-niter 150000 \
-beam-size 0.3 \
-taper-gaussian 0.25asec \
${MS}

breizorro --make-binary --fill-holes --threshold=3 --restored-image=pol-MFS-I-image.fits --boxsize=30 --outfile=pol.mask.fits

# Stokes QU imaging
wsclean \
-no-update-model-required \
-minuv-l 80.0 \
-size 1024 1024 \
-reorder \
-weight briggs -0.5 \
-parallel-reordering 6 \
-mgain 0.6 \
-data-column DATA \
-channels-out 250 \
-parallel-deconvolution 1024 \
-parallel-gridding 6 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-fits-mask pol.mask.fits \
-pol qu \
-no-mf-weighting \
-fit-rm \
-beam-size 0.3 \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-name 0.3arcsec \
-scale 0.075arcsec \
-join-polarizations \
-join-channels \
-squared-channel-joining \
-nmiter 12 \
-niter 20000 \
-taper-gaussian 0.25asec \
${MS}

# Cleanup
rm *-0???-*residual*.fits
rm *-0???-*psf*.fits
rm *-0???-*model*.fits
rm *-0???-*dirty*.fits
