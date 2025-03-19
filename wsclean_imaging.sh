#!/bin/bash

# Input MS
MS=$1

# Run wsclean
wsclean \
-no-update-model-required \
-minuv-l 1500.0 \
-size 1600 1600 \
-reorder \
-weight briggs -1.5 \
-parallel-reordering 4 \
-mgain 0.8 \
-data-column DATA \
-channels-out 96 \
-auto-mask 2.5 \
-auto-threshold 0.5 \
-pol iqu \
-gridder wgridder \
-wgridder-accuracy 0.0001 \
-use-differential-lofar-beam \
-facet-beam-update 120 \
-name pol_imaging \
-scale 0.075arcsec \
-join-polarizations \
-join-channels \
-fit-spectral-pol 5 \
-nmiter 12 \
-niter 15000 \
${MS}
