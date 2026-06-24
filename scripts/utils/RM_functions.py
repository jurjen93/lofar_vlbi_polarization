import numpy as np

def functionRMdepol(lam2, norm, RM, offset, sigmaRMsq, lambdaref2):
    """
    Model for fractional Stokes Q and U with Faraday rotation and depolarisation.

    The polarisation angle rotates as:
        phi = RM * (lambda^2 - lambdaref2) + offset

    Depolarisation follows a Burn (1966) screen:
        exp(-2 * sigmaRMsq * lambda^4)

    Args:
        lam2      : lambda^2 array
        norm      : Intrinsic fractional polarisation (dimensionless).
        RM        : Rotation measure in rad m^-2.
        offset    : Polarisation angle at lambdaref2 (chi0) in rad.
        sigmaRMsq : Depolarisation term (sigma_RM^2) in rad^2 m^-4.
                    Set to 0 for no depolarisation.
        lambdaref2: Reference lambda^2 in m^2 at which offset is defined.
                    Should be np.median(lambda^2) of the data.

    Returns:
        Concatenated array [Q_model | U_model] of fractional Stokes parameters.
    """

    phase_Q = norm * np.cos(2.0 * (RM * (lam2 - lambdaref2) + offset)) \
              * np.exp(-2.0 * sigmaRMsq * lam2**2)
    phase_U = norm * np.sin(2.0 * (RM * (lam2 - lambdaref2) + offset)) \
              * np.exp(-2.0 * sigmaRMsq * lam2**2)

    return np.append(phase_Q, phase_U)

def function_synch_simple(freqvec, norm, alpha, freq_ref):
    """
    Simple power-law synchrotron spectrum.

    Models Stokes I as a power law in frequency:
        I(nu) = norm * (nu / freq_ref)^alpha

    Args:
        freqvec : Frequency array in Hz.
        norm    : Flux density at freq_ref in Jy.
        alpha   : Spectral index (typically negative, e.g. -0.7 for steep spectrum).
        freq_ref: Reference frequency in Hz. Defaults to 150 MHz.

    Returns:
        Flux density array in Jy at each frequency in freqvec.
    """
    return norm * ((freqvec / freq_ref) ** (alpha))
