import numpy as np


def functionRM(wav, norm, RM, offset, lambdaref2=4.5):
    phase_Qpart = norm * np.cos(2. * (RM * ((wav[0:int(len(wav) / 2)]) ** 2 - lambdaref2) + offset))
    phase_Upart = norm * np.sin(2. * (RM * ((wav[int(len(wav) / 2)::]) ** 2 - lambdaref2) + offset))
    QU = np.append(phase_Qpart, phase_Upart)
    return QU


def functionRMdepol(wav, norm, RM, offset, sigmaRMsq, lambdaref2=4.5):
    phase_Qpart = norm * np.cos(2. * (RM * ((wav[0:int(len(wav) / 2)]) ** 2 - lambdaref2) + offset)) * np.exp(
        -2. * sigmaRMsq * (wav[0:int(len(wav) / 2)]) ** 4)

    phase_Upart = norm * np.sin(2. * (RM * ((wav[int(len(wav) / 2)::]) ** 2 - lambdaref2) + offset)) * np.exp(
        -2. * sigmaRMsq * (wav[int(len(wav) / 2)::]) ** 4)
    QU = np.append(phase_Qpart, phase_Upart)
    return QU


def function_synch_simple(freqvec, norm, alpha, freq_ref=150.e6):
    return norm * ((freqvec / freq_ref) ** (alpha))


def make_P(Qflux, Uflux, sigma_Q, sigma_U):
    sigma_QU = 0.5 * (sigma_Q + sigma_U)
    P = np.sqrt((Qflux ** 2 + Uflux ** 2) - (2.3 * (sigma_QU ** 2)))
    P[~np.isfinite(P)] = 0.
    return P