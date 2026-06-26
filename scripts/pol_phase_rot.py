import os
from argparse import ArgumentParser
from shutil import copy

import numpy as np
from scipy.constants import speed_of_light
import tables
from casacore.tables import table

circ2lin_math = """
-----------------------------
XX = RR + RL + LR + LL
XY = iRR - iRL + iLR - iLL
YX = -iRR - iRL + iLR + iLL
YY = RR - RL - LR + LL
-----------------------------
"""


class PhaseRotate:
    """
    Make h5parm for polarization alignment between different observations
    (see van Weeren et al. 2026; https://arxiv.org/pdf/2606.18333)
    """

    def __init__(self, ms_in, h5_out):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        source_file = os.path.join(parent_dir, "scripts/tmpdata/tmp.h5")
        copy(source_file, h5_out)

        self.h5 = tables.open_file(h5_out, 'r+')
        self.axes = ['time', 'freq', 'ant', 'dir', 'pol']

        with table(ms_in+"::SPECTRAL_WINDOW", ack=False) as ms:
            self.freqs = ms.getcol("CHAN_FREQ")[0]
        with table(ms_in, ack=False) as ms:
            self.time = np.array([np.unique(ms.getcol("TIME"))[0]])
        with table(ms_in+"::ANTENNA", ack=False) as ms:
            self.ant = np.array(ms.getcol("NAME")).astype("S")
        with table(ms_in+"::FIELD", ack=False) as ms:
            phasedir = list(ms.getcol("PHASE_DIR").squeeze())
            values = np.array([(b'Dir00', phasedir)], dtype=[('name', 'S128'), ('dir', '<f4', (2,))])
            self.h5.root.sol000.source._f_remove()
            self.h5.create_table(self.h5.root.sol000, 'source', values, title='Source names and directions')

    def update_array(self, st, new_val, arrayname):
        """
        Update array

        :param st: soltab
        :param new_val: new values
        :param array: array name (val, weight, pol, dir, or freq)
        """
        try:
            valtype = str(st._f_get_child(arrayname).dtype)
            st._f_get_child(arrayname)._f_remove()
        except:
            if 'pol' in arrayname:
                valtype = '|S2'
        if 'float' in str(valtype):
            if '16' in valtype:
                atomtype = tables.Float16Atom()
            elif '32' in valtype:
                atomtype = tables.Float32Atom()
            elif '64' in valtype:
                atomtype = tables.Float64Atom()
            else:
                atomtype = tables.Float64Atom()
            self.h5.create_array(st, arrayname, new_val.astype(valtype), atom=atomtype)
        else:
            self.h5.create_array(st, arrayname, new_val.astype(valtype))
        if arrayname == 'val' or arrayname == 'weight':
            st._f_get_child(arrayname).attrs['AXES'] = bytes(','.join(self.axes), 'utf-8')

        return self

    def make_template(self):
        """
        Make template h5parm
        """

        print("1) MAKE TEMPLATE H5PARM")
        for solset in self.h5.root._v_groups.keys():
            ss = self.h5.root._f_get_child(solset)
            for soltab in ss._v_groups.keys():
                st = ss._f_get_child(soltab)

                shape = (1, len(self.freqs), len(self.ant), 1, 4)

                if 'phase' in soltab:
                    new_val = np.zeros(shape)
                elif 'amplitude' in soltab:
                    new_val = np.ones(shape)
                    new_val[..., 1] = 0
                    new_val[..., 2] = 0
                else:
                    continue
                self.update_array(st, new_val, 'val')
                self.update_array(st, np.ones(shape), 'weight')
                self.update_array(st, self.time, 'time')
                self.update_array(st, np.array(['XX', 'XY', 'YX', 'YY']), 'pol')
                self.update_array(st, self.freqs, 'freq')
                self.update_array(st, self.ant, 'ant')

        return self

    def circ2lin(self):
        """
        Convert circular polarization to linear polarization

        XX = RR + RL + LR + LL
        XY = iRR - iRL + iLR - iLL
        YX = -iRR - iRL + iLR + iLL
        YY = RR - RL - LR + LL

        :return: linear polarized solutions
        """

        print('\n3) CONVERTING CIRCULAR TO LINEAR POLARIZATION\n'
              '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
              + circ2lin_math +
              '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        G = self.h5.root.sol000.amplitude000.val[:] * np.exp(self.h5.root.sol000.phase000.val[:] * 1j)

        XX = (G[..., 0] + G[..., -1])
        XY = 1j * (G[..., 0] - G[..., -1])
        YX = 1j * (G[..., -1] - G[..., 0])
        YY = (G[..., 0] + G[..., -1])

        XX += (G[..., 2] + G[..., 1])
        XY += 1j * (G[..., 2] - G[..., 1])
        YX += 1j * (G[..., 2] - G[..., 1])
        YY -= (G[..., 1] + G[..., 2])

        XX /= 2
        XY /= 2
        YX /= 2
        YY /= 2

        G_new = np.zeros(G.shape[0:-1] + (4,)).astype(np.complex128)

        G_new[..., 0] += XX
        G_new[..., 1] += XY
        G_new[..., 2] += YX
        G_new[..., 3] += YY

        G_new = np.where(abs(G_new) < 10 * np.finfo(float).eps, 0, G_new)

        phase = np.angle(G_new)
        amplitude = abs(G_new)

        self.update_array(self.h5.root.sol000.phase000, phase, 'val')
        self.update_array(self.h5.root.sol000.amplitude000, amplitude, 'val')

        return G_new

    def rotate(self, intercept, rotation_measure):
        """
        Rotate angle by the following matrix:
         /e^(i*rho)  0 \
        |              |
         \ 0         1/

        :param intercept: intercept
        :param rotation_measure: rotation measure in rad/m^2
        """
        print('\n2) ADD PHASE ROTATION')

        phaserot = intercept + rotation_measure * (speed_of_light / self.freqs) ** 2

        mapping = list(zip(list(self.freqs), list(phaserot)))
        print('########################\nFrequency to rotation in radian (circular base):\n------------------------')
        for element in mapping:
            print(str(int(element[0])) + 'Hz --> ' + str(round(element[1], 3)) + 'rad')

        # print('Rotate with rotation angle: '+str(intercept) + ' radian')
        for solset in self.h5.root._v_groups.keys():
            ss = self.h5.root._f_get_child(solset)
            for soltab in ss._v_groups.keys():
                if 'phase' in soltab:
                    st = ss._f_get_child(soltab)
                    phaseval = st.val[:]
                    phaseval[0, :, :, 0, 0] += phaserot[:, np.newaxis]
                    self.update_array(st, phaseval, 'val')
        print("########################")

        self.circ2lin()

        return self


def parse_args():
    """
    Command line argument parser

    :return: parsed arguments
    """
    parser = ArgumentParser("Generate h5parm for polarization alignment between different observations.")
    parser.add_argument('--ms_in', type=str, help='Input MS (from which to extract the frequencies and antennas).')
    parser.add_argument('--h5_out', type=str, help='Output name (output solution file).', default='polrot.h5')
    parser.add_argument('--intercept', type=float, help='Intercept for rotation angle.')
    parser.add_argument('--RM', type=float, help='Rotation measure.')
    return parser.parse_args()


def main():
    args = parse_args()

    phaserot = PhaseRotate(ms_in=args.ms_in, h5_out=args.h5_out)
    phaserot.make_template()
    phaserot.rotate(intercept=args.intercept, rotation_measure=args.RM)
    phaserot.h5.close()


if __name__ == '__main__':
    main()
