"""
Modified version from lofar_helpers
"""

import tables
from subprocess import call

class ApplyCal:
    def __init__(self, msin: str = None, h5: str = None, msincol: str = "DATA", msoutcol: str = "CORRECTED_DATA",
                 msout: str = '.', dysco: bool = True):
        """
        Apply calibration solutions

        :param msin: input measurement set
        :param h5: solution file to apply
        :param msincol: input column
        :param msoutcol: output column
        :param msout: output measurement set
        :param dysco: compress with dysco
        """

        self.cmd = ['DP3', 'msin=' + msin]
        self.cmd += ['msout=' + msout]
        self.cmd += ['msin.datacolumn=' + msincol]
        if msout == '.':
            self.cmd += ['msout.datacolumn=' + msoutcol]
        if dysco:
            self.cmd += ['msout.storagemanager=dysco']

        steps = []

        poldim_num = self.poldim_num(h5)

        # fulljones
        if poldim_num==4:
            steps.append('ac')
            self.cmd += ['ac.type=applycal',
                         'ac.parmdb=' + h5,
                         'ac.correction=fulljones',
                         'ac.soltab=[amplitude000,phase000]',
                         'ac.updateweights=True']

        # add non-fulljones solutions apply
        else:
            ac_count = 0
            with tables.open_file(h5) as T:
                for corr in T.root.sol000._v_groups.keys():
                    self.cmd += [f'ac{ac_count}.type=applycal',
                                 f'ac{ac_count}.parmdb={h5}',
                                 f'ac{ac_count}.correction={corr}']
                    steps.append(f'ac{ac_count}')
                    ac_count += 1

        # non-scalar
        if poldim_num>1:
            # this step inverts the beam at the infield and corrects beam at phase center
            steps.append('beam_center')
            self.cmd += ['beam_center.type=applybeam', 'beam_center.direction=[]',
                         'beam_center.updateweights=True']

        self.cmd += ['steps=' + str(steps).replace(" ", "").replace("\'", "")]

    @staticmethod
    def poldim_num(h5: str = None):
        """
        Verify if file is fulljones

        :param h5: h5 file
        """
        with tables.open_file(h5) as T:
            soltab = list(T.root.sol000._v_groups.keys())[0]
            if 'pol' in T.root.sol000._f_get_child(soltab).val.attrs["AXES"].decode('utf8'):
                return T.root.sol000._f_get_child(soltab).pol[:].shape[0]
            else:
                return 0

    def print_cmd(self):
        """Print DP3 command"""
        print('\n'.join(self.cmd))
        return self

    def run(self):
        """Run DP3 command"""
        retval = call(' '.join(self.cmd), shell=True)
        if retval != 0:
            print('FAILED to run ' + ' '.join(self.cmd) + ': return value is ' + str(retval))
            raise Exception(' '.join(self.cmd))
        return retval
