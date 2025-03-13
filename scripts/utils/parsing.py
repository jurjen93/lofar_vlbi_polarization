import re
from astropy.io import fits
import sys

def extract_l_number(filename):
    """
    Parse L-number
    Args:
        filename: File name

    Returns: Parsed L-number

    """

    regex = r'L\d{6,7}'

    match = re.search(regex, filename)
    if match:
        return match.group(0)
    else:
        with fits.open(filename) as hdul:
            head = hdul[0].header
            match = re.search(regex, str(head).replace("HISTORY ",""))
            if match:
                return match.group(0)
            else:
                sys.exit("ERROR: No L-number parsed from input")

