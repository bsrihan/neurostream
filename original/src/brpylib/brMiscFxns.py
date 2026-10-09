"""
Random functions that may be useful elsewhere (or necessary)
current version: 1.2.0 --- 08/04/2016

@author: Mitch Frankel - Blackrock Microsystems

Version History:
v1.0.0 - 07/05/2016 - initial release
v1.1.0 - 07/12/2016 - minor editing changes to print statements and addition of version control
v1.2.0 - 08/04/2016 - minor modifications to allow use of Python 2.6+
"""
from os import path

# Version control
brmiscfxns_ver = "1.2.0"


def openfilecheck(open_mode, file_name="", file_ext=""):
    """
    :param open_mode: {str} method to open the file (e.g., 'rb' for binary read only)
    :param file_name: [optional] {str} full path of file to open
    :param file_ext:  [optional] {str} file extension (e.g., '.nev')
    :return: {file} opened file
    """

    # Ensure file exists
    if path.isfile(file_name) and file_ext:
        # Ensure given file matches file_ext
        if file_ext:
            _, fext = path.splitext(file_name)

            # check for * in extension
            if file_ext[-1] == "*":
                test_extension = file_ext[:-1]
            else:
                test_extension = file_ext

            if fext[0:len(test_extension)] != test_extension:
                raise ValueError("File given is not a " + file_ext +
                                 " file, try again")
    else:
        raise FileNotFoundError("File given does exist, try again")

    return open(file_name, open_mode)
