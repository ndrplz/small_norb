def matrix_type_from_magic(magic_number):
    """
    Get matrix data type from magic number
    See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.

    Parameters
    ----------
    magic_number: tuple
        First 4 bytes read from small NORB files 

    Returns
    -------
    element type of the matrix
    """
    convention = {'1E3D4C51': 'single precision matrix',
                  '1E3D4C52': 'packed matrix',
                  '1E3D4C53': 'double precision matrix',
                  '1E3D4C54': 'integer matrix',
                  '1E3D4C55': 'byte matrix',
                  '1E3D4C56': 'short matrix'}
    magic_str = bytearray(reversed(magic_number)).hex().upper()
    return convention[magic_str]
