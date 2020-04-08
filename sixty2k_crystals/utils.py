import urllib
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def download_data(directory=None):
    '''
    Downloads the dataset for the 62k crystal analysis.

    Parameters
    ----------
    directory : str or None
        Directory where the 'm1507656' file currently lives
        if already downloaded, or None if not already
        downloaded

    Returns
    -------
    filename : str
        The absolute filename of the 'df_62k.json' file
        used for the downstream analysis

    '''
    if directory is None:
        directory = os.getcwd()
    else:
        pass

    filename = os.path.join(directory, 'm1507656.zip')
    url = 'https://dataserv.ub.tum.de/s/m1507656/download'
    urllib.request.urlretrieve(url, filename)

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory)

    return os.path.join(directory, 'm1507656\\df_62k.json')


def unpack_data(filename):
    '''
    Stores data from the json file into a pandas dataframe
    and preps SMILES strings data for downstream analysis.

    Parameters
    ----------
    filename : str
        File location of the data.

    Returns
    -------
    df_62k : pandas
        A pandas dataframe of the data
    molecules : list
        A list of the SMILES strings expressed as lists of
        individual characters

    '''

    # Unpack data into pandas dataframe
    df_62k = pd.read_json(filename, orient='split')

    # Extract SMILES strings
    molecules = df_62k['canonical_smiles'].values
    maxlen = len(max(molecules, key=len))

    molecules = molecules.reshape(-1, 1)
    molecules = molecules.astype(str)

    # Pad strings so all the same length
    molecules = np.char.zfill(molecules, width=maxlen)

    # Turns the string array into a list of single characters
    molecules = molecules.tolist()
    molecules2 = [list(x[0]) for x in molecules]

    return df_62k, molecules2


def encoded_smiles(string_array):
    '''
    Convert SMILES Strings into OneHotEncode 2D arrays

    Parameters
    ----------
    string_array : list
        A list of lists containing SMILES strings broken up into
        indvidual characters

    Returns
    -------
    enc : model
        The encoder model used to transform SMILES strings
        into binary representation
    x1 : list
        A list of binary representations of the SMILES strings
        inputs

    '''

    enc = OneHotEncoder(handle_unknown='ignore')
    x1 = enc.fit(string_array)
    x1 = x1.transform(string_array).toarray()

    return enc, x1
