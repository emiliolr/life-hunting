import pandas as pd
from io import StringIO

def read_csv_non_utf(filepath):

    """
    A wrapper function to handle cases where a CSV contains non-UTF-8 characters.

    Parameters
    ----------
    filepath : string
        the path to CSV file

    Returns
    -------
    dataset : pd.DataFrame
        the read dataframe with non-UTF-8 characters removed
    """

    # Removing the non-UTF-8 characters present in the CSV file
    data = ''
    with open(filepath, 'rb') as f:
        for line in f:
            line = line.decode('utf-8', 'ignore')
            data += line

    # Turning the string into a pandas dataframe
    dataset = pd.read_csv(StringIO(data))

    return dataset
