import os
import time
import pandas as pd
import numpy as np

start = None

# helper functions


def clean_output(verbose: bool) -> None:
    '''puts all the files (other than the README) in the outputfiles/ folder into a subfolder

    :param bool verbose: Can turn on to print the file names it has moved.
    '''
    files = os.listdir('outputfiles')
    now = time.time()
    if not os.path.isdir(f'outputfiles/{now}'):
        os.makedirs(f'outputfiles/{now}')
    for f in files:
        if f == 'README.md' or os.path.isdir(f'outputfiles/{f}'):
            continue
        else:
            try:
                os.rename(f'outputfiles/{f}', f'outputfiles/{now}/{f}')
                if verbose:
                    print(f'moving {f} to {now}/{f}')
            except:
                print(f'Had issues moving {f} to a new folder')
    return


def set_start(start_time: time.time) -> None:
    '''This sets the start time so that the clock time function can work

    :param time.time start_time: The time that will be used as the start time of the program
    '''
    global start
    start = start_time
    return


def write_scorecard(line: str) -> None:
    '''function used to write the key results to a scorecard (for posterity)

    :param str line: text to be put onto the scorecard
    '''
    with open('outputfiles/scorecard.txt', 'a+') as f:
        f.write(line)
        f.write('\n')


def clock_time(message: str) -> None:
    '''Get the time from the start of the process and write it to the scorecard with some message

    :param str message: The message to put before the time being spent
    '''
    now = time.time()
    rt = now-start
    write_scorecard(f'{message}: {rt}')
    return


def save_matrix_csv(matrix: list[list], filename: str) -> None:
    '''Function to save the matrix as a CSV file

    :param list[list] matrix: matrix to be written
    :param str filename: place to write it
    '''
    pd.DataFrame(matrix).to_csv(filename, index=False, header=False)
    return


def ricci_normalizing(R: float) -> float:
    '''
    Using the normalization function sigma(R)
    Where sigma(x) is the standard sigmoid function 1/(1+\exp(-x))

    :param float R: the ORC value to be normalized 
    :return float: The normalized ORC value
    '''
    return (1/(1 + np.exp(-R)))
