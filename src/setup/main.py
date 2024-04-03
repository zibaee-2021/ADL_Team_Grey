# This is just to check you have all the data saved
# please use the notebook to access the Animals-10 data

from src.utils.paths import *  # bad practice in general
import os

if __name__ == '__main__':
    assert os.path.exists(animals_10_dir)
    assert os.path.exists(oxford_3_dir)
    # add more as necessary
    print("Datasets saved in correct location")
