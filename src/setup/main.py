# This is just to check you have all the data saved
# please use the notebook to access the Animals-10 data

from src.utils.paths import *  # bad practice in general
import os

if __name__ == '__main__':
    assert os.path.exists(animals_10_dir)
    assert os.path.exists(os.path.join(animals_10_dir,"raw-img"))
    assert os.path.exists(oxford_3_dir)
    assert os.path.exists(os.path.join(oxford_3_dir,"images"))
    assert os.path.exists(os.path.join(oxford_3_dir,"annotations/trimaps"))
    # add more as necessary
    print("Directories for datasets exist in expected location.")
