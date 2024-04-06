# Use the README.md to setup the datasets correctly

import os
from src.utils.paths import animals_10_dir, oxford_3_dir

def main():
    paths = [
        animals_10_dir,
        os.path.join(animals_10_dir, "raw-img"),
        oxford_3_dir,
        os.path.join(oxford_3_dir, "images"),
        os.path.join(oxford_3_dir, "annotations/trimaps"),
        # add more as necessary
    ]

    for path in paths:
        assert os.path.exists(path), f"Path does not exist: {path}"

    print("Datasets Correctly Setup!")

if __name__ == '__main__':
    main()