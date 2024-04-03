### Important things to Note (and possibly improve)
- There is some shared architecture between the MVAE and Fine tuning (found in shared_network_architecture).
- There are also shared paramaters (currently these live in the two different train,py's but they should live in a central config).
- Please copy the dataset file structure I posted on WhatsApp - if you run src/setup/main.py it will check you have the correct directories.
- You can download the Animals-10 dataset using the notebook read_datasets/read_datasets.ipynb (you will have to "pip install datasets").
- Could consider also pushing the trained models to the repo (at least for MVAE) - they are not so big (100 mb).