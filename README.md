## Project Setup

### Installing Requirements

In the main project directory, there should be a `requirements.txt` file that lists these packages.

To install the required packages, follow these steps:

0. Have a venv created and activated
1. Open your terminal or command prompt.
2. Navigate to the main project directory using the `cd` command.
3. Run the following command:

```bash
pip install -r requirements.txt
```

### Updating the Requirements File

During development, if you install new packages and want to add them to the `requirements.txt` file, follow these steps:

1. Make sure you have installed the new package(s) in your current environment.
2. Open your terminal or command prompt.
3. Navigate to the main project directory using the `cd` command.
4. Run the following command:

```bash
pip freeze > requirements.txt
```

## Dataset Setup

The datasets required for this project are available at the following sources:

- [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- [Oxford University](https://www.robots.ox.ac.uk/~vgg/data/pets/)

### Setup Instructions

Create a `datasets` directory within the `src` directory of the project. 
The path should look like this: `/ADL_Team_Grey/src/datasets`

Then add 2 further directories within `datasets`, `Oxford-3` and `Animals-10`

### Animals10 Dataset
After downloading the Animals10 dataset, locate the `raw-img` folder within the downloaded archive. Move this folder into the `datasets/Animals-10` directory. The final path should look like this: `/ADL_Team_Grey/src/datasets/Animals-10/raw-img`

### Oxford-3 Dataset
After downloading both the annotations and the images, move each folder within `/ADL_Team_Grey/src/datasets/Oxford-3` 

### Dataset Setup Verification

After placing the datasets in the designated directories, you can confirm the setup by running a provided verification script. This script checks for the existence of the necessary directories and files.

To run the verification script, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the main project directory `/ADL_Team_Grey/` using the `cd` command.
3. Execute the verification script with the following command:

```bash
python -m src.setup.main
```
