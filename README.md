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
## Experiment Tracking

To facilitate experiment tracking, parameter management, and output visualization, we'll be utilizing [WandB](https://wandb.ai/home).

### Setup Instructions

1. **Create an Account**: Begin by creating an account on WandB.

2. **Provide Username**: Send your WandB username to Rich via WhatsApp at +447506219401. Rich will then add you to our organization, granting access to view and add to our experiments.

3. **Install WandB**: Install WandB by running the following command in your terminal:
    ```
    pip install wandb
    ```

4. **Authenticate**: After installation, run `wandb login` in your terminal. This will prompt you to authenticate with your WandB account. Upon successful authentication, you'll be ready to proceed.

### Experiment Tracking

- **Initialization**: To begin tracking an experiment, include the following line at the top of your experiment script:
    ```python
    import wandb

    # your models/experiment parameters here
    params = {
        'example',
        'params'
    }

    wandb.init(project="mvae", entity="adl_team_grey", config=params)
    ```

    - Modify the `project` parameter to specify the project name. This will allow all runs within a project to be compared (check out the project page in WandB for an example). Ensure `entity` is kept as "adl_team_grey" to ensure visibility within the team.

- **Logging**: Throughout your experiment, use `wandb.log()` to log relevant metrics and outputs. For example:
    ```python
    wandb.log({"Epoch Loss": epoch_loss, "Epoch Time": epoch_time})
    ```

    - You can log various metrics, including numerical values, text, and even plots. For image outputs, simply log the matplotlib plot object:
    ```python
    
    wandb.log({plot_name: plt})
    ```

    - Ensure consistency in logged metrics for easier comparison across experiments.

With these steps, you'll effectively track experiments, parameters, and outputs using WandB, facilitating collaboration and analysis within our team.
