# A Reinforcement-Driven Multiple Instance Learning Method for Multi-Task Speaker Attribute Prediction

This repository contains the official implementation for the Master's Thesis, "A Reinforcement-Driven Multiple Instance Learning Method for Multi-Task Speaker Attribute Prediction".

This guide provides instructions to set up the environment, prepare the dataset, and reproduce the experiments for the single-task (RL-MIL) and multi-task (RL-MTL-MIL) models.

## 1. Setup

### Requirements

* Python 3.8+
* PyTorch
* Transformers
* scikit-learn
* pandas
* Weights & Biases

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/st0nelake/MIL-for-Multi-Task-Speaker-Attribute-Prediction.git](https://github.com/st0nelake/MIL-for-Multi-Task-Speaker-Attribute-Prediction.git)
    cd MIL-for-Multi-Task-Speaker-Attribute-Prediction/RL-MTL-MIL
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Dataset Preparation

The models are trained on the U.S. Congressional Record dataset. Follow these steps to download and preprocess the data.

### Step 1: Download the Raw Data

1.  Download the parsed Congressional Record text data from the Stanford Libraries collection: [Congressional Record Corpus](https://data.stanford.edu/congress_text).
2.  You will need the `hein-daily` speech files.
3.  Create a directory `data/hein-daily/` inside the `RL-MTL-MIL` folder and place the downloaded speech files there. The structure should look like this:
    ```
    RL-MTL-MIL/
    └── data/
        └── hein-daily/
            ├── speeches_097.txt
            ├── speeches_098.txt
            └── ...
    ```

### Step 2: Prepare the Dataset

Run the provided shell script to process the raw speeches and create the final dataset splits (`train`, `dev`, `test`).

```bash
bash scripts/prepare_dataset.sh
```

This script will run the full preprocessing pipeline and generate the final dataset file at `data/political_data.pkl`, which will be used for training.

## 3. Experiment Tracking with Weights & Biases

This project uses [Weights & Biases (wandb)](https://wandb.ai/) to log and visualize experiment results.

1.  If you don't have an account, sign up for free at [wandb.ai](https://wandb.ai).
2.  Log in to your `wandb` account from your terminal. You will need to provide your API key.
    ```bash
    wandb login
    ```
3.  Once you are logged in, the training scripts will automatically log all metrics, configurations, and results to your `wandb` dashboard under the project named `RL_MIL_small`.

## 4. Reproducing the Experiments

You can run the models using the provided shell scripts. To run experiments with different hyperparameters, you can modify the configuration variables directly within these scripts.

### Running the RL-MIL Model (Single-Task)

The `scripts/run_rlmil.sh` script automates the process of running single-task experiments. It iterates through different tasks (`age`, `gender`, `party`) and model configurations defined within the script.

To run the full suite of single-task experiments, execute the following commands:
```bash
cd scripts
./run_rlmil.sh
```

You can customize the experiments by editing the variables at the top of the `scripts/run_rlmil.sh` file, such as `target_labels`, `baseline_types`, and `embedding_models`.

**Note:** The `run_rlmil.sh` script expects the dataset to be named `political_data_with_age`. Please ensure the `dataset` variable in the script is set to `political_data` to match the output of the preparation script, or rename the output file accordingly.

### Running the RL-MTL-MIL Model (Multi-Task)

The `scripts/run_rl_mtl_sweep.sh` script trains the multi-task model on all three speaker attributes simultaneously and performs a hyperparameter sweep.

To run the multi-task experiment, execute the following command:
```bash
cd scripts
./run_rl_mtl_sweep.sh
```

This will start the training process, and all results will be logged to your `wandb` dashboard under the `RL_MIL_small` project.

## Citation

If you use this work, please cite the thesis as follows:

```
@mastersthesis{lakeman2025,
  author  = {Stijn Lakeman},
  title   = {A Reinforcement-Driven Multiple Instance Learning Method for Multi-Task Speaker Attribute Prediction},
  school  = {University of Amsterdam},
  year    = {2025},
  address = {Amsterdam, The Netherlands},
  month   = {June}
}
