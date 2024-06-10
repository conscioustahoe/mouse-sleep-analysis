# Sleep scoring model

## Prerequisite
A CUDA-enabled GPU machine.

## Setup

1. Run `setup.sh` to install neccessary python packages.
2. Run `cudamat_example.ipynb` to verify cudamat installation, cudamat provides an interface to perform matrix calculations on CUDA-enabled GPUs.

## Preprocess data
Add feature file to `sample_data/` dir. The feature file should be of the following format.

### Bands
NxM data array, where N is the number of epochs, and columns refer to Delta PFC, Theta HPC etc.

### EpochsLinked
Nx4 data array, where N is the number of epochs, and columns are described as follows:
- column 1: epoch ID
- column 2: epoch index (currently not used)
- column 3: ground truth sleep stage ID, where
  - 1 is associated with wakefulness,
  - 2 is associated with NREM sleep,
  - 3 is associated with REM sleep
- column 4: the subject ID (used in multi-subject analysis only)

### EpochTime
Nx3 data array, where N is the number of epochs, and columns are described as follows:
- column 1: epoch ID
- column 2: recording mode (i.e. baseline or recovery), where
  - 1 is associated with baseline,
  - 2 is associated with recovery (after sleep deprivation)
- column 3: the epoch date-time

If your feature file is not in the above format, use `mcRBM_input_features.ipynb` in `sample_data/` dir to generate the feature file (`.npz`) needed as input for the model.

## Configuration
Once you have the feature file (`.npz`) ready, update the experiment details in `configuration_files/exp_details` file. 

1. Set `dsetDir` to the absolute path of `sample_data` dir.
2. Set `dSetName` to the name of the feature file.
3. Set `expsDir` to the absolute path of output or analysis dir where model weights, inference analysis and plots will be stored.
4. Set `expID` to the unique name of your experiment

There are some parameters and flags that you can set that are useful for training the model and they are described in the `configuration_files/exp_details` file.

You can also tune parameters described in the `configuration_files/input_configuration` file that are also useful for training the model.


# Training and inference

Now, the model can be trained following which we can do inference analysis to get latent states and classification of the states.

## Step 1: Train the model
Run `train_model.ipynb`
## Step 2: Latent state inference
Run `infer_states.ipynb`
## Step 3: Latent state analysis
Run `latent_states_analysis.ipynb`

