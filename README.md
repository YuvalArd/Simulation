# Simulation Framework README

## Table of Contents

- [Overview](#overview)
- [Setup Instructions](#setup-instructions)
- [Configuration File (`config.json`)](#configuration-file-configjson)
  - [General Parameters](#general-parameters)
  - [Function-Specific Parameters](#function-specific-parameters)
    - [`simulations_discrete_pcs`](#simulations_discrete_pcs)
    - [`simulations_rearranged_discrete_pcs`](#simulations_rearranged_discrete_pcs)
    - [`multiple_simulations_joined_pcs`](#multiple_simulations_joined_pcs)
- [Running the Simulations](#running-the-simulations)
  - [simulations_discrete_pcs](#simulations_discrete_pcs-1)
  - [simulations_rearranged_discrete_pcs](#simulations_rearranged_discrete_pcs-1)
  - [multiple_simulations_joined_pcs](#multiple_simulations_joined_pcs-1)
- [Interpreting Results](#interpreting-results)
- [Additional Notes](#additional-notes)

## Overview

The framework consists of the following main components:

- **`analysis_functions.py`**: Contains functions for performing ridge regression and cross-validation.
- **`config.py`**: Handles the loading and parsing of simulation configuration parameters.
- **`simulation.py`**: Implements simulation classes and methods for running different types of simulations.

## Setup Instructions

1. **Clone or Download the Repository**: Ensure you have all the necessary Python files (`analysis_functions.py`, `config.py`, `simulation.py`) and a configuration file (`config.json`).

2. **Install Dependencies**:

   - Python 3.6 or higher.
   - Required Python packages:
     - `numpy`
     - `scipy`
     - `scikit-learn`

   Install the packages using pip:

   ```bash
   pip install numpy scipy scikit-learn
   ```

3. **Prepare the Configuration File**:

   - Ensure you have a `config.json` file in the same directory as the Python scripts.
   - Customize the configuration parameters as needed (see the [Configuration File](#configuration-file-configjson) section).

## Configuration File (`config.json`)

The `config.json` file contains all the parameters required to run the simulations. The parameters are divided into general parameters and function-specific parameters.
The name and path of the config file can be changed in the beggining of the simulation.py file.

### General Parameters

These parameters are common to all simulation functions.

- **`consts`**: Contains constant values used in simulations.
  - **`N`**: (int) Number of observations\samples (rows in the data matrix).
  - **`P`**: (int) Number of predictors\simulated neurons (columns in the data matrix).
  - **`slope`**: (float) Controls the power-law decay of singular values.

- **`partial_pcs`**: (float or null) If specified, indicates the fraction of principal components out of P to retain (e.g., `0.5` to retain 50% of PCs). If `null`, all PCs are used.

### Function-Specific Parameters

#### `simulations_discrete_pcs`

Parameters specific to running `simulations_discrete_pcs`:

- **`subspace_sizes`**: (dict) Specifies the sizes (number of PCs) for each subspace.
  - **`video_public`**: (list of int) List of sizes to iterate over for the **video_public** subspace.
  - **`video_private`**: (list of int) List of sizes for the **video_private** subspace.
  - **`communication`**: (list of int) List of sizes for the **communication** subspace.

Example:

```json
"subspace_sizes": {
  "video_public": [5, 10],
  "video_private": [0, 3, 9],
  "communication": [4, 7, 10]
}
```

#### `simulations_rearranged_discrete_pcs`

This function allows to control the order of weights of the different PCs.
In addition to `subspace_sizes`, this function uses:

- **`shift`**: (bool) Indicates if singular values should be shifted. used to start the shared dimensions at a later PC, defualt is first.
  - **`true`**: Shifting is applied.
  - **`false`**: No shifting.

- **`shift_list`**: (list of int) Specifies the shift amounts to apply.

- **`mix`**: (bool) Indicates if singular values should be mixed.
  - **`true`**: Mixing is applied.
  - **`false`**: No mixing.

- **`index_change`**: (dict) Contains mixing configurations.
  - **`from`**: (list of int) Original indices of singular values to move.
  - **`to`**: (list of list of int) Target indices to move singular values to.

Use either shift or mix, not both.

Examples:

```json
"index_change": {
  "shift": true,
  "shift_list": [0, 2, 4]
}
```
```json
"index_change": {
  "from": [0, 1],
  "to": [[2, 3], [4, 5]],
  "mix": true,
}
```

#### `multiple_simulations_joined_pcs` and `single_simulation_joined_pcs`

Parameters specific to running `multiple_simulations_joined_pcs` or `single_simulation_joined_pcs`:

- **`subspace_weights`**: (dict) Specifies the weight functions and arguments for each subspace.
  - **Subspaces**: `video_public`, `video_private`, `communication`, `private`
  - Each subspace contains:
    - **`function`**: (str) Name of the weight function (e.g., `"linear"`).
    - **`arguments`**: (dict) Arguments required by the weight function.
      - For `"linear"` function: `"start"` and `"end"` weights.
    - **`limits`**: (dict) Minimum and maximum values for weights to iterate over.
      - `"min"` and `"max"`.

- **`weight_step`**: (float) Step size for iterating weights. default is 0.1.

To use `single_simulation_joined_pcs` choose specific arguments for each subspace.
For "linear"` function: `"start"` and `"end"`.
Setting `"limits"` is not required.

To use `multiple_simulations_joined_pcs` choose the limits for each subspace.
Setting `"arguments"` is not required.

Example:

```json
"subspace_weights": {
  "video_public": {
    "function": "linear",
    "arguments": {
      "start": 0.5,
      "end": 0.1
    },
    "limits": {
      "min": 0.1,
      "max": 1.0
    }
  },
  "video_private": {
    "function": "linear",
    "arguments": {
      "start": 0.4,
      "end": 0.0
    },
    "limits": {
      "min": 0.0,
      "max": 0.5
    }
  },
  // ... similar for "communication" and "private"
},
"weight_step": 0.1
```

## Running the Simulations

All simulations are run using the `Simulation` class in `simulation.py`. You need to instantiate the class and call the appropriate method.

### simulations_discrete_pcs

**Purpose**: Runs simulations by selecting discrete numbers of PCs from each subspace, without rearrangements.

**Steps**:

1. **Set Configuration**:
   - Define `N`, `P`, `slope`, and `partial_pcs` (if needed).
   - Specify `subspace_sizes` for `video_public`, `video_private`, and `communication`.

2. **Ensure Rearrangement Parameters are Disabled**:
   - Set `shift` to `false` or remove it.
   - Set `mix` to `false` or remove it.

3. **Run the Simulation**:

   ```python
   from simulation import Simulation

   sim = Simulation()
   sim.simulations_discrete_pcs()
   ```

   - The results will be saved to the output file specified in the beggining of `simulation.py`.

### simulations_rearranged_discrete_pcs

**Purpose**: Runs simulations with rearrangements (shifting or mixing) of singular values.

**Steps**:

1. **Set Configuration**:
   - Define `N`, `P`, `slope`, and `partial_pcs` (if needed).
   - Specify `subspace_sizes` as above.
   - Set `shift` to `true` if you want to apply shifting.
     - Provide `shift_list` with the shift amounts.
   - Set `mix` to `true` if you want to apply mixing.
     - Provide `index_change` configurations (`from`, `to` indices).

2. **Run the Simulation**:

   ```python
   from simulation import Simulation

   sim = Simulation()
   sim.simulations_rearranged_discrete_pcs()
   ```

   - The simulation will iterate over all combinations of sizes and rearrangements.
   - Results are saved to the output file specified in the beggining of `simulation.py`.

### single_simulation_joined_pcs

**Purpose**: Runs simulations by applying weights to PCs from each subspace and summing their contributions.

**Steps**:

1. **Set Configuration**:
   - Define `N`, `P`, `slope`, and `partial_pcs` (if needed).
   - Define `subspace_weights` for each subspace.
     - Set `function` according to the mapping at Simulation.weight_functions.
     - Provide `arguments` with `"start"` and `"end"` weights.

2. **Run the Simulation**:

   ```python
   from simulation import Simulation

   sim = Simulation()
   sim.single_simulation_joined_pcs()
   ```

   - Results are saved to the output file specified in the beggining of `simulation.py`.

### multiple_simulations_joined_pcs

**Purpose**: Runs simulations by applying weights to PCs from each subspace and summing their contributions.

**Steps**:

1. **Set Configuration**:
   - Define `N`, `P`, `slope`, and `partial_pcs` (if needed).
   - Define `subspace_weights` for each subspace.
     - Set `function` according to the mapping at Simulation.weight_functions.
     - Provide `limits` with `"min"` and `"max"` weights to iterate over.
   - Set `weight_step` to control the granularity of weight iteration.

2. **Run the Simulation**:

   ```python
   from simulation import Simulation

   sim = Simulation()
   sim.multiple_simulations_joined_pcs()
   ```

   - The simulation will iterate over all combinations of weights within specified limits.
   - Results are saved to the output file specified in the beggining of `simulation.py`.

## Interpreting Results

The results are saved in `output.json` and typically include:

- **For `simulations_discrete_pcs` and `simulations_rearranged_discrete_pcs`**:
  - A list of entries, each containing:
    - The subspace sizes used.
    - The R² metrics obtained:
      - `"video"`: R² for video prediction.
      - `"cs"`: R² for cross-space prediction.
      - `"residual"`: R² for residual prediction.

- **For `multiple_simulations_joined_pcs`**:
  - A list of entries, each containing:
    - The weight configurations used.
    - The R² metrics as above.

**Understanding R² Metrics**:

- **`"video"` R²**: Measures how well the video space predicts the target space (e.g., `dst`).
- **`"cs"` R²**: Measures cross-space prediction (e.g., how well `src` predicts `dst`).
- **`"residual"` R²**: Measures prediction after removing the influence of the video (i.e., residual prediction).

**Analyzing Results**:

- Higher R² values indicate better predictive performance.
- Comparing R² metrics across different configurations helps identify which subspace contributions and configurations yield the best predictions.

## Additional Notes

- **Logging**: You can enable logging in `multiple_simulations_joined_pcs` by passing a logger to the method for detailed output.
- **Custom Weight Functions**: You can add custom weight functions in the `Simulation` class's `weight_functions` dictionary.
- **Extensibility**: The framework is designed to be extensible. You can modify or add new simulation methods as needed.
