# GPR

This repository contains a small MATLAB example for Gaussian Process Regression (GPR) applied to global temperature data.

Current contents
- `untitled.m`
  - A MATLAB script that loads a variable named `temperature` from the workspace (or from `temperature.mat`) and runs a GPR prediction/plot.
  - The script contains three functions: `SqrExp1`, `SqrExp2`, and `Optm` which implement the squared-exponential covariance, the covariance with noise, and a simple hyperparameter optimizer (gradient-ascent on the log-likelihood), respectively.
  - The script produces a plot of the predictive mean and a 95% confidence interval for the example run.
- `temperature.mat`
  - A MAT-file used by `untitled.m`. The script expects `temperature` to be a numeric array and uses `temperature(:,2)` as the temperature series.
- `Predicting World Temperature with GPR-Siqi Li.mp4`
  - Presentation / demonstration video.
- `Predicting World Temperature with Gaussian Process Regression, Siqi Li.jpg`
  - Image asset (slide or figure).

Requirements
- MATLAB (no specific release required in the files, but modern MATLAB versions are recommended).
- The repository does not currently require any additional external MATLAB toolboxes to run `untitled.m` as-is, but if you modify the code you may choose to use toolboxes (not included or assumed here).

Quickstart — run the included script
1. Clone the repository:
   ```bash
   git clone https://github.com/LSQamI/GPR.git
   cd GPR
   ```

2. Start MATLAB and add the repository to the path:
   ```matlab
   addpath(genpath(pwd));  % from repository root
   ```

3. Load the data and run the script:
   ```matlab
   load('temperature.mat');  % this should create a variable named `temperature`
   run('untitled.m');
   ```

Notes about how `untitled.m` expects the data
- The script references `temperature(:,2)` as the temperature values. Ensure `temperature` is a numeric array with at least 2 columns and that the second column contains the temperature values (e.g., anomaly or average).
- The script sets variables such as `today`, `train_day`, and `test_day` at the top — edit these values before running to change the training/test split or prediction horizon.

What the script does (concise)
- Centers the training target by subtracting its mean.
- Calls `Optm` to (iteratively) optimize hyperparameters `sigma_f` and `rho` (and uses a fixed `sigma_n`).
- Constructs covariance matrices with the squared-exponential kernel and computes the predictive mean and variance using matrix operations.
- Plots predicted mean and a shaded 95% confidence band together with training observations.
