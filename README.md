# GPR — Gaussian Process Regression for World Temperature Prediction

[![MATLAB](https://img.shields.io/badge/Language-MATLAB-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Materials, code, and examples for modeling and predicting global temperature time series using Gaussian Process Regression (GPR) implemented in MATLAB.

Table of contents
- Overview
- Features
- Requirements
- Repository layout
- Getting started (quick start)
- Data: source & preparation
- Experiments & running scripts
- Configuration and common options
- Output and expected results
- Visualizations & notebooks
- Reproducibility
- Contributing
- License
- Citation & contact
- Troubleshooting & FAQ

Overview
--------
This repository contains MATLAB code for applying Gaussian Process Regression to global temperature datasets. It includes preprocessing, model training, evaluation, kernel experiments, and plotting utilities to reproduce results and figures. The code is intended for researchers and students exploring non-parametric regression for climate/temperature time series.

Features
--------
- Preprocessing pipeline for raw temperature datasets
- GPR implementations using:
  - MATLAB's built-in fitrgp / predict (Statistics and Machine Learning Toolbox), and/or
  - GPML toolbox (Rasmussen & Williams) integration for advanced kernels (optional)
- Support for multiple kernels (RBF, Matern, spectral mixture, composite kernels)
- Scripts to train, cross-validate, and evaluate (RMSE, MAE, NLL)
- Plotting utilities for time-series, predictive intervals, and kernel visualisations
- Example experiments and Live Scripts (.mlx) to reproduce figures

Requirements
------------
- MATLAB R2019b or later recommended
- Required MATLAB toolboxes:
  - Statistics and Machine Learning Toolbox (for fitrgp, fitrgp-related utilities)
  - Optimization Toolbox (recommended for custom hyperparameter optimization)
  - Signal Processing Toolbox (optional, for spectral preprocessing)
- Optional: GPML toolbox (for advanced GP kernels and inference) — http://www.gaussianprocess.org/gpml/code/matlab/
- Disk space to store datasets (see Data section)

If you do not have MATLAB, parts of the preprocessing might run in GNU Octave but GPR implementations rely on MATLAB toolboxes or GPML; Octave is not officially supported.

Repository layout
-----------------
(Adjust this if your repo uses different folders — this is the recommended layout.)
- README.md                — This file
- LICENSE                  — License file
- data/
  - raw/                   — Raw source files (gitignored; small placeholders allowed)
  - processed/             — Preprocessed datasets (gitignored)
- src/
  - preprocess.m           — Data cleaning and resampling functions
  - features.m             — Feature extraction utilities (e.g., seasonality, trends)
  - gpr_train.m            — High-level wrapper to train a GPR model
  - gpr_predict.m          — Wrapper for making predictions & intervals
  - kernels/               — Custom kernel implementations (if any)
- scripts/
  - download_data.m        — Script to download source datasets (if applicable)
  - quick_start.m          — Minimal end-to-end example (download→prepare→train→plot)
  - run_experiment.m       — Script to run a named experiment/config
- notebooks/ or live/      — MATLAB Live Scripts (.mlx) with examples and figures
- configs/                 — Example experiment configuration files (.m or .mat)
- outputs/                 — Saved model checkpoints, logs, and plots (gitignored)
- docs/                    — Additional documentation, papers, reproducibility notes
- requirements.txt         — Optional file listing MATLAB toolboxes & versions (informational)

Getting started — Quick start
-----------------------------
1. Clone the repo:
```bash
git clone https://github.com/LSQamI/GPR.git
cd GPR
```

2. Open MATLAB and add the project to the path:
```matlab
addpath(genpath(pwd));  % from repository root
savepath;
```

3. Run the quick example (interactive):
- Open `scripts/quick_start.m` in MATLAB and run, or in the MATLAB command window:
```matlab
run('scripts/quick_start.m')
```

4. Run non-interactively from a shell (headless):
```bash
matlab -batch "addpath(genpath(pwd)); run('scripts/quick_start.m')"
```

Data — source & preparation
---------------------------
The repository does not (by default) include large raw datasets. Recommended data sources:
- NOAA Global Surface Temperature datasets (GISTEMP) — https://data.giss.nasa.gov/gistemp/
- HadCRUT — https://www.metoffice.gov.uk/hadobs/hadcrut5/
- Berkeley Earth — http://berkeleyearth.org/data/

Place raw data files into:
- data/raw/

Run the provided download/prepare scripts:
```matlab
% Download (if script available)
run('scripts/download_data.m')

% Preprocess
run('src/preprocess.m');  % or run a wrapper that writes to data/processed/
```

Preprocessing steps include:
- Parsing CSV / NetCDF inputs
- Aggregating to monthly / annual time series
- Filling small missing gaps with interpolation
- Normalization or detrending (optional, configurable)

Experiments & running scripts
-----------------------------
Experiments are configured in `configs/`. Each config may be a MATLAB `.m` file that defines a struct `cfg` with fields:
- cfg.name             — experiment name
- cfg.data.file        — path to processed data
- cfg.model.kernel     — kernel type ('RBF', 'Matern', 'SpectralMixture', 'Composite', etc.)
- cfg.model.optimize   — true/false whether to optimize hyperparameters
- cfg.train.split      — train/validation/test split (e.g., [0.7, 0.15, 0.15])
- cfg.seed             — random seed for reproducibility
- cfg.output.dir       — where to save models and plots

Example: run an experiment:
```matlab
% From MATLAB
cfg = configs/example_experiment();   % if configs are functions returning cfg
run_experiment(cfg);

% Or using a script wrapper
run('scripts/run_experiment.m')  % takes experiment name or path to config
```

Training and evaluation functions
- src/gpr_train.m: accepts data and cfg, trains a GPR model, returns a model structure
- src/gpr_predict.m: accepts model + test data, returns mean predictions and prediction intervals
- src/evaluate.m: computes RMSE, MAE, log-likelihood, and calibration diagnostics

Configuration and common options
-------------------------------
- Kernel selection: RBF (squared exponential), Matern (nu = 3/2 or 5/2), Spectral Mixture (for complex seasonality)
- Likelihood noise model: Gaussian noise with learnable variance
- Hyperparameter optimization: use built-in optimization (fitrgp) or custom optimization routines
- Cross-validation: time-series-aware splits (rolling-window, blocked CV)

Output & expected results
-------------------------
When an experiment completes, results are saved to `outputs/<experiment-name>/`:
- model.mat      — saved trained model and metadata
- metrics.json   — RMSE, MAE, NLL, etc.
- plots/         — PNG/PDF visualizations: time series, predicted mean + 95% CI, residuals
- logs/          — console output and optimization traces

Example expected results (illustrative):
- RBF kernel: RMSE ≈ 0.1°C on held-out yearly-averaged series
- Composite kernel (RBF + periodic) captures seasonal components and reduces NLL

Visualizations & interactive notebooks
-------------------------------------
Include MATLAB Live Scripts (.mlx) in `notebooks/` or `live/`:
- notebooks/quick_start.mlx — walkthrough (load data, train a small model, display plots)
- notebooks/results.mlx — reproducible generation of figures used in reports

Reproducibility
---------------
- Seed your experiments: set rng(cfg.seed) at the start of any run
- Save full config (cfg struct) alongside outputs so experiments are fully reproducible
- Use `requirements.txt` or `docs/DEPENDENCIES.md` to list exact MATLAB release and toolbox versions

Contributing
------------
Contributions are welcome. Please:
1. Open an issue to discuss features/bug fixes before major work.
2. Fork the repo and submit a pull request with clear description and tests (if applicable).
3. Keep commits small and focused; update documentation and examples if behavior or interface changes.

Suggested workflow:
- Create a branch feature/your-feature
- Add unit-like checks (e.g., small test scripts)
- Update notebooks/examples if behavior or interface changes

License
-------
This repository is licensed under the MIT License — see LICENSE for details. (If you prefer a different license, replace LICENSE accordingly.)

Citation & contact
------------------
If you use this work, please cite the repository and any associated paper:
- Author(s). "GPR — Gaussian Process Regression for World Temperature Prediction", Year. GitHub: https://github.com/LSQamI/GPR

Contact / maintainer:
- GitHub: https://github.com/LSQamI
- (Optionally add an email or affiliation)

Troubleshooting & FAQ
---------------------
Q: I get an error about missing fitrgp or other functions.
A: Ensure Statistics and Machine Learning Toolbox is installed. Check with `ver`.

Q: Can I run this on GNU Octave?
A: Most GPR code uses MATLAB toolboxes and may not run in Octave. You can try data preprocessing steps, but model training likely requires MATLAB or GPML adapted for Octave.

Q: Where do I change kernel options / hyperparameters?
A: Edit the config in `configs/` for the experiment and re-run `scripts/run_experiment.m`.

Q: Results look too confident (narrow intervals)
A: Check noise model, re-run with more flexible kernel or Bayesian hyperparameter estimation. Try leaving noise variance free and regularizing hyperparameter search.

Optional improvements to add to the repository
---------------------------------------------
- Add a `Dockerfile` or MATLAB runtime instructions for CI
- Add GitHub Actions to run quick smoke tests (requires MATLAB license on runners — consider using unit-only tests)
- Add `examples/` with small synthetic datasets for fast testing
- Add `CONTRIBUTING.md` and a `CODE_OF_CONDUCT.md`

Acknowledgements & references
-----------------------------
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
- GPML toolbox: http://www.gaussianprocess.org/gpml/code/matlab/
