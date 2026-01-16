## physDL_XAJ: Differentiable Xin'anjiang Hydrological Model

This repository implements a PyTorch-based Xin'anjiang (XAJ) hydrological model and a differentiable parameter learning framework. Recurrent neural networks (RNNs) infer XAJ parameters from static catchment attributes, which are then used in a vectorized, differentiable XAJ model for streamflow simulation.

---

### 1. Main Features

- **XAJ hydrological model**
  - Static-parameter XAJ model: `hydroMLlib/model/xaj_static.py`
  - Dynamic-parameter XAJ model: `hydroMLlib/model/xaj_dynamic.py`
  - Supports multi-component parameterization and river routing.

- **Deep-learning-based parameter inversion**
  - RNN / CNN-LSTM implementations in `hydroMLlib/model/dLmodels.py`
  - Training loops and model assembly in `hydroMLlib/model/train.py` and the `experiments` folder.

- **Data utilities**
  - CAMELS-style data loading and preprocessing: `hydroMLlib/utils/camels.py`
  - Evaluation metrics and loss functions: `hydroMLlib/utils/metrics.py`, `hydroMLlib/utils/criterion.py`
  - File and configuration utilities: `hydroMLlib/utils/fileManager.py`, `hydroMLlib/utils/defaultSetting.py`, `hydroMLlib/utils/time.py`

---

### 2. Repository Structure (Brief)

```text
physDL_XAJ/
  hydroMLlib/
    model/
      xaj_static.py
      xaj_dynamic.py
      train.py
      rnn.py
      dLmodels.py
      hydroRouting.py
    utils/
      camels.py
      criterion.py
      metrics.py
      fileManager.py
      defaultSetting.py
      time.py
  experiments/
    trainXAJ-Static.py
    testXAJ-Static.py
    trainXAJ-Dynamic.py
    testXAJ-Dynamic.py
    save_results_plot.py
  requirements.txt
  README.md
```

---

### 3. Environment and Installation

```bash

# (optional) create a virtual environment
conda create -n physDL_XAJ python=3.10
conda activate physDL_XAJ

# install dependencies
pip install -r requirements.txt
```

Python ≥ 3.8 and PyTorch ≥ 1.10 are recommended (adjust the CUDA-enabled build to your system).

---

### 4. Data Preparation (Brief)

1. Prepare a CAMELS or CAMELS-like basin dataset.  
2. Set data root paths and output directories in:
   - `hydroMLlib/utils/fileManager.py`
   - `hydroMLlib/utils/defaultSetting.py`
3. Use functions in `hydroMLlib/utils/camels.py` to test that several basins can be loaded without errors.

---

### 5. Quick Start

#### 5.1 Train the static XAJ model

```bash
cd experiments
python trainXAJ-Static.py
```

Hyperparameters (time periods, basin list, network architecture, learning rate, etc.) can be modified in `trainXAJ-Static.py` and `hydroMLlib/model/train.py`.

#### 5.2 Test the static XAJ model

```bash
python testXAJ-Static.py --ckpt PATH_TO_CHECKPOINT.pt
```

The script runs simulations over the test period specified inside the script, computes metrics using `metrics.py`, and saves results/figures to paths configured in the code.

#### 5.3 Dynamic XAJ model

Training:

```bash
python trainXAJ-Dynamic.py
```

Testing:

```bash
python testXAJ-Dynamic.py --ckpt PATH_TO_CHECKPOINT.pt
```

---

