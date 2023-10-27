<br />
<p align="center">
  <a href=" ">
    <img src="xai/images/misc/logo.png" alt="Logo" width="600"> 
  </a>

  <h1 align="center">Discovering Process Dynamics for Scalable Perovskite Solar Cell Manufacturing with Explainable AI</h1>

  <p align="center">
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9-3776AB?&logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12-EE4C2C?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Pytorch Lightning 2.0-792EE5?logo=pytorchlightning&logoColor=white"></a>
    <a href="https://black.readthedocs.io/en/stable"><img alt="L: Hydra" src="https://img.shields.io/badge/Code Style-Black-black" ></a>
    <br>
    <a href="https://onlinelibrary.wiley.com/journal/15214095"><strong>Read the paper »</strong></a>
    <br />
  </p>
</p>

This study uses explainable AI to analyze photoluminescence data from perovskite thin-film processing, linking it to solar cell performance, and offering actionable insights for scalable solar cell manufacturing.

> Abstract: <br>*Large-area processing of perovskite semiconductor thin-films is complex and evokes unexplained variance in quality, posing a major hurdle for the commercialization of perovskite photovoltaics. Advances in scalable fabrication processes are currently limited to gradual and arbitrary trial-and-error procedures. While the in-situ acquisition of photoluminescence videos has the potential to reveal important variations in the thin-film formation process, the high dimensionality of the data quickly surpasses the limits of human analysis. In response, this study leverages deep learning and explainable artificial intelligence (XAI) to discover relationships between sensor information acquired during the perovskite thin-film formation process and the resulting solar cell performance indicators, while rendering these relationships humanly understandable. We further show how gained insights can be distilled into actionable recommendations for perovskite thin-film processing, advancing towards industrial-scale solar cell manufacturing. Our study demonstrates that XAI methods will play a critical role in accelerating energy materials science.*

<br>

<p align="center">
  <img src="xai/images/misc/drying.gif" width="250"> 
</p>

## 📝&nbsp;&nbsp;Citing this Work

If you use perovskite-xai please cite our [paper]()

```bibtex
@inproceedings{}
```

## 🧭&nbsp;&nbsp;Table of Contents
* [Installation](#Installation)
* [Project Structure](#project-structure)
* [Dataset](#dataset)
* [Reproducing the Results](#reproducing-the-results)
* [Acknowledgements](#acknowledgements)

## ⚙️&nbsp;&nbsp;Installation

perovskite-xai requires Python version 3.9 or later. All essential libraries for the execution of the code are provided in the `requirements.txt` file from which a new environment can be created (Linux only):

```bash
git clone https://github.com/IML-DKFZ/repo
cd perovskite-xai
conda create -n perovskite-xai python=3.9
source activate perovskite-xai
pip install -r requirements.txt
```

Depending on your GPU, you need to install an appropriate version of PyTorch and torchvision separately. All scripts run also on CPU, but can take substantially longer depending on the experiment. Testing and development were done with the Pytorch version using CUDA 11.6. 

## 🗃&nbsp;&nbsp;Project Structure

<p align="center">
  <img src="xai/images/misc/overview.png" width="650"> 
</p>

```text
├── base_model.py                   -               
├── data                            -
│   ├── add_new_labels.py           -
│   ├── augmentations               -
│   ├── cv_splits.py                -
│   ├── perovskite_dataset.py       -
│   ├── preprocessing.py            -
│   └── split_data.py               -
├── main.py                         -
├── models                          -
│   ├── resnet.py                   -
│   └── slowfast.py                 -
├── predict_from_checkpoint.py      -
├── predict_testset.py              -
├── preprocessed                    -
├── README.md                       -
├── requirements.txt                -
├── utils.py                        -
└── xai                             
    ├── images                      - Figure export location
    ├── results                     - Interim data export location
    ├── src                         
    │   ├── attribution             - Attribution computation & visualization
    │   ├── counterfactuals         - CF computation & visualization
    │   ├── evaluation              - Evaluation computation & visualization
    │   └── tcav                    - TCAV computation & visualization
    ├── utils                       - util scripts for evaluation
    └── util_vis                    
        ├── util_error_vis.ipynb    - Residual and parity plots
        └── util_vis.ipynb          - Various paper figures
```
## 💾&nbsp;Dataset
<p align="center">
  <img src="xai/images/misc/representations.png" width="550">
  Representations
</p>


## ♻️&nbsp;Reproducing the Results
### 🚀&nbsp;Model Training
### 🔎&nbsp;XAI Computation
### 📊&nbsp;XAI Evaluation

## 📣&nbsp;&nbsp;Acknowledgements

The code is developed by the authors of the paper. However, it does also contain pieces of code from the following packages:

- Captum by Kokhlikyan, Narine et al.: https://github.com/pytorch/captum
- DiCE by Mothilal, Ramaravind K. et al.: https://github.com/interpretml/DiCE
- Image Classification by Ziegler, Sebastian: https://github.com/MIC-DKFZ/image_classification

____

<br>

<p align="center">
  <img src="https://polybox.ethz.ch/index.php/s/I6VJEPrCDW9zbEE/download" width="190"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="91"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.in-part.com/thumbnail?stripmeta=true&noprofile=true&quality=95&url=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fassets.in-part.com%2Funiversities%2F227%2FGdzZTi4yThyBhFzWlOxu_DKFZ_Logo-3zu-Research_en_Black-Blue_sRGB.png&width=750" width="120"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/3a/Logo_KIT.svg" width="180">  
</p>

perovskite-xai is developed and maintained by the Interactive Machine Learning Group and the Applied Computer Vision Lab of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the [DKFZ](https://www.dkfz.de/de/index.html), as well as the Light Technology Institute of the [Karlsruhe Institute of Technology](https://www.lti.kit.edu/index.php).