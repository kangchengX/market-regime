<p align="center">
    <h1 align="center">MARKET-REGIMES</h1>
</p>
<p align="center">
    <em>Extensive study on regime identification in financial markets.</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
   <img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="Pandas">
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="PyTorch">	
   <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikit-learn">
   <img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=default&logo=SciPy&logoColor=white" alt="SciPy">
   <img src="https://img.shields.io/badge/seaborn-2C6E91.svg?style=default&logo=Seaborn&logoColor=white" alt="seaborn">
</p>

<br>
<details open>
  <summary>Table of Contents</summary><br>

- [Introduction](#introduction)
   - [CFM](#regime-identification-based-on-correlation-features-cfm)
   - [DFM](#regime-identification-based-on-deep-features-dfm)
   - [DCFM](#concatenation-of-deep-features-and-correlation-features-dcfm)
   - [EEM](#end-to-end-regime-identification-eem)
- [Data](#data)
- [Example Results](#example-results)
   - [A Representative Instance of DFM](#a-representative-instance-of-dfm)
   - [An Ablation Study for DFM](#an-ablation-study-for-dfm)
- [Repository Structure](#repository-structure)
- [Modules](#modules)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data download](#data-download)
  - [Usage](#usage)
- [Acknowledgments](#acknowledgments)
</details>
<hr>

##  Introduction

This project investigates various statistical, machine learning, and deep learning methodologies
for regime identification in financial markets. This research is conducted in collaboration with Insight
Investment, involving a series of experiments and extensive ablation studies. We explore four
different unsupervised models to identify patterns and trends from multiple indices belonging to
different assets, providing insights into designing regime identification frameworks and approaches
for time series representation learning with a focus on the local structures. Notably, the model
using pure CNN AutoEncoder without recurrent connection effectively detects the 2008 Global
Financial Crisis and 2019 COVID-19 periods.

The overview of the four models can be seen in the table below:

| Model         | Model Structure | Deep Module Structure |
|---------------|-----------------|-----------------------|
| CFM (C-H)     | two-stage       | N/A                   |
| DFM           | two-stage       | CNN AutoEncoder       |
| DCFM          | two-stage       | CNN AutoEncoder       |
| EEM           | end-to-end      | Siamese CNN           |


### Regime Identification Based on Correlation Features: CFM

In this Correlation Feature-
Based Model (CFM) explored in this chapter, we use correlations between price changes of
financial indices as features, applying variations with meta and cophenetic similarity measures,
combined with k-means++ and hierarchical clustering algorithms. CFM effectively detects
stable and predictable regime structures, successfully identifying significant events like the
2008 Global Financial Crisis (GFC). However, it struggles to capture deeper, more complex
market patterns due to its insufficiency of correlation features to capture market trends.

The model structure can be seen in the figure below:

![CFM structure](/example%20images/CFM.png)

### Regime Identification Based on Deep Features: DFM
In this Deep Feature-Based Model
(DFM) explored in this chapter, we use a CNN AutoEncoder to extract deep features from
multivariate financial time series data. DFM excels in detecting specific market states and
capturing complex behaviours, more effectively identifying events like the GFC and COVID-
19 periods than CFM. It identifies shorter and more frequent regime transitions, highlighting
sensitivity to short-term market signals. Despite its effectiveness, DFM might introduce more
noise and instability than CFM, reflected by higher entropy and shorter regime durations.

The model structure can be seen in the figure below:

![DFM structure](/example%20images/DFM.png)

### Concatenation of Deep Features and Correlation Features: DCFM

In this Deep-Correlation
Feature-Based Model (DCFM) explored in this chapter, we combine deep features from the
CNN AutoEncoder with correlation features to achieve a trade-off between short-term and
long-term trends, while reducing the noise and enhancing the detection and differentiation of
market states. However, it does not exhibit the expected superior performance. The regime
structure resembles that of CFM more than DFM, indicating the dominance of correlation
features and the failure to leverage deep features’ deep and complex understanding of market
structures.

The model structure can be seen in the figure below:

![DCFM structure](/example%20images/DCFM.png)


### End-to-End Regime Identification: EEM

In this End-to-End Model (EEM) explored in this
chapter, we use a Siamese CNN to directly identify market regimes without a separate clustering
stage. However, EEM faces a significant class collapse problem, favouring only a subset of
regimes due to the inability to distinguish market states effectively. It overfits input data while
underfitting underlying patterns, leading to less meaningful regime structures. Incorporating
inverse entropy loss helps to mitigate the issue, but challenges persist.

The model structure can be seen in the figure below:

![EEM structure](/example%20images/EEM.png)

---
## Data

The data used in this project was sourced from Bloomberg and covers the earliest available record
up to June 11, 2024, with a daily frequency. The names of indices and columns we collected are
shown in the table below:

| **Index**          | **Price** | **Daily Total Return** | **Description**                           | **Asset Class**      |
|--------------------|-----------|------------------------|-------------------------------------------|----------------------|
| **CSI BARC Index** | ✓         |                        | Barclays Credit Spread Index              | Credit Spread        |
| **DXY Curncy**     | ✓         |                        | US Dollar Index                           | Currency             |
| **MXWO Index**     | ✓         | ✓                      | MSCI World Equity Index                   | Equities             |
| **SPGSIN Index**   | ✓         |                        | S&P GS Industrial Metals Index            | Commodities          |
| **SPX Index**      |           | ✓                      | S&P 500 Index                             | Equities             |
| **USGG10YR Index** | ✓         |                        | US 10 year Gov Bond Yield                 | Interest Rates       |
| **VIX Index**      | ✓         |                        | VIX Index                                 | Volatility           |
| **XAU Curncy**     | ✓         |                        | Gold Price                                | Commodities          |


---

## Example Results
### A Representative Instance of DFM
<img src="./example images/regimes_vix.png" alt="regimes_vix" />
<img src="./example images/returns_ana_within_regimes.png" alt="returns" />
<p align="center">
<img src="./example images/transition matrix regime level.png" alt="transition" width="301"/>
<img src="./example images/durations.png" alt="duration" width="493"/>
</p>

### An Ablation Study for DFM

<img src="./example images/features-fixed-entropy.png" alt="entropy" />
<img src="./example images/features-fixed-median.png" alt="median" />
<img src="./example images/features-fixed-silhouette_cluster.png" alt="silhouette" />

---

##  Repository Structure

```sh
└── market-regimes/
    ├── cluster.py
    ├── constant.py
    ├── correlation.py
    ├── data.py
    ├── example images
    ├── flow
    │   ├── cluster_assess_flow.py
    │   ├── end_to_end_flow.py
    │   ├── feature_concat_flow.py
    │   ├── similarity_generate_flow.py
    │   └── train_flow.py
    ├── loss.py
    ├── main.py
    ├── networks.py
    ├── preprocess.py
    ├── process.py
    ├── README.md
    ├── requirements.txt
    ├── stats.py
    ├── utils.py
    └── visualization.py
```

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [cluster.py](cluster.py)             | This script contains all the clustering stuff as well as the metrics and methods to evaluate the clustering results.|
| [constant.py](constant.py)           | `constant.py` contains constants for this project, including neural network module configurations, data column mappings for returns, and custom color palettes for data visualization.  |
| [correlation.py](correlation.py)     | Calculates and analyzes rolling correlation matrices among indices using multiple statistical methods. It constructs similarity matrices from these correlations using both cophenetic and meta-methodologies, enhancing the understanding of inter-temporal index behaviors potentially crucial for regime detection in financial markets. |
| [data.py](data.py)                   | Transforms multivariate time series data into images and correlation matrices, including all the dataset classes.|
| [loss.py](loss.py)                   | `loss.py` defines KLDivergenceLoss, integrating KL divergence with optional L2 and the inverse entropy regularization |
| [main.py](main.py)                   | Execute all the experiments in this project.|
| [networks.py](networks.py)           | Implements all the deep learning modules including CNN AutoEncoder and the Siamese CNN.|
| [preprocess.py](preprocess.py)       | This script is for the preprossing of the raw data.|
| [process.py](process.py)             | Provides the core functionalities for model training, inference, and feature extraction. It defines workflow classes crucial for regime identification and analysis in financial data series. |
| [stats.py](stats.py)                 | Provides comprehensive statistical analyses and metrics essential for assessing financial regime durations, transitions, and returns across different timeframes, contributing to the analysis of the regime identification results.|
| [utils.py](utils.py)                 | The `utils.py` file in the market-regimes project serves as a utility module, providing support functions across the application. This module plays a critical role by handling common functionalities like file operations, data parsing, and functions to manage the numerous result folders.|
| [visualization.py](visualization.py) | This contains all the visualization in this project, including those for a single model and a summary of a ablation study|

</details>

<details open><summary>flow</summary>
This directory contains all the workflows in this project.

| File                                 | Summary |
| ---                                  | --- |
| [cluster_assess_flow.py](flow\cluster_assess_flow.py)           | This is for the clustering based on the extracted features, as well as the analysis of the regime identification results.|
| [end_to_end_flow.py](flow\end_to_end_flow.py)                   | This is for the end-to-end regime identification, including the analysis of the regime identification results.|
| [feature_concat_flow.py](flow\feature_concat_flow.py)           | This is for the concatenation of the deep features and the correlation features. |
| [similarity_generate_flow.py](flow\similarity_generate_flow.py) | Generates similarity matrices, i.e., features based on correlations.|
| [train_flow.py](flow\train_flow.py)                             | Generates deep features. |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.11`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the repository:
>
> ```console
> $ git clone https://github.com/kangchengX/market-regime.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd market-regime
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

### Data Download
According to the company's policy, the raw data is not allowed to public. However, the same data can be obtained through the steps described in Section [Data](#data).

###  Usage

<h4>From <code>source</code></h4>

> Use the command below:
> ```console
> $ python main.py
> ```

---

##  Acknowledgments

Thanks for the help from the Senior Portfolio Manager, Zacharias Bobolakis, and the Senior VP and Quantitative Researcher, Mauricio Bouabci at *Insight Investment*, as well as Professor Philip Treleaven at *UCL*.

[**Return**](#overview)

---
