# AI/ML Flood Risk Classification – Simplified PoC



## Overview



This project is a simplified proof-of-concept inspired by the Hydro-TransformerNet research paper for urban flood susceptibility assessment.



The assessment permits the use of a public or synthetic dataset. This implementation uses a tabular India flood-risk dataset and constructs a synthetic flood-risk target using hydrologically motivated features.



This is not a reproduction of the original Sharjah, UAE Hydro-TransformerNet dataset or architecture.



\## Dataset



Dataset: `flood\_risk\_dataset\_india.csv`



The dataset contains 10,000 samples with environmental, geographic, hydrological, and categorical features.



Important features include:



\- Rainfall

\- River Discharge

\- Water Level

\- Elevation

\- Historical Floods

\- Temperature

\- Humidity

\- Latitude

\- Longitude

\- Land Cover

\- Soil Type

\- Population Density

\- Infrastructure



A synthetic flood-risk target was created using rainfall, river discharge, water level, inverse elevation, and historical flood information.



## Approach



Pipeline:



\*\*Data → Preprocessing → CNN Feature Extraction → Self-Attention → Prediction → Evaluation\*\*



A Random Forest model is also trained as a classical machine-learning baseline.



## Preprocessing



\- Categorical features are one-hot encoded.

\- Numerical features are standardized using StandardScaler.

\- Data is divided into training and testing sets.

\- PyTorch tensors are created for the neural-network models.



\## Models



\### 1. CNN



A 1D CNN is used to extract local feature patterns from the tabular feature representation.



\### 2. CNN + Self-Attention



The CNN feature representation is passed through a multi-head self-attention layer to model relationships between features.



\### 3. Random Forest



Random Forest is included as a traditional machine-learning baseline.



\## Results



| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |

|---|---:|---:|---:|---:|---:|

| CNN | 0.7865 | 0.7885 | 0.7830 | 0.7858 | 0.8783 |

| CNN + Self-Attention | 0.7965 | 0.7698 | 0.8460 | 0.8061 | 0.8833 |

| Random Forest | 0.8285 | 0.8269 | 0.8310 | 0.8289 | 0.9153 |



The Random Forest baseline achieved the highest test metrics in this simplified tabular experiment. The CNN + Self-Attention model improved recall and F1 compared with the CNN alone.



## Output Files



### `outputs/figures/`



\- `confusion\_matrices.png`

\- `feature\_importance.png`

\- `roc\_curves.png`

\- `training\_loss.png`



### `outputs/models/`



\- `cnn\_model.pth`

\- `cnn\_attention\_model.pth`

\- `random\_forest\_model.pkl`

\- `scaler.pkl`



### `outputs/results/`



\- `model\_comparison\_results.csv`

\- `feature\_importance\_results.csv`



## How to Run



Install dependencies:



```bash

pip install -r requirements.txt

Run:



python main.py



The script can also accept a CSV path:



python main.py --csv flood\_risk\_dataset\_india.csv

Technology Stack

Python

PyTorch

Scikit-learn

Pandas

NumPy

Matplotlib

Joblib

Research Reference



The implementation is conceptually inspired by:



"Urban Flood Susceptibility Assessment in Arid Environment Using a Novel Hybrid Deep Learning Approach"



The research architecture uses spatial feature extraction followed by Transformer-based modelling and hydro-aware attention.



Limitations

The dataset is tabular rather than geospatial raster data.

The target is synthetic and should not be interpreted as historical flood observations.

The implementation is a simplified proof-of-concept.

It does not reproduce the complete Hydro-TransformerNet architecture.

Results should not be directly compared with results from the research paper.

Future Work

Use real historical flood masks.

Use aligned geospatial raster layers.

Implement a full ResUNet/ResNet-based spatial encoder.

Incorporate geospatial Transformer modelling.

Add explainability methods such as SHAP.

Evaluate on geographically independent test regions.

Author



Supriya M



AI/ML PoC – Code-X-Novas Technical Assessment

