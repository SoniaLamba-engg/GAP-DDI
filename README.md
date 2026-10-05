## GAP-DDI: Predicting Drug-Dusease Associations

This repository has been created to hold the source code for the following paper
## "GAP-DDI: Leveraging Graph Attention and Low-Rank Optimization for Accurate Drug–Disease Interaction Discovery"

## About
GAP-DDI is a graph-based framework for predicting novel drug–disease associations to support drug repurposing. It integrates a Graph Convolutional Network (GCN) with an attention mechanism and Parallel Proximal Optimization (PPXA) to learn from sparse, multi-relational biomedical data.
By combining multi-source similarities and known interactions, GAP-DDI captures both local and global patterns, achieving superior predictive performance and uncovering clinically relevant drug indications validated by CTD.

## Manuscript version  
- Repository: https://github.com/SoniaLamba-engg/GAP-DDI  
- Release/Tag: v1.0-manuscript
- Commit: 48fc69eecd030b2b48b27516d4832a748aa05d10

## Environment
- MATLAB R2024a 
- Python 3.6.8. The required packages are as follows:
numpy == 1.15.4
scipy == 1.1.0
tensorflow == 1.12.0

## Dataset Description
This repository contains two dataset folders: data_raw and data_processed.

**1. data_raw/ – Original Input Data**
This directory contains the original biological datasets used to construct the heterogeneous drug–disease graph and to compute the node embeddings. Each dataset appears in three variants: C, F, and Main, following the naming conventions used in the manuscript. For each dataset, the following files are provided:

**DiDrA.txt**: Known binary drug–disease associations.
**DrugSim.txt**: Precomputed drug–drug similarity matrix.
**DiseaseSim.txt**: Precomputed disease–disease similarity matrix.

**2. data_processed/ – Embedding-Based Feature Data**
This directory contains the processed features generated after the embedding stage and cosine similarity computation. For each dataset (C, F, Main), the folder includes:

**DiDrA.txt**: Known binary drug–disease associations.
**DrugSim.txt**: Cosine drug–drug similarity matrix.
**DiseaseSim.txt**: Cosine disease–disease similarity matrix.

## Pre-processing
1. Similarity normalisation for drugs and diseases.
2. Heterogeneous adjacency construction (Eq. 1 and 3 of the paper), μ = 6.
3. GCN encoder: B = 64, L = 3, layer attention α_l = 1/(l+1).
4. Cosine similarity of embeddings, then p = 5 nearest-neighbour sparsification and normalised graph Laplacians.
5. Missing/unknown pairs are treated as negatives; no extra negative sampling.

## Implementation
**Stage 1 – Feature learning**
- Step 1: construct the heterogeneous network from similarities and known associations.
- Step 2: attention-guided GCN learns drug/disease embeddings; feature vectors are
  compared by cosine similarity.

**Stage 2 – Optimization and prediction**
- Step 3: PPXA with graph-Laplacian regularization reconstructs the association
  matrix (θ = 5, 20 iterations, X ∈ [0, 1]).

## Reproducing the main results
Random seed: **42** (`rng(42)` at the top of `<script>`); folds are stratified over positive associations.

## Reproducing the Main Results

The experiments corresponding to the submitted manuscript can be reproduced using the code at the following version:

**Step 1: Feature Learning**

Navigate to the directory containing the Python implementation and run:

-python main.py

This step performs the feature-learning stage, including heterogeneous graph construction, attention-guided GCN embedding generation, and cosine-similarity computation.

**Step 2: Optimization and Prediction**

Open MATLAB R2024a, set the repository root as the working directory, and run:

-main2.m

This step performs PPXA-based optimization with graph-Laplacian regularization and generates the predicted drug–disease association scores.

**Reproducibility Settings**

The main experiments use a fixed random seed of 42. In MATLAB, the random number generator is initialized using:

rng(42);

The evaluation uses stratified folds over positive drug–disease associations. Missing/unknown drug–disease pairs are treated as negatives, and no additional negative sampling is performed.

**The hyperparameters used for the manuscript experiments are fixed as follows:**

GCN embedding dimension: B = 64
Number of GCN layers: L = 3
Layer attention: α_l = 1/(l+1)
Heterogeneous graph parameter: μ = 6
Nearest-neighbour sparsification: p = 5
PPXA parameter: θ = 5
PPXA iterations: 20
Prediction range: X ∈ [0,1]
Evaluation: 10-fold cross-validation
