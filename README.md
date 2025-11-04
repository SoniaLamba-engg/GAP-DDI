**GAP-DDI**
Predicting Drug-Dusease Associations

This repository has been created to hold the source code for the following paper
**"GAP-DDI: Leveraging Graph Attention and Low-Rank Optimization for Accurate Drug–Disease Interaction Discovery"**

**About**
GAP-DDI is a graph-based framework for predicting novel drug–disease associations to support drug repurposing. It integrates a Graph Convolutional Network (GCN) with an attention mechanism and Parallel Proximal Optimization (PPXA) to learn from sparse, multi-relational biomedical data.
By combining multi-source similarities and known interactions, GAP-DDI captures both local and global patterns, achieving superior predictive performance and uncovering clinically relevant drug indications validated by CTD.

**Implementation**
The GAP-DDI framework is executed in two main stages:

**Stage 1: Feature Learning**

Step 1 – Construct Heterogeneous Network:
Integrate multi-source biological similarities and known associations to build a drug–disease–disease heterogeneous network.

Step 2 – Feature Engineering:
Employ an attention-guided Graph Convolutional Network (GCN) to learn low-dimensional embeddings for drugs and diseases.
The attention layer emphasizes informative relationships and computes feature vectors (FVs) based on cosine similarity.

**Stage 2: Optimization and Prediction**

Step 3 – PPXA-based Matrix Completion:
Use Parallel Proximal Optimization (PPXA) with graph Laplacian regularization to reconstruct the drug–disease association matrix, ensuring stable convergence and accurate prediction of potential associations.
