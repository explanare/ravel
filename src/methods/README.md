# Interpretability Methods

This module implements five families of interpretability methods that localize a target concept to model components. Given a set of model representations $X\subset \mathbb{R}^n$ as inputs, each interpretability method defines a bijective featurizer $\mathcal{F}$ (e.g., a rotation matrix or sparse autoencoder), and identify a feature $F$ that represents the target concept (e.g., a linear subspace of the residual stream in a Transformer that represents a concept). 


## Principal Component Analysis (PCA)

* Featurizer: An $n \times n$ orthogonal matrix formed by the principal components.
* Features: Selected by a classifier.
* Training: Featurizer does not require training; Feature selection requires training an attribute value classifier.

## Sparse Autoencoder (SAE)

* Featurizer: The encoder and decoder, assuming the reconstruction loss is sufficiently small.
* Features: Selected by a classifier.
* Training: Featurizer is trained on unsupervised reconstruction loss; Feature selection requires training an attribute value classifier.

## Relaxed Linear Adversarial Probe (RLAP)

* Featurizer: An $n \times n$ orthogonal matrix formed by the set of $k$ orthonormal vectors that span the row space of the probe and $n-k$ orthonormal vectors that span the null space of the probe, where $k$ is the rank of the probe.
* Features: The first $k$ dimensions, i.e., the $k$ dimensions correspond to the row space.
* Training: Featurizer is trained on attribute value classification; No additional feature selection required.

## Differential Binary Masking (DBM)

* Featurizer: An identity matrix.
* Features: Selected by a learned binary mask.
* Training: Features require training with counterfactual signals.


## Distributed Alignment Search (DAS)

* Featurizer: An $n \times n$ orthogonal matrix learned with counterfactual signals.
* Features: The first $k$ dimension.
* Training: Featurizer requires training with counterfactual signals.


