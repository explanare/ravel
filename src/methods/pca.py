"""Define a featurizer and features with principal components."""

import numpy as np

import pyvene as pv
from sklearn.decomposition import PCA
import torch


def compute_pca(features, n_components):
  X = np.array(features)
  # Normalize input features to zero mean unit variance.
  pca_mean = np.mean(X, axis=0, keepdims=True)
  pca_std = X.var(axis=0)**0.5
  X = (X - pca_mean) / pca_std
  print('PCA input stats:', X.min(), X.max(), X.mean(), X.std())

  pca = PCA(n_components=n_components)
  pca.fit(X)
  print('PCA explained variance: %.2f' % np.sum(pca.explained_variance_ratio_))
  return {'mean': pca_mean, 'std': pca_std, 'components': pca.components_}


class PCARotatedSpaceIntervention(pv.TrainableIntervention):
  """Intervene in the rotated subspace defined by principal components."""

  def __init__(self, **kwargs):
    super().__init__()

  def set_pca_params(self, pca_results):
    self.pca_components = torch.tensor(pca_results['components'],
                                       dtype=torch.float32)
    self.pca_mean = torch.tensor(pca_results['mean'], dtype=torch.float32)
    self.pca_std = torch.tensor(pca_results['std'], dtype=torch.float32)

  def forward(self, base, source, subspaces=None):
    base_norm = (base - self.pca_mean) / self.pca_std
    source_norm = (source - self.pca_mean) / self.pca_std

    input_dtype, model_dtype = base.dtype, self.pca_components.dtype
    base, source = base.to(model_dtype), source.to(model_dtype)
    rotated_base = torch.matmul(base_norm, self.pca_components.T)  # B * D_R
    rotated_source = torch.matmul(source_norm, self.pca_components.T)
    # Apply interchange interventions.
    output = base + torch.matmul(
        (rotated_source - rotated_base), self.pca_components)
    output = (output * self.pca_std) + self.pca_mean
    return output.to(input_dtype)
