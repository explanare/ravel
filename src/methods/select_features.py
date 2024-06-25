import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import torch


def select_features_with_classifier(featurizer, inputs, labels, coeff=None):
  if coeff is None:
    coeff = [0.1, 10, 100, 1000]
  coeff_to_select_features = {}
  for c in coeff:
    with torch.no_grad():
      X_transformed = featurizer(inputs).cpu().numpy()
      lsvc = LinearSVC(C=c, penalty="l1", dual=False, max_iter=5000,
                       tol=0.01).fit(X_transformed, labels)
      selector = SelectFromModel(lsvc, prefit=True)
      kept_dim = np.where(selector.get_support())[0]
      coeff_to_select_features[c] = kept_dim
  return coeff_to_select_features
