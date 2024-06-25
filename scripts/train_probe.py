"""Train a linear adversarial probe using R-LACE.

Adapted from the official R-LACE implementation at
https://github.com/shauli-ravfogel/rlace-icml/blob/master/rlace.py
"""

from methods.linear_adversarial_probe import init_sgd_classifier, solve_adv_game, get_majority_acc
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
import torch


def train_linear_adversarial_probe(config, X, Y):
  dim = X['train'].shape[1]
  X_train, y_train = X['train'], Y['train']
  X_dev, y_dev = X['val_context'], Y['val_context']

  # Check if there exists a linear classifier for the target concept.
  svm = init_sgd_classifier()
  svm.fit(X_train[:], y_train[:])
  print('Classifier scores')
  score_original = svm.score(X_train, y_train)
  print('Train:', score_original)
  score_original = svm.score(X_dev, y_dev)
  print('Test:', score_original)

  # Run R-LACE to find the subspace that encodes the target concept.
  optimizer_params_P = {
      "lr": config['adv_lr'],
      "weight_decay": config['adv_decay']
  }
  optimizer_params_predictor = {
      "lr": config['cls_lr'],
      "weight_decay": config['cls_decay']
  }
  num_classes = config['num_classes']

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
      "cpu")
  output = solve_adv_game(X_train,
                          y_train,
                          X_dev,
                          y_dev,
                          config['num_classes'],
                          rank=config['rank'],
                          device=device,
                          out_iters=config['max_num_iters'],
                          in_iters_adv=config['in_iters_adv'],
                          in_iters_clf=config['in_iters_clf'],
                          optimizer_class=config['optimizer_class'],
                          optimizer_params_P=optimizer_params_P,
                          optimizer_params_predictor=optimizer_params_predictor,
                          epsilon=config['epsilon'],
                          batch_size=config['batch_size'],
                          evalaute_every=10)

  # Evaluate the subspace.
  P_svd = output["P"]
  P_before_svd = output["P_before_svd"]

  svm = init_sgd_classifier()
  svm.fit(X_train[:], y_train[:])
  score_original = svm.score(X_dev, y_dev)

  svm = init_sgd_classifier()
  svm.fit(X_train[:] @ P_before_svd, y_train[:])
  score_projected_no_svd = svm.score(X_dev @ P_before_svd, y_dev)

  svm = init_sgd_classifier()
  svm.fit(X_train[:] @ P_svd, y_train[:])
  score_projected_svd_dev = svm.score(X_dev @ P_svd, y_dev)
  score_projected_svd_train = svm.score(X_train @ P_svd, y_train)
  maj_acc_dev = get_majority_acc(y_dev)
  maj_acc_train = get_majority_acc(y_train)

  print("===================================================")
  print(
      "Original Acc, dev: {:.3f}%; Acc, projected, no svd, dev: {:.3f}%; Acc, projected+SVD, train: {:.3f}%; Acc, projected+SVD, dev: {:.3f}%"
      .format(score_original * 100, score_projected_no_svd * 100,
              score_projected_svd_train * 100, score_projected_svd_dev * 100))
  print("Majority Acc, dev: {:.3f} %".format(maj_acc_dev * 100))
  print("Majority Acc, train: {:.3f} %".format(maj_acc_train * 100))
  print("Gap, dev: {:.3f} %".format(
      np.abs(maj_acc_dev - score_projected_svd_dev) * 100))
  print("Gap, train: {:.3f} %".format(
      np.abs(maj_acc_train - score_projected_svd_train) * 100))
  print("===================================================")
  eigs_before_svd, _ = np.linalg.eigh(P_before_svd)
  print("Eigenvalues, before SVD: {}".format(eigs_before_svd[:]))

  eigs_after_svd, _ = np.linalg.eigh(P_svd)
  print("Eigenvalues, after SVD: {}".format(eigs_after_svd[:]))

  return output['W']
