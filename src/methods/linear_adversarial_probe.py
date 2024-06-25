"""Define a featurizer and features with a linear adversarial probe.

We use the R-LACE method (Ravfogel et al. 2022).

Adapted from the official R-LACE implementation at
https://github.com/shauli-ravfogel/rlace-icml/blob/master/rlace.py

Instead of using the orthogonal projection matrix P that removes a concept,
we are interested in the subspace that represents the concept,
which is the row space of orthogonal matrix W in P = I - W^TW.
"""

import random
import numpy as np
import time

import sklearn
from sklearn.linear_model import SGDClassifier
import torch
from torch.optim import SGD, Adam
import tqdm


def init_sgd_classifier(params=None):
  if params is None:
    params = {
        "loss": "log_loss",
        "tol": 1e-4,
        "iters_no_change": 15,
        "alpha": 1e-4,
        "max_iter": 25000,
    }
  return SGDClassifier(loss=params["loss"],
                       fit_intercept=True,
                       max_iter=params["max_iter"],
                       tol=params["tol"],
                       n_iter_no_change=params["iters_no_change"],
                       n_jobs=32,
                       alpha=params["alpha"])


def symmetric(X):
  X.data = 0.5 * (X.data + X.data.T)
  return X


def get_score(X_train, y_train, X_dev, y_dev, P, rank):
  P_svd, _ = get_projection(P, rank)

  loss_vals = []
  accs = []

  num_clfs_in_eval = 1
  for i in range(num_clfs_in_eval):
    clf = init_sgd_classifier()
    clf.fit(X_train @ P_svd, y_train)
    y_pred = clf.predict_proba(X_dev @ P_svd)
    print(np.argmax(y_pred, axis=-1)[:10])
    print(y_dev[:10])
    loss = sklearn.metrics.log_loss(y_dev, y_pred)
    loss_vals.append(loss)
    accs.append(clf.score(X_dev @ P_svd, y_dev))

  i = np.argmin(loss_vals)
  return loss_vals[i], accs[i]


def solve_constraint(lambdas, d=1):

  def f(theta):
    return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
    return return_val

  theta_min, theta_max = max(lambdas), min(lambdas) - 1
  assert f(theta_min) * f(theta_max) < 0

  mid = (theta_min + theta_max) / 2
  tol = 1e-4
  iters = 0

  while iters < 25:
    mid = (theta_min + theta_max) / 2
    if f(mid) * f(theta_min) > 0:
      theta_min = mid
    else:
      theta_max = mid
    iters += 1

  lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1)
  if (theta_min - theta_max)**2 > tol:
    print("didn't converge", (theta_min - theta_max)**2)
  return lambdas_plus


def get_majority_acc(y):

  from collections import Counter
  c = Counter(y)
  fracts = [v / sum(c.values()) for v in c.values()]
  maj = max(fracts)
  return maj


def get_entropy(y):

  from collections import Counter
  import scipy

  c = Counter(y)
  fracts = [v / sum(c.values()) for v in c.values()]
  return scipy.stats.entropy(fracts)


def get_projection(P, rank):
  D, U = np.linalg.eigh(P)
  U = U.T
  W = U[-rank:]
  P_final = np.eye(P.shape[0]) - W.T @ W
  return P_final, W


def prepare_output(P, rank, score):
  P_final, W = get_projection(P, rank)
  return {
      "score": score,
      "P_before_svd": np.eye(P.shape[0]) - P,
      "P": P_final,
      "W": W
  }


def solve_adv_game(X_train,
                   y_train,
                   X_dev,
                   y_dev,
                   num_classes,
                   rank=1,
                   device="cpu",
                   out_iters=75000,
                   in_iters_adv=1,
                   in_iters_clf=1,
                   epsilon=0.0015,
                   batch_size=128,
                   evalaute_every=1000,
                   optimizer_class=SGD,
                   optimizer_params_P={
                       "lr": 0.005,
                       "weight_decay": 1e-4
                   },
                   optimizer_params_predictor={
                       "lr": 0.005,
                       "weight_decay": 1e-4
                   }):
  """

    :param X: The input (np array)
    :param Y: the lables (np array)
    :param X_dev: Dev set (np array)
    :param Y_dev: Dev labels (np array)
    :param rank: Number of dimensions to neutralize from the input.
    :param device:
    :param out_iters: Number of batches to run
    :param in_iters_adv: number of iterations for adversary's optimization
    :param in_iters_clf: number of iterations from the predictor's optimization
    :param epsilon: stopping criterion .Stops if abs(acc - majority) < epsilon.
    :param batch_size:
    :param evalaute_every: After how many batches to evaluate the current adversary.
    :param optimizer_class: SGD/Adam etc.
    :param optimizer_params: the optimizer's params (as a dict)
    :return:
    """

  def get_loss_fn(X, y, predictor, P, bce_loss_fn, optimize_P=False):
    I = torch.eye(X_train.shape[1]).to(device)
    bce = bce_loss_fn(predictor(X @ (I - P)).squeeze(), y)
    if optimize_P:
      bce = -bce
    return bce

  X_torch = torch.tensor(X_train).float().to(device)
  y_torch = torch.tensor(y_train).float().to(device)

  num_labels = num_classes
  if num_labels == 2:
    predictor = torch.nn.Linear(X_train.shape[1], 1).to(device)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    y_torch = y_torch.float()
  else:
    predictor = torch.nn.Linear(X_train.shape[1], num_labels).to(device)
    bce_loss_fn = torch.nn.CrossEntropyLoss()
    y_torch = y_torch.long()

  P = 1e-1 * torch.randn(X_train.shape[1], X_train.shape[1]).to(device)
  P.requires_grad = True

  optimizer_predictor = optimizer_class(predictor.parameters(),
                                        **optimizer_params_predictor)
  optimizer_P = optimizer_class([P], **optimizer_params_P)

  maj = get_majority_acc(y_train)
  label_entropy = get_entropy(y_train)
  pbar = tqdm.tqdm(range(out_iters), total=out_iters, ascii=True)
  count_examples = 0
  best_P, best_score, best_loss = None, 1, -1

  for i in pbar:
    for j in range(in_iters_adv):
      P = symmetric(P)
      optimizer_P.zero_grad()

      idx = np.arange(0, X_torch.shape[0])
      np.random.shuffle(idx)
      X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

      loss_P = get_loss_fn(X_batch,
                           y_batch,
                           predictor,
                           symmetric(P),
                           bce_loss_fn,
                           optimize_P=True)
      loss_P.backward()
      optimizer_P.step()

      # project
      with torch.no_grad():
        # A named tuple (eigenvalues, eigenvectors)
        D, U = torch.linalg.eigh(symmetric(P))
        D = D.detach().cpu().numpy()
        D_plus_diag = solve_constraint(D, d=rank)
        D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
        P.data = U @ D @ U.T

    for j in range(in_iters_clf):
      optimizer_predictor.zero_grad()
      idx = np.arange(0, X_torch.shape[0])
      np.random.shuffle(idx)
      X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

      loss_predictor = get_loss_fn(X_batch,
                                   y_batch,
                                   predictor,
                                   symmetric(P),
                                   bce_loss_fn,
                                   optimize_P=False)
      loss_predictor.backward()
      optimizer_predictor.step()
      count_examples += batch_size

    if i % evalaute_every == 0:
      loss_val, score = get_score(X_train, y_train, X_train, y_train,
                                  P.detach().cpu().numpy(), rank)
      print('iter %d: Eval loss %.4f score %.4f' % (i, loss_val, score))
      if loss_val > best_loss:
        best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
      if np.abs(score - maj) < np.abs(best_score - maj):
        best_score = score

      # update progress bar
      best_so_far = best_score if np.abs(best_score -
                                         maj) < np.abs(score - maj) else score

      pbar.set_description(
          "{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}"
          .format(i, out_iters, score * 100, best_so_far * 100, maj * 100,
                  np.abs(best_so_far - maj) * 100, best_loss, loss_val))
      pbar.refresh()  # to show immediately the update
      time.sleep(0.01)

    if i > 1 and np.abs(best_score - maj) < epsilon:
      break
  output = prepare_output(best_P, rank, best_score)
  return output
