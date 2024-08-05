"""Utility functions for computing metrics."""

import numpy as np

import torch
from torch.nn import CrossEntropyLoss


def compute_metrics(eval_preds,
                    eval_labels,
                    pad_token_id,
                    last_n_tokens=1,
                    **kwargs):
  """Computes squence-level and token-level accuracy."""
  total_count, total_token_count = 0, 0
  correct_count, correct_token_count = 0, 0
  for eval_pred, eval_label in zip(eval_preds, eval_labels):
    actual_test_labels = eval_label[:, -last_n_tokens:]
    if len(eval_pred.shape) == 3:
      # eval_preds is in the form of logits.
      pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
    else:
      # eval_preds is in the form of token ids.
      pred_test_labels = eval_pred[:, -last_n_tokens:]
    padding_tokens = torch.logical_or(actual_test_labels == pad_token_id,
                                      actual_test_labels < 0)
    match_tokens = actual_test_labels == pred_test_labels
    correct_labels = torch.logical_or(match_tokens, padding_tokens)
    total_count += len(correct_labels)
    correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
    total_token_count += (~padding_tokens).float().sum().tolist()
    correct_token_count += (~padding_tokens &
                            match_tokens).float().sum().tolist()
  accuracy = round(correct_count / total_count, 2)
  token_accuracy = round(correct_token_count / total_token_count, 2)
  return {"accuracy": accuracy, "token_accuracy": token_accuracy}


def compute_cross_entropy_loss(logits, labels, pad_token_id, next_n_tokens=1):
  """Computes cross-entropy loss over the last n tokens."""
  vocab_size = logits.shape[-1]
  labels = labels.clone()
  shift_logits = logits[..., -next_n_tokens - 1:-1, :].contiguous()
  shift_labels = labels[..., -next_n_tokens:].contiguous()
  shift_logits = shift_logits.view(-1, vocab_size)
  shift_labels = shift_labels.view(-1)
  shift_labels = shift_labels.to(shift_logits.device)
  shift_labels[shift_labels == pad_token_id] = -100
  loss = CrossEntropyLoss()(shift_logits, shift_labels)
  return loss


def compute_disentangle_score(log_data,
                              attribute_to_iso_tasks,
                              attribute_to_cause_tasks):
  """Compute disentanglement score from iso/cause scores."""
  match_base = np.mean([
      np.mean([log_data[t]['metrics']['base_labels']['accuracy']
               for t in ts if t in log_data])
      for a, ts in attribute_to_iso_tasks.items()])
  match_source = np.mean([
      np.mean([log_data[t]['metrics']['labels']['accuracy']
               for t in ts if t in log_data])
      for a, ts in attribute_to_cause_tasks.items()])
  return {'disentangle': 0.5 * (match_base + match_source),
          'isolate': match_base,
          'cause': match_source}
