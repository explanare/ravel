"""Utility functions for preprocessing and loading datasets."""

import copy
from functools import partial
import numpy as np
import pickle as pkl
import re

from datasets import concatenate_datasets
import h5py
import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizerFast


def preproc_tokenize(tokenizer,
                     max_input_length,
                     max_output_length,
                     examples,
                     input_feature=None,
                     label_feature=None,
                     extra_input_to_tokenize=None,
                     extra_label_to_tokenize=None):
  assert tokenizer.padding_side == 'left'
  input_batch = copy.deepcopy(examples)
  input_feature = input_feature or 'input'
  label_feature = label_feature or 'label'
  input_batch.update(
      tokenizer(examples[input_feature],
                padding="max_length",
                max_length=max_input_length,
                return_tensors="pt",
                truncation=True))
  # Right padding labels.
  tokenizer.padding_side = 'right'
  labels = tokenizer(examples[label_feature],
                     padding="max_length",
                     max_length=max_output_length,
                     return_tensors="pt",
                     truncation=True)['input_ids']
  tokenizer.padding_side = 'left'
  labels[labels == tokenizer.pad_token_id] = -100
  input_batch['labels'] = labels
  if extra_input_to_tokenize:
    for feat in extra_input_to_tokenize:
      tokenized_feat = tokenizer(examples[feat],
                                 padding="max_length",
                                 max_length=max_input_length,
                                 return_tensors="pt",
                                 truncation=True)
      input_batch[f'{feat.replace("_input", "")}_input_ids'] = tokenized_feat[
          'input_ids']
      input_batch[
          f'{feat.replace("_input", "")}_attention_mask'] = tokenized_feat[
              'attention_mask']
  if extra_label_to_tokenize:
    for label in extra_label_to_tokenize:
      # Right padding labels.
      tokenizer.padding_side = 'right'
      tokenized_feat = tokenizer(examples[label],
                                 padding="max_length",
                                 max_length=max_output_length,
                                 return_tensors="pt",
                                 truncation=True)
      tokenizer.padding_side = 'left'
      input_batch[f'{label.split("_")[0] + "_labels"}'] = tokenized_feat[
          'input_ids']
  # Remove extra nesting.
  for k in input_batch:
    if isinstance(input_batch[k], list) or isinstance(input_batch[k],
                                                      torch.Tensor):
      input_batch[k] = input_batch[k][0]
  return input_batch


def kept_first_n_label_token(x, first_n, padding_offset=3):
  x['base_labels'] = x['labels'][padding_offset:padding_offset + first_n]
  # Remove the <s> and SOS Pad token.
  x['labels'] = x['inv_labels'][padding_offset:padding_offset + first_n]
  return x


def kept_first_n_label_token_multitask(x,
                                       cause_tasks,
                                       first_n,
                                       padding_offset=3):
  x['base_labels'] = x['labels'][padding_offset:padding_offset + first_n]
  # Remove the <s> and SOS Pad token.
  if x['split'] in cause_tasks:
    x['labels'] = x['inv_labels'][padding_offset:padding_offset + first_n]
  else:
    x['labels'] = x['labels'][padding_offset:padding_offset + first_n]
  return x


def is_llama_tokenizer(tokenizer):
  return isinstance(tokenizer, LlamaTokenizerFast)


def get_dataloader(eval_dataset,
                   tokenizer,
                   batch_size,
                   prompt_max_length,
                   output_max_length,
                   first_n=1,
                   drop_last=False):
  eval_dataset = eval_dataset.map(
      lambda x: preproc_tokenize(tokenizer,
                                 prompt_max_length,
                                 output_max_length,
                                 x,
                                 extra_input_to_tokenize=['source_input'],
                                 extra_label_to_tokenize=['inv_label']))
  eval_dataset = eval_dataset.map(lambda x: kept_first_n_label_token(
      x, first_n, padding_offset=3 if is_llama_tokenizer(tokenizer) else 0))
  eval_dataset = eval_dataset.with_format("torch")
  eval_dataloader = DataLoader(eval_dataset,
                               batch_size=batch_size,
                               drop_last=drop_last,
                               shuffle=True)
  return eval_dataloader


def get_multitask_dataloader(eval_dataset,
                             tokenizer,
                             batch_size,
                             prompt_max_length,
                             output_max_length,
                             cause_tasks,
                             first_n=1):
  eval_dataset = eval_dataset.map(
      lambda x: preproc_tokenize(tokenizer,
                                 prompt_max_length,
                                 output_max_length,
                                 x,
                                 extra_input_to_tokenize=['source_input'],
                                 extra_label_to_tokenize=['inv_label']))
  eval_dataset = eval_dataset.map(lambda x: kept_first_n_label_token_multitask(
      x,
      cause_tasks,
      first_n,
      padding_offset=3 if is_llama_tokenizer(tokenizer) else 0))
  eval_dataset = eval_dataset.with_format("torch")
  eval_dataloader = DataLoader(eval_dataset,
                               batch_size=batch_size,
                               drop_last=True,
                               shuffle=True)
  return eval_dataloader


class HDF5Dataset(torch.utils.data.Dataset):
  """Load model representations in HDF5 format."""

  def __init__(self, file_path, sample_range=None):
    super(HDF5Dataset, self).__init__()
    f_features = h5py.File(file_path, "r")
    print(
        f'#Entities={len(f_features)}, #Examples={sum([len(f_features[k]) for k in f_features])}'
    )
    self.entity_to_features = {
        k: torch.tensor(
            np.array(v)[[i for i in sample_range if i < len(v)]]
            if sample_range is not None else np.array(v))
        for k, v in f_features.items()
    }
    self.index_to_keys = [(k, i)
                          for k in self.entity_to_features
                          for i in range(len(self.entity_to_features[k]))]
    print('Load %d examples' % len(self.index_to_keys))

  def __getitem__(self, index):
    key, offset = self.index_to_keys[index]
    return {
        'input_text': key,
        'input_feature': self.entity_to_features[key][offset]
    }

  def __len__(self):
    return len(self.index_to_keys)


def load_entity_representation_with_label(feature_hdf5_path,
                                          entity_attr_to_label, splits):
  f = h5py.File(feature_hdf5_path, 'r')
  attributes = list(entity_attr_to_label.values())[0].keys()
  X, Y = {}, {}
  for attr in attributes:
    X[attr] = {
        split: np.array(f['%s-%s' % (attr, split)][:], np.float32)
        for split in splits
    }
    entities = {
        split: pkl.loads(np.void(f['%s-%s_entity' % (attr, split)]))
        for split in splits
    }
    labels = {
        split: [entity_attr_to_label[e][attr] for e in entities[split]
               ] for split in splits
    }
    sorted_unique_label = sorted(
        set([x for split in splits for x in labels[split]]))
    print('#unique labels=%d' % len(sorted_unique_label), sorted_unique_label)
    Y[attr] = {
        split: np.array([sorted_unique_label.index(x) for x in labels[split]
                        ], np.int64) for split in labels
    }
  return X, Y, sorted_unique_label
