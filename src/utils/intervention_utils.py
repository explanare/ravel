"""Utility functions for training and evaluating interventions."""

import collections
import copy
import numpy as np
import re

from datasets import Dataset
from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
from methods.pca import PCARotatedSpaceIntervention
from methods.sparse_autoencoder import AutoencoderIntervention
import pyvene as pv
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from utils.dataset_utils import get_dataloader, is_llama_tokenizer


def get_intervention_config(model_type,
                            intervention_representation,
                            layers,
                            intervention_type,
                            intervention_dimension=None,
                            num_unit=1):
  if isinstance(layers, int):
    layers = [layers]
  inv_config = pv.IntervenableConfig(
      model_type=model_type,
      representations=[
          pv.RepresentationConfig(
              layer,  # layer
              intervention_representation,  # intervention repr
              "pos",  # intervention unit
              num_unit,  # max number of unit
              intervention_dimension) for layer in layers
      ],
      intervention_types=intervention_type,
  )
  return inv_config


def train_intervention_step(intervenable, inputs, split_to_inv_locations,
                            pad_token_id):
  inputs = copy.deepcopy(inputs)
  b_s = inputs["input_ids"].shape[0]
  # Set intervention locations.
  # These locations are invariant to the label appended later.
  num_inv = len(intervenable.interventions)
  intervention_locations = {
      "sources->base": ([[
          split_to_inv_locations[inputs["source_split"][i]]['inv_position']
          for i in range(b_s)
      ]] * num_inv, [[
          split_to_inv_locations[inputs["split"][i]]['inv_position']
          for i in range(b_s)
      ]] * num_inv)
  }
  # Append label to input.
  inputs['labels'][inputs['labels'] < 0] = pad_token_id
  inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['labels']],
                                  dim=-1)
  inputs['attention_mask'] = torch.zeros(inputs['input_ids'].shape,
                                         dtype=inputs['attention_mask'].dtype,
                                         device=inputs['attention_mask'].device)
  inputs['attention_mask'][inputs['input_ids'] != pad_token_id] = 1
  position_ids = {
      f'{prefix}position_ids': intervenable.model.prepare_inputs_for_generation(
          input_ids=inputs[f"{prefix}input_ids"],
          attention_mask=inputs[f"{prefix}attention_mask"])['position_ids']
      for prefix in ('', 'source_')
  }
  inputs.update(position_ids)
  _, counterfactual_outputs = intervenable(
      {
          "input_ids": inputs["input_ids"],
          'attention_mask': inputs["attention_mask"],
          'position_ids': inputs['position_ids']
      }, [{
          "input_ids": inputs["source_input_ids"],
          'attention_mask': inputs["source_attention_mask"],
          'position_ids': inputs['source_position_ids']
      }] * num_inv, intervention_locations)
  return counterfactual_outputs


def remove_all_forward_hooks(model):
  for name, child in model._modules.items():
    if child is not None:
      if hasattr(child, "_forward_hooks") and len(child._forward_hooks) > 0:
        print(child._forward_hooks)
        print(name, child)
        child._forward_hooks = collections.OrderedDict()
      remove_all_forward_hooks(child)


def remove_invalid_token_id(token_ids, pad_id=2):
  token_ids = token_ids.clone()
  token_ids[token_ids == -100] = pad_id
  return token_ids


def eval_with_interventions(intervenable,
                            split_to_dataset,
                            split_to_inv_locations,
                            tokenizer,
                            compute_metrics_fn,
                            max_new_tokens=1,
                            eval_batch_size=16,
                            debug_print=False,
                            forward_only=False):
  split_to_eval_metrics = {}
  padding_offset = 3 if is_llama_tokenizer(tokenizer) else 0
  num_inv = len(intervenable.interventions)
  for split in split_to_dataset:
    # Asssume all inputs have the same max length.
    prompt_max_length = split_to_inv_locations[split_to_dataset[split][0]
                                               ['split']]['max_input_length']
    eval_dataloader = get_dataloader(split_to_dataset[split],
                                     tokenizer=tokenizer,
                                     batch_size=eval_batch_size,
                                     prompt_max_length=prompt_max_length,
                                     output_max_length=padding_offset +
                                     max_new_tokens,
                                     first_n=max_new_tokens)
    eval_labels = collections.defaultdict(list)
    eval_preds = []
    with torch.no_grad():
      if debug_print:
        epoch_iterator = tqdm(eval_dataloader, desc=f"Test")
      else:
        epoch_iterator = eval_dataloader
      for step, inputs in enumerate(epoch_iterator):
        b_s = inputs["input_ids"].shape[0]
        position_ids = {
            f'{prefix}position_ids':
            intervenable.model.prepare_inputs_for_generation(
                input_ids=inputs[f"{prefix}input_ids"],
                attention_mask=inputs[f"{prefix}attention_mask"])
            ['position_ids'] for prefix in ('', 'source_')
        }
        inputs.update(position_ids)
        for key in inputs:
          if key in ('input_ids', 'source_input_ids', 'attention_mask',
                     'source_attention_mask', 'position_ids',
                     'source_position_ids', 'labels', 'base_labels'):
            inputs[key] = inputs[key].to(intervenable.model.device)
        intervention_locations = {
            "sources->base": ([[
                split_to_inv_locations[inputs["source_split"][i]]
                ['inv_position'] for i in range(b_s)
            ]] * num_inv, [[
                split_to_inv_locations[inputs["split"][i]]['inv_position']
                for i in range(b_s)
            ]] * num_inv)
        }
        if not forward_only:
          base_outputs, counterfactual_outputs = intervenable.generate(
              {
                  "input_ids": inputs["input_ids"],
                  "attention_mask": inputs["attention_mask"]
              },
              [{
                  "input_ids": inputs["source_input_ids"],
                  'attention_mask': inputs["source_attention_mask"],
                  'position_ids': inputs['source_position_ids']
              }],
              intervention_locations,
              max_new_tokens=max_new_tokens,
              do_sample=False,
              intervene_on_prompt=True,
              pad_token_id=tokenizer.pad_token_id,
              output_original_output=True,
          )
          eval_preds.append(counterfactual_outputs)
        else:
          base_outputs, counterfactual_outputs = intervenable(
              {
                  "input_ids": inputs["input_ids"],
                  'attention_mask': inputs["attention_mask"],
                  'position_ids': inputs['position_ids']
              },
              [{
                  "input_ids": inputs["source_input_ids"],
                  'attention_mask': inputs["source_attention_mask"],
                  'position_ids': inputs['source_position_ids']
              }],
              intervention_locations,
              output_original_output=True,
          )
          eval_preds.append(counterfactual_outputs.logits)
          counterfactual_outputs = torch.argmax(counterfactual_outputs.logits,
                                                dim=-1)
          base_outputs = torch.argmax(base_outputs.logits, dim=-1)

        for label_type in ['base_labels', 'labels']:
          eval_labels[label_type].append(inputs[label_type])
        eval_labels['base_outputs'].append(base_outputs[:, -max_new_tokens:])
        if debug_print and step < 3:
          print('\nInputs:')
          print('Base:', inputs['input'][:3])
          print('Source:', inputs['source_input'][:3])
          print('Tokens to intervene:')
          print(
              'Base:',
              tokenizer.batch_decode([
                  inputs['input_ids'][i][intervention_locations["sources->base"]
                                         [1][0][i]]
                  for i in range(len(inputs["split"]))
              ]))
          print(
              'Source:',
              tokenizer.batch_decode([
                  inputs['source_input_ids'][i][
                      intervention_locations["sources->base"][0][0][i]]
                  for i in range(len(inputs["split"]))
              ]))
          base_output_text = tokenizer.batch_decode(
              base_outputs[:, -max_new_tokens:], skip_special_tokens=True)
          print('Base Output:', base_output_text)
          print(
              'Output:    ',
              tokenizer.batch_decode(counterfactual_outputs[:,
                                                            -max_new_tokens:]))
          print(
              'Inv Label: ',
              tokenizer.batch_decode(
                  remove_invalid_token_id(inputs['labels'][:, :max_new_tokens],
                                          tokenizer.pad_token_id)))
          base_label_text = tokenizer.batch_decode(remove_invalid_token_id(
              inputs['base_labels'][:, :max_new_tokens],
              tokenizer.pad_token_id),
                                                   skip_special_tokens=True)
          print('Base Label:', base_label_text)
          if base_label_text != base_output_text:
            print('WARNING: Base outputs does not match base labels!')
    eval_metrics = {
        label_type: compute_metrics_fn(eval_preds,
                                       eval_labels[label_type],
                                       last_n_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id,
                                       extra_labels=eval_labels,
                                       eval_label_type=label_type)
        for label_type in eval_labels
        if label_type.endswith('labels')
    }
    print('\n', repr(split) + ':', eval_metrics)
    split_to_eval_metrics[split] = {
        'metrics':
            eval_metrics,
        'inv_outputs':
            tokenizer.batch_decode(counterfactual_outputs[:, -max_new_tokens:]),
        'inv_labels':
            tokenizer.batch_decode(
                remove_invalid_token_id(inputs['labels'][:, :max_new_tokens],
                                        tokenizer.pad_token_id)),
        'base_labels':
            tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs['base_labels'][:, :max_new_tokens],
                    tokenizer.pad_token_id)),
    }
  return split_to_eval_metrics


class PretrainedFeaturizer(torch.nn.Module):
  """A pretrained featurizer, which is typically a linear layer."""

  def __init__(self, pretrained_weight_or_path):
    super().__init__()
    if isinstance(pretrained_weight_or_path, str):
      if pretrained_weight_or_path.endswith('.pt'):
        self.weight = torch.load(pretrained_weight_or_path)
      elif pretrained_weight_or_path.endswith('.npy'):
        self.weight = torch.tensor(np.load(pretrained_weight_or_path))
    else:
      # Convert input weight to torch.Tensor.
      self.weight = torch.tensor(pretrained_weight_or_path)
    if self.weight.shape[0] > self.weight.shape[1]:
      self.weight = self.weight.T

  def forward(self, x):
    return torch.matmul(x.to(self.weight.dtype), self.weight.T)


def load_intervenable_with_pca(model, pca_param_path):
  pca_params = torch.load(pca_param_path)
  layer_dim_match = re.search(r'layer(\d+)[\-_.]', pca_param_path)
  layer = int(layer_dim_match.group(1))
  inv_config = get_intervention_config(type(model),
                                       "block_output",
                                       layer,
                                       PCARotatedSpaceIntervention,
                                       num_unit=1)
  intervenable = pv.IntervenableModel(inv_config, model)
  intervenable.set_device("cuda")
  intervenable.disable_model_gradients()
  key = list(intervenable.interventions)[0]
  intervenable.interventions[key][0].set_pca_params(pca_params)
  print('#Principal Components=%d' %
        intervenable.interventions[key][0].pca_components.shape[0])
  return intervenable


def load_intervenable_with_autoencoder(model, autoencoder, inv_dims, layer):
  inv_config = get_intervention_config(type(model),
                                       "block_output",
                                       layer,
                                       AutoencoderIntervention,
                                       num_unit=1)
  intervenable = pv.IntervenableModel(inv_config, model)
  intervenable.set_device("cuda")
  intervenable.disable_model_gradients()
  for k in intervenable.interventions:
    intervenable.interventions[k][0].autoencoder = autoencoder
    intervenable.interventions[k][0].inv_dims = inv_dims
  intervenable.model.eval()
  return intervenable


def load_intervenable(base_model, pretrained_weight_or_path):
  """Load interventions that involve a linear transformation."""

  run_name = pretrained_weight_or_path.rsplit('.', 1)[0].rsplit('/', 1)[-1]
  # Support formats: {inv_key: torch.Tensor}, torch.Tensor, numpy.array
  rotate_layers = {}
  if pretrained_weight_or_path.endswith('.pt'):
    inv_key_to_weights = torch.load(pretrained_weight_or_path)
    if isinstance(inv_key_to_weights, dict):
      for k, v in inv_key_to_weights.items():
        rotate_layer = PretrainedFeaturizer(v).eval()
        print(k)
        print('Loaded feature projection matrix shape:',
              rotate_layer.weight.shape)
        rotate_layers[k] = rotate_layer
    else:
      # Weights saved without intervention key.
      layer_match = re.search(r'layer(\d+)[\-_.]', run_name)
      layer = int(layer_match.group(1))
      rotate_layer = PretrainedFeaturizer(inv_key_to_weights).eval()
      print('Loaded feature projection matrix shape:',
            rotate_layer.weight.shape)
      rotate_layers[
          f'layer.{layer}.comp.block_output.unit.pos.nunit.1#0'] = rotate_layer
  layers = [int(k.split('.')[1]) for k in rotate_layers]
  inv_config = get_intervention_config(type(base_model), "block_output", layers,
                                       LowRankRotatedSpaceIntervention, 0)
  intervenable = pv.IntervenableModel(inv_config, base_model)
  intervenable.set_device("cuda")
  intervenable.disable_model_gradients()
  for k, v in rotate_layers.items():
    intervenable.interventions[k][0].rotate_layer = v
    intervenable.interventions[k][0].set_interchange_dim(
        interchange_dim=v.weight.shape[0])
  intervenable.model.eval()
  return intervenable
