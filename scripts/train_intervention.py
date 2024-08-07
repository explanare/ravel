import collections
import numpy as np
import re

from datasets import concatenate_datasets
from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
from methods.differential_binary_masking import DifferentialBinaryMasking
import pyvene as pv
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import get_scheduler
from utils.dataset_utils import get_multitask_dataloader
from utils.intervention_utils import train_intervention_step, eval_with_interventions, get_intervention_config, remove_all_forward_hooks, remove_invalid_token_id
from utils.metric_utils import compute_metrics, compute_cross_entropy_loss


def train_intervention(config, model, tokenizer, split_to_dataset):
  print('Training Tasks: %s' % config['training_tasks'])
  # Load datasets.
  concat_split_to_dataset = {
      f'joint-{split}': concatenate_datasets([
          split_to_dataset[f'{task_name}-{split}'].select(
              np.random.choice(
                  min(
                      len(split_to_dataset[f'{task_name}-{split}']) - 1,
                      config['cause_task_sample_size']),
                  size=config['iso_task_sample_size'] if
                  config['training_tasks'][task_name] == 'match_base' else min(
                      len(split_to_dataset[f'{task_name}-{split}']) -
                      1, config['cause_task_sample_size']),
                  replace=False))
          for task_name in config['training_tasks']
          if f'{task_name}-{split}' in split_to_dataset
      ]) for split in ('train',)
  }
  cause_tasks = [
      task_name for task_name, label in config['training_tasks'].items()
      if 'match_source' in label
  ]
  # Remove the -train suffix.
  cause_tasks = [t.rsplit('-', 1)[0] for t in cause_tasks]
  print('Training tasks matching source label: %s' % cause_tasks)
  max_train_example = int(config['max_train_percentage'] *
                          len(concat_split_to_dataset['joint-train']))
  max_input_length = max([
      v['max_input_length'] for v in config['split_to_inv_locations'].values()
  ])
  train_dataloader = get_multitask_dataloader(
      concat_split_to_dataset['joint-train'].select(range(max_train_example)),
      tokenizer=tokenizer,
      batch_size=config['training_batch_size'],
      prompt_max_length=max_input_length,
      output_max_length=3 + config['max_output_tokens'],
      cause_tasks=[
          p for t in cause_tasks for p in config['task_to_prompts'][t]
      ],
      first_n=config['max_output_tokens'])

  # Create model.
  split_to_inv_locations = config['split_to_inv_locations']
  intervenable_config = get_intervention_config(
      type(model),
      config['intervenable_config']['intervenable_representation_type'],
      config['intervenable_config']['intervenable_layer'],
      config['intervenable_config']['intervenable_interventions_type'],
      intervention_dimension=config['intervention_dimension'])
  intervenable = pv.IntervenableModel(intervenable_config, model)
  intervenable.set_device(model.device)
  intervenable.disable_model_gradients()

  # Set up optimizer.
  num_epoch = config['training_epoch']
  regularization_coefficient = config['regularization_coefficient']
  optimizer_params = []
  for k, v in intervenable.interventions.items():
    if isinstance(v[0], LowRankRotatedSpaceIntervention):
      optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
    elif isinstance(v[0], DifferentialBinaryMasking):
      optimizer_params += [{'params': v[0].parameters()}]
  optimizer = torch.optim.AdamW(optimizer_params,
                                lr=config['init_lr'],
                                weight_decay=0)
  scheduler = get_scheduler('constant',
                            optimizer=optimizer,
                            num_training_steps=num_epoch *
                            len(train_dataloader))
  print("Model trainable parameters: ", pv.count_parameters(intervenable.model))
  print("Intervention trainable parameters: ", intervenable.count_parameters())
  temperature_schedule = None
  if (config['intervenable_config']['intervenable_interventions_type'] ==
      DifferentialBinaryMasking):
    temperature_start, temperature_end = config['temperature_schedule']
    temperature_schedule = torch.linspace(temperature_start, temperature_end,
                                          num_epoch * len(train_dataloader) +
                                          1).to(torch.bfloat16).to(model.device)
    for k, v in intervenable.interventions.items():
      if isinstance(v[0], DifferentialBinaryMasking):
        intervenable.interventions[k][0].set_temperature(
            temperature_schedule[scheduler._step_count])

  # Training loop.
  train_iterator = trange(0, int(num_epoch), desc="Epoch")
  tb_writer = SummaryWriter(config['log_dir'])
  num_output_tokens = config['max_output_tokens']
  for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader,
                          desc=f"Epoch: {epoch}",
                          position=0,
                          leave=True)
    aggreated_stats = collections.defaultdict(list)
    for step, inputs in enumerate(epoch_iterator):
      for k, v in inputs.items():
        if v is not None and isinstance(v, torch.Tensor):
          inputs[k] = v.to("cuda")
      b_s = inputs["input_ids"].shape[0]
      position_ids = {
          f'{prefix}position_ids':
          intervenable.model.prepare_inputs_for_generation(
              input_ids=inputs[f"{prefix}input_ids"],
              attention_mask=inputs[f"{prefix}attention_mask"])['position_ids']
          for prefix in ('', 'source_')
      }
      inputs.update(position_ids)
      for key in inputs:
        if key in ('input_ids', 'source_input_ids', 'attention_mask',
                   'source_attention_mask', 'position_ids',
                   'source_position_ids'):
          inputs[key] = inputs[key].to(model.device)

      # Run training step.
      counterfactual_outputs = train_intervention_step(
          intervenable,
          inputs,
          split_to_inv_locations,
          pad_token_id=tokenizer.pad_token_id)
      # Only compute the accuracy of the last N tokens, i.e., the label tokens.
      eval_metrics = compute_metrics([counterfactual_outputs.logits[:, :-1]],
                                     [inputs['labels'][:, :num_output_tokens]],
                                     last_n_tokens=num_output_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
      loss = compute_cross_entropy_loss(counterfactual_outputs.logits,
                                        inputs["labels"][:, :num_output_tokens],
                                        next_n_tokens=num_output_tokens,
                                        pad_token_id=tokenizer.pad_token_id)
      # Add sparsity loss for Differential Binary Masking.
      for k, v in intervenable.interventions.items():
        if isinstance(
            list(intervenable.interventions.values())[0][0],
            DifferentialBinaryMasking):
          loss += regularization_coefficient * intervenable.interventions[k][
              0].get_sparsity_loss()
          intervenable.interventions[k][0].set_temperature(
              temperature_schedule[scheduler._step_count])

      aggreated_stats['loss'].append(loss.item())
      aggreated_stats['acc'].append(eval_metrics["accuracy"])
      epoch_iterator.set_postfix(
          {k: round(np.mean(aggreated_stats[k]), 2) for k in aggreated_stats})

      # Backprop.
      loss.backward()
      optimizer.step()
      scheduler.step()
      intervenable.set_zero_grad()

      # Logging.
      if step % 10 == 0:
        tb_writer.add_scalar("lr",
                             scheduler.get_last_lr()[0], scheduler._step_count)
        tb_writer.add_scalar("loss", loss, scheduler._step_count)
        tb_writer.add_scalar("accuracy", eval_metrics["accuracy"],
                             scheduler._step_count)
      if step < 3:
        print('\nTokens to intervene:')
        intervention_locations = [
            split_to_inv_locations[inputs["split"][i]]['inv_position']
            for i in range(len(inputs["split"]))
        ]
        source_intervention_locations = [
            split_to_inv_locations[inputs["source_split"][i]]['inv_position']
            for i in range(len(inputs["split"]))
        ]
        print(inputs['input'][:3])
        print(inputs['source_input'][:3])
        print(
            'Base:',
            tokenizer.batch_decode([
                inputs['input_ids'][i][intervention_locations[i]]
                for i in range(len(inputs["split"]))
            ]))
        print(
            'Source:',
            tokenizer.batch_decode([
                inputs['source_input_ids'][i][source_intervention_locations[i]]
                for i in range(len(inputs["split"]))
            ]))
        print(
            'Output:',
            tokenizer.batch_decode(
                torch.argmax(
                    counterfactual_outputs.logits[:, -num_output_tokens - 1:-1],
                    dim=-1)))
        print(
            'Label     :',
            tokenizer.batch_decode(
                remove_invalid_token_id(inputs['labels'][:, :num_output_tokens],
                                        tokenizer.pad_token_id)))
        print(
            'Base Label:',
            tokenizer.batch_decode(
                remove_invalid_token_id(
                    inputs['base_labels'][:, :num_output_tokens],
                    tokenizer.pad_token_id)))
  tb_writer.flush()
  tb_writer.close()
  return intervenable, intervenable_config
