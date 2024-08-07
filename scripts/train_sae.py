import collections
import os

from methods.sparse_autoencoder import SparseAutoencoder
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import transformers
from transformers import get_scheduler


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_autoencoder_step(autoencoder, input_batch):
  outputs = autoencoder(input_batch['input_feature'])
  losses = autoencoder.get_autoencoder_losses()
  return losses


def train_sae(config, train_dataloader, val_dataloader):
  # Setup the model.
  task_name = config['task_name']
  device = (torch.device("cuda")
            if torch.cuda.is_available() else torch.device("cpu"))
  autoencoder = SparseAutoencoder(config['input_dim'],
                                  config['latent_dim'],
                                  device=device)

  optimizer = AdamW(autoencoder.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay'])
  num_epochs = config['num_epochs']
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      name="cosine_with_min_lr",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps,
      scheduler_specific_kwargs={
          'min_lr_rate': config['end_learning_rate_ratio']
      },
  )

  print(f'{task_name}')
  print('Total trainable parameters: %d' % count_parameters(autoencoder))

  tb_writer = SummaryWriter(config['log_dir'])
  reg_coeff = config['reg_coeff']
  for epoch in range(num_epochs):
    autoencoder.train()
    for step, input_batch in enumerate(train_dataloader):
      for k in ['input_feature']:
        input_batch[k] = input_batch[k].to(device)

      losses = train_autoencoder_step(autoencoder, input_batch)
      loss = losses['l2_loss'] + reg_coeff * losses['kl_loss' if 'kl_loss' in
                                                    losses else 'l1_loss']
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      if step % 100 == 0:
        print('Epoch %d Step %d: Loss %.4f %.4f %.4f LR %.2E' %
              (epoch, step, loss.detach().cpu().numpy(),
               losses['l2_loss'].detach().cpu().numpy(),
               losses['l1_loss'].detach().cpu().numpy(),
               lr_scheduler.get_last_lr()[0]))
        # Log to TensorBoard
        for k in losses:
          tb_writer.add_scalar(f"Train/{k}", losses[k],
                               lr_scheduler._step_count)
        tb_writer.add_scalar(f"Train/loss_total", loss,
                             lr_scheduler._step_count)
        tb_writer.add_scalar(f"LR",
                             lr_scheduler.get_last_lr()[0],
                             lr_scheduler._step_count)
    print('Epoch %d Done.' % epoch)
    # run eval
    autoencoder.eval()
    for split, eval_dataloader in {
        'train': train_dataloader,
        'val': val_dataloader
    }.items():
      agg_metrics = collections.defaultdict(list)
      with torch.no_grad():
        for eval_batch in eval_dataloader:
          for k in ['input_feature']:
            eval_batch[k] = eval_batch[k].to(device)
          metrics = train_autoencoder_step(autoencoder, eval_batch)
          metrics['loss'] = metrics['l2_loss'] + reg_coeff * metrics['l1_loss']
        for k in metrics:
          agg_metrics[k].append(metrics[k].detach().cpu().numpy())
      for k in agg_metrics:
        tb_writer.add_scalar(f"Eval-{split}/{k}", losses[k], epoch)
  torch.save(autoencoder, os.path.join(config['model_dir'], f'{task_name}.pt'))
  tb_writer.flush()
  tb_writer.close()
  return autoencoder
