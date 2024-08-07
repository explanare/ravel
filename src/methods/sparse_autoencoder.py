"""Define a featurizer and features with a sparse autoencoder.

Adapted from https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing#scrollTo=Kn1E_44gCa-Z.
"""

import pyvene as pv
import torch


class SparseAutoencoder(torch.nn.Module):
  """Sparse Autoencoder with a single-layer encoder and a single-layer decoder."""

  def __init__(self, embed_dim, latent_dim, device):
    super().__init__()
    self.dtype = torch.float32
    self.latent_dim = latent_dim
    self.encoder = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, self.latent_dim, bias=True, device=device),
        torch.nn.ReLU())
    self.decoder = torch.nn.Sequential(
        torch.nn.Linear(self.latent_dim, embed_dim, bias=True, device=device))
    self.autoencoder_losses = {}

  def encode(self, x, normalize_input=True):
    if normalize_input:
      x = x - self.decoder[0].bias
    latent = self.encoder(x)
    return latent

  def decode(self, z):
    return self.decoder(z)

  def forward(self, base):
    base_type = base.dtype
    base = base.to(self.dtype)
    self.autoencoder_losses.clear()
    z = self.encode(base)
    base_reconstruct = self.decode(z)
    # The sparsity objective.
    l1_loss = torch.nn.functional.l1_loss(z, torch.zeros_like(z))
    # The reconstruction objective.
    l2_loss = torch.mean((base_reconstruct - base)**2)
    self.autoencoder_losses['l1_loss'] = l1_loss
    self.autoencoder_losses['l2_loss'] = l2_loss
    return {'latent': z, 'output': base_reconstruct.to(base_type)}

  def get_autoencoder_losses(self):
    return self.autoencoder_losses


class AutoencoderIntervention(pv.TrainableIntervention):
  """Intervene in the latent space of an autoencoder."""

  def __init__(self, embed_dim, **kwargs):
    super().__init__()
    del embed_dim
    self.autoencoder = (kwargs['autoencoder']
                        if 'autoencoder' in kwargs else None)
    self.inv_dims = kwargs['inv_dims'] if 'inv_dims' in kwargs else None

  def set_interchange_dim(self, interchange_dim):
    pass

  def forward(self, base, source, subspaces=None):
    base_outputs = self.autoencoder(base)
    source_outputs = self.autoencoder(source)
    if self.inv_dims is not None:
      # Test intervention.
      z_inv = base_outputs['latent']
      z_inv[..., self.inv_dims] = source_outputs['latent'][..., self.inv_dims]
      inv_output = self.autoencoder.decode(z_inv)
    else:
      # Test reconstruction.
      inv_output = base_outputs['output']
    return inv_output.to(base.dtype)
