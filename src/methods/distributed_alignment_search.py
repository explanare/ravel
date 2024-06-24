"""Define a featurizer and features with low-rank parameterized DAS."""

import pyvene as pv
import torch


class LowRankRotatedSpaceIntervention(pv.TrainableIntervention):
  """Intervene in the rotated subspace defined by (low-rank) DAS."""

  def __init__(self, embed_dim, **kwargs):
    super().__init__()
    self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
        torch.nn.Linear(embed_dim,
                        kwargs["low_rank_dimension"],
                        bias=False))
    self.embed_dim = embed_dim

  def forward(self, base, source, subspaces=None):
    input_dtype, model_dtype = base.dtype, self.rotate_layer.weight.dtype
    base, source = base.to(model_dtype), source.to(model_dtype)
    rotated_base = self.rotate_layer(base)
    rotated_source = self.rotate_layer(source)
    # Apply interchange interventions.
    output = base + torch.matmul(
        (rotated_source - rotated_base), self.rotate_layer.weight)
    return output.to(input_dtype)
