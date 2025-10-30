# RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations

Individual neurons participate in the representation of multiple high-level concepts. To what extent can different interpretability methods successfully disentangle these roles?  To help address this question, we present a benchmark: RAVEL (Resolving Attribute–Value Entanglements in Language Models).


## :loudspeaker: Updates

* RAVEL has been integrated into multiple mech interp benchmarks! Besides using this repo, you can also evaluate your methods on RAVEL through one of the following interfaces:
  * [MIB](https://huggingface.co/spaces/mib-bench/leaderboard): A mechanistic interpretability benchmark with two tracks. RAVEL is part of the Causal Variable Localization Track.  
  * [SAEBench](https://www.neuronpedia.org/sae-bench/info): A comprehensive evaluation suite that measures SAE performance across four fundamental capabilities. RAVEL is used to evaluate the "Feature Disentanglement" capability.
  * [SAE-RAVEL](https://github.com/MaheepChaudhary/SAE-Ravel): A benchmark that evaluates different open-source Sparse Autoencoders for GPT-2 small.

* RAVEL is available on HuggingFace now at [hij/ravel](https://huggingface.co/datasets/hij/ravel)!
  * The HuggingFace version includes 3000+ cities and their attributes, along with the Wikipedia URLs where the attribute values are sourced from. We hope this would allow researchers to further expand the attributes in the dataset.


## :yarn: Quickstart

A demo on how to evaluate Sparse Autoencoder (SAE), Distributed Alignment Search (DAS), and Multi-task Distributed Alignment Search (MDAS) on RAVEL with TinyLlama.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dc9KiVGKqwI6Etpf67OUXLey_McFPtQz)


## Requirements

* Install dependencies in `requirements.txt`

  ```setup
  pip install -r requirements.txt
  ```
* Install [pyvene](https://github.com/stanfordnlp/pyvene) from GitHub:

  ```setup
  git clone git@github.com:stanfordnlp/pyvene.git
  ``` 

The code has been tested against the pyvene version at commit `d29f9591ca61753d66ba25f6cc3a4c05bab48480`.

## Dataset

RAVEL provides an entity-attribute dataset covering factual, linguistic, and commonsense knowledge. The dataset contains five types of entities, each with at least 500 instances, at least 4 attributes, and at least 50 prompt templates, as shown in the table below.

|Entity Type|Attributes|\#Entities|\#Prompt Templates|
|---|---|---|---|
|City|Country, Language, Latitude, Longitude,Timezone, Continent|3552|150|
|Nobel Laureate|Award Year, Birth Year, Country of Birth, Field, Gender|928|100|
|Verb|Definition, Past Tense, Pronunciation, Singular | 986 | 60 |
| Physical Object|Biological Category, Color, Size, Texture | 563 | 60 |
|Occupation| Duty, Gender Bias, Industry, Work Location | 799 | 50 |

Compared with existing entity-attribute/relation datasets, RAVEL offers two unique features:
* **multiple attributes** per entity to evaluate how well interpretability methods **isolate individual concepts**
* **x10 more entities** per entity type to evaluate how well interpretability methods **generalize**

### Data format

Each `entity_type` is associated with five files:
- entity: `ravel_{entity_type}_entity_attributes.json`
- prompt: `ravel_{entity_type}_attribute_to_prompts.json`
- wiki prompt: `wikipedia_{entity_type}_entity_prompts.json` 
- entity split: `ravel_{entity_type}_entity_to_split.json`
- prompt split: `ravel_{entity_type}_prompt_to_split.json`

The first three contain all the entities and prompt templates.
The last two contain the dataset splits.

The entity file is structured as follows:

```python
{
  "Paris": {
    "Continent": "Europe",
    "Country": "France",
    "Language": "French",
    "Latitude": "49",
    "Longitude": "2",
    "Timezone": "Europe/Paris"
  },
  ...
}
```

The prompt file is structured as follows:

```python
{
  "Country": [
    "%s is a city in the country of",
    ...
  ],
  "Continent": [
    "Los Angeles is a city in the continent of North America. %s is a city in the continent of",
    ...
  ],
  "Latitude": [
    "[{\"city\": \"Bangkok\", \"lat\": \"13.8\"}, {\"city\": \"%s\", \"lat\": \"",
    ...
  ],
  ...
}
```


## A Framework to Evaluate Interpretability Methods

We evaluate whether interpretability methods can disentangle related concepts, e.g., can a method find a feature of hidden activations that isolate the continent a city is in from the country that city is in? If so, an intervention on the feature should change the first without changing the latter, as shown in the figure below.


![An overview of the evaluation framework.](/figures/ravel-overview.svg)

Each interpretability method defines a bijective featurizer $\mathcal{F}$ (e.g., a rotation matrix or sparse autoencoder), and identify a feature $F$ that represents the target concept (e.g., a linear subspace of the residual stream in a Transformer that represents "country"). We apply **interchange interventions** on the feature that localizes target concept and evaluate the causal effects.

The main evaluation logic is implemented in the function [`utils.intervention_utils.eval_with_interventions`](https://github.com/explanare/ravel/blob/main/src/utils/intervention_utils.py), with each method implements its own interchange intervention logic in [src/methods](https://github.com/explanare/ravel/blob/main/src/methods). 


### Generating Evaluation Data

A core operation in our evaluation framework is interchange intervention, which puts models into counterfactual states that allow us to isolate the causal effects of interest. Interchange intervention involves a pair of examples, which are referred to as `base` and `source`. For each pair, we specify the desired model output upon interventions, namely, whether the output should match the attribute value of the base entity or the attribute value of the source entity.


Each evaluation example is structured as follows:

```python
{
  'input': 'city to country: Rome is in Italy. Tokyo is in',
  'label': ' Japan',
  'source_input': ' in what is now southern Vancouver',
  'source_label': ' Island',
  'inv_label': ' Canada',
  'split': 'city to country: Rome is in Italy. %s is in',
  'source_split': ' in what is now southern %s',
  'entity': 'Tokyo',
  'source_entity': 'Vancouver'
}
```

The input and label fields are:
* `input`: the base example input
* `source_input`: the source example input
* `inv_label`: specifies the desired output when the intervention should **cause** the attribute value to change to attribute value of the source entity
* `label`: specifies the desired output when the intervention should **isolate** the attribute, i.e., having no causal effect on the output.

The rest of the fields are for tracking the intervention locations and aggreating metrics.

#### Demo: Create a RAVEL instance for TinyLlama

A demo on how to create evaluation data using TinyLlama as the target language model. The resulting dataset is used for evaluating the interpretability methods in the Quickstart demo.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16OMrDsnNkRNK0-Xa2Uy2i1QupqsbunAl)


### Evaluating Existing Methods

We have implemented five families of interpretability methods in this repo:
* PCA
* Sparse Autoencoder (SAE)
* Linear adversarial probing (RLAP)
* Differential Binary Masking (DBM)
* Distributed Alignment Search (DAS)
* Multi-task extensions of DBM and DAS

You can find implementations of these methods in the [src/methods](https://github.com/explanare/ravel/tree/main/src/methods) directory.

#### Demo: Evaluate SAE, DAS, and MDAS on RAVEL-TinyLlama

Check out the demo in the Quickstart!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dc9KiVGKqwI6Etpf67OUXLey_McFPtQz)

### Evaluating a New Method

To evaluate a new interpretability method, one simply needs to convert the method into a bijective featurizer:

* $\mathcal{F}$, a function that takes in model representations and outputs (1) a set of features, e.g., a vector (2) a specification of which subset of features localize the target concept, e.g., a set of indicies.
* $\mathcal{F}^{-1}$, a function that takes in the set of features produced by $\mathcal{F}$ and outputs the original model representations

The featurizer will then be evaluated with an interchange intervention as follows:

```python
class InterventionWithNewFeaturizer(pv.TrainableIntervention):
  """Intervene in the featurizer output space."""

  def __init__(self, **kwargs):
    super().__init__()
    # Define a bijective featurizer. A featurizer could be any callable object,
    # such as a subclass of torch.nn.Module
    self.featurizer = ...
    self.featurizer_inverse = ...
    # Specify which subset of features localize the target concept.
    # For some methods, the intervention dimensions are implicitly defined by
    # the featurizer function.
    self.dimensions_to_intervene = ...

  def forward(self, base, source):
    base_features = self.featurizer(base)
    source_features = self.featurizer(source)
    # Apply interchange interventions.
    base_features[..., self.dimensions_to_intervene] = source_features[..., self.dimensions_to_intervene]
    output = self.featurizer_inverse(base_features)
    return output
```

You can find examples of interventions with featurizers in [src/methods](https://github.com/explanare/ravel/blob/main/src/methods), where `AutoencoderIntervention` is an example that specifies which subset of features to intervene, while `LowRankRotatedSpaceIntervention` is an example that implicitly defines features through the low-rank rotation matrix.


# Citation

If you use our dataset or method implmentations, please consider citing the following work. For each interpretablity method, please also consider citing their original papers -- you can find a list of related work in each method section in our paper.

```
@inproceedings{huang-etal-2024-ravel,
    title = "{RAVEL}: Evaluating Interpretability Methods on Disentangling Language Model Representations",
    author = "Huang, Jing  and
      Wu, Zhengxuan  and
      Potts, Christopher  and
      Geva, Mor  and
      Geiger, Atticus",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.470",
    pages = "8669--8687",
}

```

If you use the `pyvene` framework, please also consider citing the following:

```
@inproceedings{wu-etal-2024-pyvene,
    title = "pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions",
    author = "Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Goodman, Noah and Manning, Christopher and Potts, Christopher",
    editor = "Chang, Kai-Wei and Lee, Annie and Rajani, Nazneen",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-demo.16",
    pages = "158--165",
}
```
