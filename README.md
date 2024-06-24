# RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations

Individual neurons participate in the representation of multiple high-level concepts. To what extent can different interpretability methods successfully disentangle these roles?  To help address this question, we present a benchmark: RAVEL (Resolving Attribute–Value Entanglements in Language Models).


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

RAVEL contains five types of entities, each with at least 500 instances, at least 4 attributes, and at least 50 prompt templates, as shown in the table below.

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


Each interpretability method defines a bijective featurizer $\mathcal{F}$ (e.g., a rotation matrix or sparse autoencoder), and identify a feature $F$ that represents the target concept (e.g., a linear subspace of the residual stream in a Transformer that represents "country"). We apply interchange interventions on the feature that localizes target concept and evaluate the causal effects.

The main evaluation logic is implemented in the function [`utils.intervention_utils.eval_with_interventions`](https://github.com/explanare/ravel/blob/main/src/utils/intervention_utils.py), with each method implements its own intervention logic in [src/methods](https://github.com/explanare/ravel/blob/main/src/methods). 


### Interpretablity Methods

We evaluate the following interpretability method:
* PCA
* Sparse Autoencoder (SAE)
* Linear adversarial probing (RLAP)
* Differential Binary Masking (DBM)
* Distributed Alignment Search (DAS)
* Multi-task extensions of DBM and DAS

You can find implementations of these methods in the [src/methods](https://github.com/explanare/ravel/tree/main/src/methods) directory.

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

You can find examples of interventions with featurizers in [scr/methods](https://github.com/explanare/ravel/blob/main/src/methods), where `AutoencoderIntervention` is an example that specifies which subset of features to intervene, while `LowRankRotatedSpaceIntervention` is an example that implicitly defines features through the low-rank rotation matrix.
