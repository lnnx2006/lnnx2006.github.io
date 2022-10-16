---
layout: post
title: Introduction to TensorFlow GNN
date: 2022-06-23 00:00:00
description: A Graph Attention baseline for drug cardiotoxicity detection
tags: python graphs neural-networks tensorflow-gnn tensorflow-datasets
categories: programming
comments: true
---

For my first post in this blog, I've decided to write a beginner-level introduction to Graph Neural Networks with TensorFlow-GNN.
An interactive version of this post, where you can run and modify the code, is available as a [Kaggle notebook](https://www.kaggle.com/code/fidels/introduction-to-tf-gnn) and also on [GoogleColab](https://colab.research.google.com/drive/1FQPBpdrd17WzX_rcFAMsIaN8EK0rP4Ai?usp=sharing). 
Please let me know if you find any bugs or errors, and also if you thought this was useful or you'd like more details about some particular point.

**[July 1, 2022 UPDATE]** The [Kaggle notebook](https://www.kaggle.com/code/fidels/introduction-to-tf-gnn) version of this post has been awarded the [Google Open Source Expert Prize](https://www.kaggle.com/google-oss-expert-prize) for the month of July.

**[July 5, 2022 UPDATE]** I've now updated this notebook to work with the latest TF-GNN release, v0.2.0; some typos were also corrected, and more resources have been added to [§5.1](#conclusions_resources).

## <a id="contents">Contents</a>

- [1. Introduction](#introduction)
    - [1.1 Graphs and Graph Neural Networks](#introduction_graphs)
    - [1.2 TensorFlow GNN](#introduction_tfgnn)
    - [1.3 Graph classification](#introduction_graph_classification)
    - [1.4 Dependencies and imports](#introduction_dependencies)
- [2. Data preparation](#dataprep)
    - [2.1 The `DatasetProvider` protocol](#dataprep_datasetprovider)
    - [2.2 Data inspection](#dataprep_inspection)
    - [2.3 `GraphTensor` batching](#dataprep_batching)
- [3. Vanilla MPNN models](#mpnn)
    - [3.1 Initial graph embedding](#mpnn_embedding)
    - [3.2 Stacking message-passing layers](#mpnn_stack)
    - [3.3 Model construction](#mpnn_construction)
- [4. Graph binary classification](#classification)
    - [4.1 Task specification](#classification_task)
    - [4.2 Training](#classification_training)
    - [4.3 Metric visualization](#classification_metrics)
- [5. Conclusions](#conclusions)
    - [5.1 Resources and acknowledgements](#conclusions_resources)

<div class="section separator"></div>

## <a id="introduction">1. Introduction</a>

<div class="subsection separator"></div>

### <a id="introduction_graphs">1.1 Graphs and Graph Neural Networks</a>

<div class="text separator"></div>

A _[graph](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics))_ consists of a collection $$\mathcal{V}=\{1,\dots,n\}$$ of _nodes_, sometimes also called _vertices_, and a set of _edges_ $$\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$$. An edge $$(i,j) \in \mathcal{E}$$, also denoted $$i\to j$$, represents a relation from node $$i$$ to node $$j$$ which in the general case is directed (_i.e._ does not necessarily imply the converse relation $$j\to i$$). _Heterogeneous_ graphs can have different types of nodes and/or edges, and _decorated_ graphs may have features associated to their nodes, $$h_i \in \mathbb{R}^{d_\mathcal{V}}$$ for $$i=1,\dots,n$$, and/or their edges, $$h_{i\to j} \in \mathbb{R}^{d_\mathcal{E}}$$ for $$(i,j)\in\mathcal{E}$$.

Graphs are clearly a very general and powerful concept, which can be used to represent many different kinds of data. For example:
- A social network may be thought of as a graph where the nodes correspond to different users, and the edges correspond to their relationships (_i.e._ "friendship", "follower", etc).
- A country may be represented as a graph where cities are considered to be nodes, with roads connecting them playing the role of edges.
- ...

---

[Graph Neural Networks (GNN)](https://distill.pub/2021/gnn-intro/) are one way in which we can apply neural networks to graph-structured data in order to learn to make predictions about it. Their architecture typically involves stacking _message passing_ layers, each of which updates the features of each node $$i$$ in the graph by applying some function to the features of its neighbors, _i.e._ all the nodes $$j$$ which have edges going from $$j$$ to $$i$$. More formally, a message passing layer implements the transformation

$$h_i' = f_{\theta_\mathcal{V}}\left(h_i, \tilde{\sum}_{(j,i) \in \mathcal{E}} g_{\theta_\mathcal{E}}\left(h_j, h_{j\to i}\right)\right) \qquad\text{for}\qquad i = 1, \dots, n,$$

where $$f_{\theta_\mathcal{V}}$$ is a neural network applied at each node with parameters $$\theta_\mathcal{V}$$, and $$g_{\theta_\mathcal{E}}$$ is a neural network applied at each edge with parameters $$\theta_\mathcal{E}$$. Note that the latter may be trivial if edges do not have features. In the formula above $$\tilde\sum$$ in fact represents _any_ orderless aggregation function, which can be a conventional summation, an averaging operation, etc. The key point is that, being orderless, a GNN constructed in this way will be invariant under the permutation of the node labels $$i=1,\dots,n$$, a property which of course should hold since the labels themselves are generally speaking arbitrary: the information is encoded in the graph structure itself, not the particular labels chosen to represent it.

### <a id="introduction_tfgnn">1.2 TensorFlow GNN</a>

[TensorFlow GNN (TF-GNN)](https://github.com/tensorflow/gnn) is a fairly recent addition to the TensorFlow ecosystem, having been first released in late 2021. At this time there are relatively few resources available explaining how to use TF-GNN in practice (see [§5.1](#conclusions_resources) for more); this notebook attempts to start filling the gap by providing a basic example of the application of TF-GNN to a real-world problem, namely the classification of molecules to detect cardiotoxicity.

TF-GNN is very powerful and flexible, and can deal with large, heterogeneous graphs. We will however keep things simple here, and work instead with rather small, homogeneous graphs having few features. The main focus will be on showcasing the basic concepts and design abstractions introduced in TF-GNN to facilitate and accelerate end-to-end training of GNN models, particularly:

- [`GraphTensor` objects](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/graph_tensor.md), encapsulating graph structure and features in a TF-compatible format.
- [Pre-built GraphUpdate layers](https://github.com/tensorflow/gnn/tree/main/tensorflow_gnn/models) implementing some of the message-passing protocols most commonly used in the literature.
- The [orchestrator](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md) and related protocols, which can be used to skip most of the technical boilerplate that would otherwise be required for end-to-end training.

### <a id="introduction_graph_classification">1.3 Graph classification</a>

A prototypical example of application of GNNs are _graph classification_ problems, whereupon we are tasked to predict the _class_ of a given graph among a finite set of possible options. In the supervised setting, we have a set of labeled graphs from which we can learn the parameters $$\{\theta_\mathcal{V}, \theta_{\mathcal{E}}\}$$ of the GNN by minimizing a categorical crossentropy loss for our predictions. These are obtained from the GNN after applying one final aggregation or _pooling_ operation over all nodes, which again should be orderless to preserve the important property that the output is invariant under relabeling of the nodes.

In this notebook we will undertake the classification of molecules which have been labeled as being either toxic or non-toxic in the [CardioTox dataset](https://www.tensorflow.org/datasets/catalog/cardiotox) introduced by:

**<a id="cardiotox">[1]</a>** K. Han, B. Lakshminarayanan and J. Liu, "Reliable Graph Neural Networks for Drug Discovery Under Distributional Shift," ([arxiv:2111.12951](https://arxiv.org/abs/2111.12951))

We will take the nodes of our graphs to be the atoms in a molecule, with edges representing atomic bonds between them. Thus, our problem is a **binary classification** one (because we have just two classes) on **homogeneous graphs** (because we have only one type of nodes and one type of edges) which are both **node and edge decorated** (because, as will be seen later, the data contains features for atoms and for the bonds between them).

---

The authors of [[1]](#cardiotox) introduce various clever architectural choices to improve the performance of their model and reduce the risk of false-negative predictions, which are particularly undesirable in a toxicity-detection problem. For illustration purposes here we will not attempt to reproduce their results in full generality, but instead implement a very simple baseline based on Graph Attention Networks introduced in:

**<a id="gat">[2]</a>** P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò and Y. Bengio "Graph Attention Networks," ([arxiv:1710.10903](https://arxiv.org/abs/1710.10903))

**<a id="gat2">[3]</a>** S. Brody, U. Alon and E. Yahav, "How Attentive are Graph Attention Networks?," ([arxiv:2105.14491](https://arxiv.org/abs/2105.14491))

The interested reader can later try different improvements to the code in this notebook, and thanks to TF-GNN these will be easy to introduce once the training pipeline has been established.

### <a id="introduction_dependencies">1.4 Dependencies and imports</a>

Since TF-GNN is currently in an early alpha release stage, installation may have a few hiccups. I have found that the following combination works well in the [Kaggle](https://www.kaggle.com/) notebook environment:

{% highlight python %}
from IPython.display import clear_output

# install non-Python dependencies
!apt-get -y install graphviz graphviz-dev

# Upgrade to TensorFlow 2.8
!pip install tensorflow==2.8 tensorflow-io==0.25.0 tfds-nightly pygraphviz

# Install TensorFlow-GNN
!pip install tensorflow_gnn==0.2.0

# Fix some dependencies
!pip install httplib2==0.20.4

clear_output()
{% endhighlight %}

For [GoogleColab](https://colab.research.google.com/drive/1FQPBpdrd17WzX_rcFAMsIaN8EK0rP4Ai?usp=sharing), you can instead use:

{% highlight python %}
from IPython.display import clear_output

# Install non-Python dependencies
!apt-get -y install graphviz graphviz-dev

# Install Python dependencies
!pip install tfds-nightly pygraphviz

# Install TensorFlow-GNN
!pip install tensorflow_gnn==0.2.0

clear_output()
{% endhighlight %}

Either way, once everything is installed we can proceed:

{% highlight python %}
import pygraphviz as pgv
from tqdm import tqdm
from IPython.display import Image

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_gnn as tfgnn
import tensorflow_datasets as tfds

from tensorflow_gnn import runner
from tensorflow_gnn.models import gat_v2, gnn_template

print(f'Using TensorFlow v{tf.__version__} and TensorFlow-GNN v{tfgnn.__version__}')
print(f'GPUs available: {tf.config.list_physical_devices("GPU")}')
{% endhighlight %}
<pre class="output">
Using TensorFlow v2.8.0 and TensorFlow-GNN v0.2.0
GPUs available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
</pre>

<div class="section separator"></div>

## <a id="dataprep">2. Data preparation</a>

Unlike other types of data, there is no standard encoding for graphs. Indeed, depending on the intended application the graph structure can be represented by:
- An adjacency matrix $$A_{ij}$$ specifying whether the edge $$i\to j$$ is present or not in $$\mathcal{E}$$.
- An adjacency list $$N_i$$ for each node $$i\in\mathcal{V}$$, specifying all the nodes $$j\in\mathcal{V}$$ for which an edge $$(i, j)\in\mathcal{E}$$
- A list of edges $$(i,j)\in\mathcal{E}$$

To address this issue TF-GNN introduces the [`GraphTensor` object](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/graph_tensor.md), which encapsulates both the graph structure and the features of the nodes, edges and the graph itself. These objects follow a [_graph schema_](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/schema.md), which specifies the types of nodes and edges as well as all the features that should be present in the graphs. The first step in any TF-GNN training pipeline should therefore be to convert the input data from whichever format is given into `GraphTensor` format. These `GraphTensor` objects can then be batched and consumed by our TF-GNN models just like a `tf.Tensor`, hugely simplifying our lives in the process.

The goal of this section is to perform the task described above, resulting in `DatasetProvider` objects for our data (see [here](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md#data-reading) and the following section for a description of the `DatasetProvider` protocol). Let us then take a look at the CardioTox dataset, which we can automatically download from TensorFlow Datasets (TF-DS):

{% highlight python %}
dataset_splits, dataset_info = tfds.load('cardiotox', data_dir='data/tfds', with_info=True)

clear_output()

print(dataset_info.description)
{% endhighlight %}
<pre class="output">
Drug Cardiotoxicity dataset [1-2] is a molecule classification task to detect
cardiotoxicity caused by binding hERG target, a protein associated with heart
beat rhythm. The data covers over 9000 molecules with hERG activity.

Note:

1. The data is split into four splits: train, test-iid, test-ood1, test-ood2.

2. Each molecule in the dataset has 2D graph annotations which is designed to
facilitate graph neural network modeling. Nodes are the atoms of the molecule
and edges are the bonds. Each atom is represented as a vector encoding basic
atom information such as atom type. Similar logic applies to bonds.

3. We include Tanimoto fingerprint distance (to training data) for each molecule
in the test sets to facilitate research on distributional shift in graph domain.

For each example, the features include:
  atoms: a 2D tensor with shape (60, 27) storing node features. Molecules with
    less than 60 atoms are padded with zeros. Each atom has 27 atom features.
  pairs: a 3D tensor with shape (60, 60, 12) storing edge features. Each edge
    has 12 edge features.
  atom_mask: a 1D tensor with shape (60, ) storing node masks. 1 indicates the
    corresponding atom is real, othewise a padded one.
  pair_mask: a 2D tensor with shape (60, 60) storing edge masks. 1 indicates the
    corresponding edge is real, othewise a padded one.
  active: a one-hot vector indicating if the molecule is toxic or not. [0, 1]
    indicates it's toxic, otherwise [1, 0] non-toxic.


## References
[1]: V. B. Siramshetty et al. Critical Assessment of Artificial Intelligence
Methods for Prediction of hERG Channel Inhibition in the Big Data Era.
    JCIM, 2020. https://pubs.acs.org/doi/10.1021/acs.jcim.0c00884

[2]: K. Han et al. Reliable Graph Neural Networks for Drug Discovery Under
Distributional Shift.
    NeurIPS DistShift Workshop 2021. https://arxiv.org/abs/2111.12951
</pre>

As mentioned before, we will have only one set of nodes, which we call `'atom'`, and one set of edges, which we call `'bond'`. Of course, all edges have `'atom'`-type nodes at both endpoints. Both nodes and edges have a single feature vector, the former being a 27-dimensional `'atom_features'` vector, and the latter being a 12-dimensional `'bond_features'` vector. Moreover, the graphs themselves have global features giving their _context_, in this case a toxicity class `'toxicity'`, which is in fact the label we want to predict, and a molecule id `'molecule_id'` which we will mostly ignore.

All of the above can be encoded in the following graph schema specifying the structure and contents of our graphs:

{% highlight python %}
graph_schema_pbtxt = """
node_sets {
  key: "atom"
  value {
    description: "An atom in the molecule."

    features {
      key: "atom_features"
      value: {
        description: "[DATA] The features of the atom."
        dtype: DT_FLOAT
        shape { dim { size: 27 } }
      }
    }
  }
}

edge_sets {
  key: "bond"
  value {
    description: "A bond between two atoms in the molecule."
    source: "atom"
    target: "atom"

    features {
      key: "bond_features"
      value: {
        description: "[DATA] The features of the bond."
        dtype: DT_FLOAT
        shape { dim { size: 12 } }
      }
    }
  }
}

context {
  features {
    key: "toxicity"
    value: {
      description: "[LABEL] The toxicity class of the molecule (0 -> non-toxic; 1 -> toxic)."
      dtype: DT_INT64
    }
  }
  
  features {
    key: "molecule_id"
    value: {
      description: "[LABEL] The id of the molecule."
      dtype: DT_STRING
    }
  }
}
"""
{% endhighlight %}

This schema is a textual protobuf, which we can parse to obtain a `GraphTensorSpec` (think `TensorSpec` for `GraphTensor` objects):

{% highlight python %}
graph_schema = tfgnn.parse_schema(graph_schema_pbtxt)
graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
{% endhighlight %}

We should then convert the input dataset into `GraphTensor` objects complying to `graph_spec`, which we will do with the following helper function:

{% highlight python %}
def make_graph_tensor(datapoint):
    """
    Convert a datapoint from the TF-DS CardioTox dataset into a `GraphTensor`.
    """
    # atom_mask is non-zero only for real atoms
    # [ V, ]
    atom_indices = tf.squeeze(tf.where(datapoint['atom_mask']), axis=1)
    
    # only keep features of real atoms
    # [ V, 27 ]
    atom_features = tf.gather(datapoint['atoms'], atom_indices)
    
    # restrict the bond mask to real atoms
    # [ V, V ]
    pair_mask = tf.gather(tf.gather(datapoint['pair_mask'], atom_indices, axis=0), atom_indices, axis=1)
    
    # restrict the bond features to real atoms
    # [ V, V, 12 ]
    pairs = tf.gather(tf.gather(datapoint['pairs'], atom_indices, axis=0), atom_indices, axis=1)
    
    # pair_mask is non-zero only for real bonds
    # [ E, 2 ]
    bond_indices = tf.where(pair_mask)
    
    # only keep features of real bonds
    # [ E, 12 ]
    bond_features = tf.gather_nd(pairs, bond_indices)
    
    # separate sources and targets for each bond
    # [ E, ]
    sources, targets = tf.unstack(tf.transpose(bond_indices))

    # active is [1, 0] for non-toxic molecules, [0, 1] for toxic molecules
    # [ ]
    toxicity = tf.argmax(datapoint['active'])
    
    # the molecule_id is included for reference
    # [ ]
    molecule_id = datapoint['molecule_id']

    # create a GraphTensor from all of the above
    atom = tfgnn.NodeSet.from_fields(features={'atom_features': atom_features},
                                     sizes=tf.shape(atom_indices))
    
    atom_adjacency = tfgnn.Adjacency.from_indices(source=('atom', tf.cast(sources, dtype=tf.int32)),
                                                  target=('atom', tf.cast(targets, dtype=tf.int32)))
    
    bond = tfgnn.EdgeSet.from_fields(features={'bond_features': bond_features},
                                     sizes=tf.shape(sources),
                                     adjacency=atom_adjacency)
    
    context = tfgnn.Context.from_fields(features={'toxicity': [toxicity], 'molecule_id': [molecule_id]})
    
    return tfgnn.GraphTensor.from_pieces(node_sets={'atom': atom}, edge_sets={'bond': bond}, context=context)
{% endhighlight %}

We can now map this function over the datasets to have them stream `GraphTensor` objects:

{% highlight python %}
train_dataset = dataset_splits['train'].map(make_graph_tensor)
{% endhighlight %}

{% highlight python %}
graph_tensor = next(iter(train_dataset))
graph_tensor
{% endhighlight %}
<pre class="output">
GraphTensor(
  context=Context(features={'toxicity': <tf.Tensor: shape=(1,), dtype=tf.int64>, 'molecule_id': <tf.Tensor: shape=(1,), dtype=tf.string>}, sizes=[1], shape=(), indices_dtype=tf.int32),
  node_set_names=['atom'],
  edge_set_names=['bond'])
</pre>

And check that the `GraphTensor` thus produced are compatible with the `GraphTensorSpec` we defined before:

{% highlight python %}
graph_spec.is_compatible_with(graph_tensor)
{% endhighlight %}
<pre class="output">
True
</pre>

However, to avoid processing the data multiple times, which would slow down all of our input pipeline, it is convenient to first dump all the data into TFRecord files. Later on we can easily load these instead of the original TF-DS datasets over which we mapped the `make_graph_tensor` function.

**NOTE:** The `create_tfrecords` method below works nicely, is rather general and could be immediately reused for other small-scale applications. For large-scale datasets, however, alternative approaches using `tf.data.Dataset.cache` or `tf.data.Dataset.snapshot` would be preferable, as they would allow for more optimizations such as _e.g._ sharding.

{% highlight python %}
def create_tfrecords(dataset_splits, dataset_info):
    """
    Dump all splits of the given dataset to TFRecord files.
    """
    for split_name, dataset in dataset_splits.items():
        filename = f'data/{dataset_info.name}-{split_name}.tfrecord'
        print(f'creating {filename}...')
        
        # convert all datapoints to GraphTensor
        dataset = dataset.map(make_graph_tensor, num_parallel_calls=tf.data.AUTOTUNE)
        
        # serialize to TFRecord files
        with tf.io.TFRecordWriter(filename) as writer:
            for graph_tensor in tqdm(iter(dataset), total=dataset_info.splits[split_name].num_examples):
                example = tfgnn.write_example(graph_tensor)
                writer.write(example.SerializeToString())
{% endhighlight %}

{% highlight python %}
create_tfrecords(dataset_splits, dataset_info)
{% endhighlight %}
<pre class="output">
creating data/cardiotox-train.tfrecord...
100%|██████████| 6523/6523 [00:46<00:00, 140.23it/s]
creating data/cardiotox-validation.tfrecord...
100%|██████████| 1631/1631 [00:11<00:00, 138.38it/s]
creating data/cardiotox-test.tfrecord...
100%|██████████| 839/839 [00:05<00:00, 142.62it/s]
creating data/cardiotox-test2.tfrecord...
100%|██████████| 177/177 [00:01<00:00, 140.72it/s]
</pre>

Finally, we can use the `TFRecordDatasetProvider` class to create `DatasetProvider`-compliant objects that read these TFRecord files and provide `tf.data.Dataset` objects for us to use through their `get_dataset` method:

{% highlight python %}
train_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/cardiotox-train.tfrecord')
valid_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/cardiotox-validation.tfrecord')
test1_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/cardiotox-test.tfrecord')
test2_dataset_provider = runner.TFRecordDatasetProvider(file_pattern='data/cardiotox-test2.tfrecord')
{% endhighlight %}

### <a id="dataprep_datasetprovider">2.1 The DatasetProvider protocol</a>

Each of the `DatasetProvider` defined above conventionally produces a dataset of serialized `GraphTensor` objects, which we need to parse before we can inspect. This is mentioned here for reference purposes only: the [orchestrator](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md#orchestration) will transparently deal with this during actual training.

To obtain the dataset we need to provide an input context:<a id="input_context"></a>

{% highlight python %}
train_dataset = train_dataset_provider.get_dataset(context=tf.distribute.InputContext())
{% endhighlight %}

We then map `tfgnn.parse_single_example` over this dataset, specifying the appropriate `GraphTensorSpec` for our graphs:

{% highlight python %}
train_dataset = train_dataset.map(lambda serialized: tfgnn.parse_single_example(serialized=serialized, spec=graph_spec))
{% endhighlight %}

And we can then stream `GraphTensor` objects as before

{% highlight python %}
graph_tensor = next(iter(train_dataset))
graph_tensor
{% endhighlight %}
<pre class="output">
GraphTensor(
  context=Context(features={'toxicity': <tf.Tensor: shape=(1,), dtype=tf.int64>, 'molecule_id': <tf.Tensor: shape=(1,), dtype=tf.string>}, sizes=[1], shape=(), indices_dtype=tf.int32),
  node_set_names=['atom'],
  edge_set_names=['bond'])
</pre>

### <a id="dataprep_inspection">2.2 Data inspection</a>

The node and edge features are not particularly illustrative, but we can nevertheless access them directly if necessary. First, note that this particular molecule has the following number $$V = \left\vert\mathcal{V}\right\vert$$ of atoms:

{% highlight python %}
graph_tensor.node_sets['atom'].sizes
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(1,), dtype=int32, numpy=array([33], dtype=int32)>
</pre>

Their features are collected in a tensor of shape `(V, 27)`, which we can access like so:

{% highlight python %}
graph_tensor.node_sets['atom']['atom_features']
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(33, 27), dtype=float32, numpy=
array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.]], dtype=float32)>
</pre>

Similarly, the number $$E = \left\vert\mathcal{E}\right\vert$$ of bonds in the molecule is:

{% highlight python %}
graph_tensor.edge_sets['bond'].sizes
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(1,), dtype=int32, numpy=array([68], dtype=int32)>
</pre>

Their features are collected in a tensor of shape `(E, 12)`, which we can access like so:

{% highlight python %}
graph_tensor.edge_sets['bond']['bond_features']
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(68, 12), dtype=float32, numpy=
array([[1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.],
       [0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.]], dtype=float32)>
</pre>

The ids of the edge endpoints are then stored in a couple of tensors of shape `(E,)`

{% highlight python %}
graph_tensor.edge_sets['bond'].adjacency.source
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(68,), dtype=int32, numpy=
array([ 0,  1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  5,  5,  5,  6,  7,  7,
        8,  8,  9,  9, 10, 10, 10, 11, 12, 12, 13, 13, 14, 14, 15, 15, 15,
       16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 21, 22, 23, 23, 23, 24, 25,
       25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 30, 31, 31, 31, 32],
      dtype=int32)>
</pre>

{% highlight python %}
graph_tensor.edge_sets['bond'].adjacency.target
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(68,), dtype=int32, numpy=
array([ 1,  0,  2, 31,  1,  3, 23,  2,  4,  3,  5,  4,  6,  7,  5,  5,  8,
        7,  9,  8, 10,  9, 11, 12, 10, 10, 13, 12, 14, 13, 15, 14, 16, 17,
       15, 15, 18, 17, 19, 18, 20, 19, 21, 22, 20, 20,  2, 24, 25, 23, 23,
       26, 30, 25, 27, 26, 28, 27, 29, 28, 30, 25, 29, 31,  1, 30, 32, 31],
      dtype=int32)>
</pre>

Finally, global information about the graph is provided by its context

{% highlight python %}
graph_tensor.context['toxicity']
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(1,), dtype=int64, numpy=array([0])>
</pre>

{% highlight python %}
graph_tensor.context['molecule_id']
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(1,), dtype=string, numpy=
array([b'CC1=C(C/C=C(\\C)CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C(=O)c2ccccc2C1=O'],
      dtype=object)>
</pre>

With all of this we can write the following helper function to visualize the graphs:

{% highlight python %}
def draw_molecule(graph_tensor):
    """
    Plot the `GraphTensor` representation of a molecule.
    """
    (molecule_id,) = graph_tensor.context['molecule_id'].numpy()
    (toxicity,) = graph_tensor.context['toxicity'].numpy()

    sources = graph_tensor.edge_sets['bond'].adjacency.source.numpy()
    targets = graph_tensor.edge_sets['bond'].adjacency.target.numpy()

    pgvGraph = pgv.AGraph()
    pgvGraph.graph_attr['label'] = f'toxicity = {toxicity}\n\nmolecule_id = {molecule_id.decode()}'

    for edge in zip(sources, targets):
        pgvGraph.add_edge(edge)

    return Image(pgvGraph.draw(format='png', prog='dot'))
{% endhighlight %}

{% highlight python %}
draw_molecule(graph_tensor)
{% endhighlight %}
{% include figure.html path="assets/img/blog/tfgnn-intro/molecule.png" class="img-fluid rounded z-depth-1" %}

### <a id="dataprep_batching">2.3 GraphTensor batching</a>

`GraphTensor` datasets can be batched as usual, resulting in new datasets that produce higher-rank `GraphTensor` objects:

{% highlight python %}
batch_size = 64
batched_train_dataset = train_dataset.batch(batch_size)
{% endhighlight %}

{% highlight python %}
graph_tensor_batch = next(iter(batched_train_dataset))
graph_tensor_batch.rank
{% endhighlight %}
<pre class="output">
1
</pre>

The resulting `GraphTensor` now contains features in the form of `tf.RaggedTensor`, since different graphs can have different numbers of nodes and edges:

{% highlight python %}
graph_tensor_batch.node_sets['atom']['atom_features'].shape
{% endhighlight %}
<pre class="output">
TensorShape([64, None, 27])
</pre>

{% highlight python %}
graph_tensor_batch.edge_sets['bond']['bond_features'].shape
{% endhighlight %}
<pre class="output">
TensorShape([64, None, 12])
</pre>

where the shapes now correspond to `(batch_size, V, 27)` and `(batch_size, E, 12)`.

However, all layers in TF-GNN expect scalar graphs as their inputs, so before actually using a batch of graphs we should always "merge" the different graphs in the batch into a single graph with multiple disconnected components (of which TF-GNN automatically keeps track):

{% highlight python %}
scalar_graph_tensor = graph_tensor_batch.merge_batch_to_components()
scalar_graph_tensor.rank
{% endhighlight %}
<pre class="output">
0
</pre>

Now the atom features again have shape `(V', 27)`, where $$V' = \sum_{k=1}^{\rm batch\_size} V_k$$

{% highlight python %}
scalar_graph_tensor.node_sets['atom']['atom_features'].shape
{% endhighlight %}
<pre class="output">
TensorShape([1562, 27])
</pre>

And the bond fetures have shape $$(E', 12)$$ with $$E' = \sum_{k=1}^{\rm batch\_size} E_k$$

{% highlight python %}
scalar_graph_tensor.edge_sets['bond']['bond_features'].shape
{% endhighlight %}
<pre class="output">
TensorShape([3370, 12])
</pre>

We should note, however, that once more the [orchestrator](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md#orchestration) will transparently take care of batching and merging components for us, so that we needn't worry about this as long as we don't customize the training routines.

<div class="section separator"></div>

## <a id="mpnn">3. Vanilla MPNN models</a>

A [common architecture for GNNs](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/gnn_modeling.md) consists of an initial layer which preprocesses the graph features, typically producing hidden states for nodes and/or edges, which is followed by one or more layers of message-passing working as described in the [Introduction](#introduction). The goal of this section is to define a `vanilla_mpnn_model` function which can be used to create such simple GNNs from:

- an initial layer performing the pre-processing
- a layer stacking multiple message-passing layers

### <a id="mpnn_embedding">3.1 Initial graph embedding</a>

For the first of these tasks we will use a `tfgnn.keras.layers.MapFeatures` layer to create hidden state vectors for atoms and bonds from their respective features, by passing these through a dense layer. The resulting hidden states will have dimension `hidden_size`, corresponding in the notation of the [Introduction](#introduction_graphs) to $$d_\mathcal{V}$$ and $$d_\mathcal{E}$$.

The following helper function will create an initial `MapFeatures` layer for the given hyperparameters:
- `hidden_size`: the hidden dimensions $$d_\mathcal{V}$$ and $$d_\mathcal{E}$$
- `activation`: the activation for the dense layers

{% highlight python %}
def get_initial_map_features(hidden_size, activation='relu'):
    """
    Initial pre-processing layer for a GNN (use as a class constructor).
    """
    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'atom':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(node_set['atom_features'])
    
    def edge_sets_fn(edge_set, edge_set_name):
        if edge_set_name == 'bond':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(edge_set['bond_features'])
    
    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,
                                          edge_sets_fn=edge_sets_fn,
                                          name='graph_embedding')
{% endhighlight %}

We can check that the resulting layer replaces the `'atom_features'` and `'bond_features'` with hidden states of the specified dimensions

{% highlight python %}
graph_embedding = get_initial_map_features(hidden_size=128)
{% endhighlight %}

{% highlight python %}
embedded_graph = graph_embedding(scalar_graph_tensor)
{% endhighlight %}

{% highlight python %}
embedded_graph.node_sets['atom'].features
{% endhighlight %}
<pre class="output">
{'hidden_state': <tf.Tensor: shape=(1562, 128), dtype=float32, numpy=
array([[0.15272579, 0.        , 0.        , ..., 0.        , 0.        ,
        0.00244589],
       [0.20299977, 0.15906705, 0.        , ..., 0.2211346 , 0.        ,
        0.23549727],
       [0.20299977, 0.15906705, 0.        , ..., 0.2211346 , 0.        ,
        0.23549727],
       ...,
       [0.3452986 , 0.        , 0.01703222, ..., 0.        , 0.        ,
        0.        ],
       [0.23036027, 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.595324  , 0.        , 0.3375612 , ..., 0.23380664, 0.        ,
        0.11104017]], dtype=float32)>}
</pre>

{% highlight python %}
embedded_graph.edge_sets['bond'].features
{% endhighlight %}
<pre class="output">
{'hidden_state': <tf.Tensor: shape=(3370, 128), dtype=float32, numpy=
array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.19551635, 0.05461648, ..., 0.        , 0.06973472,
        0.        ],
       ...,
       [0.01244138, 0.        , 0.        , ..., 0.        , 0.02111641,
        0.        ],
       [0.01244138, 0.        , 0.        , ..., 0.        , 0.02111641,
        0.        ],
       [0.01244138, 0.        , 0.        , ..., 0.        , 0.02111641,
        0.        ]], dtype=float32)>}
</pre>

Note that both the atom and bond features are now named `'hidden_state'`; we could of course have chosen a different name, but leaving the default `tfgnn.HIDDEN_STATE` will save us from having to specify feature names in what follows.

### <a id="mpnn_stack">3.2 Stacking message-passing layers</a>

To illustrate how to build a stack of message-passing layers, we will use the pre-built [Graph Attention (GAT) [2]](#gat) layers provided in the [`models.gat_v2` module](https://github.com/tensorflow/gnn/tree/main/tensorflow_gnn/models/gat_v2). We then define a message-passing neural network (MPNN) layer successively applying these layers with hyperparameters:
- `hidden_size`: the hidden dimensions $$d_\mathcal{V}$$ and $$d_\mathcal{E}$$
- `hops`: the number of layers in the stack

{% highlight python %}
class MPNN(tf.keras.layers.Layer):
    """
    A basic stack of message-passing Graph Attention layers.
    """
    def __init__(self, hidden_size, hops, name='gat_mpnn', **kwargs):
        self.hidden_size = hidden_size
        self.hops = hops
        super().__init__(name=name, **kwargs)
        
        self.mp_layers = [self._mp_factory(name=f'message_passing_{i}') for i in range(hops)]
    
    def _mp_factory(self, name):
        return gat_v2.GATv2GraphUpdate(num_heads=1,
                                       per_head_channels=self.hidden_size,
                                       edge_set_name='bond',
                                       sender_edge_feature=tfgnn.HIDDEN_STATE,
                                       name=name)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'hops': self.hops
        })
        return config
        
    def call(self, graph_tensor):
        for layer in self.mp_layers:
            graph_tensor = layer(graph_tensor)
        return graph_tensor
{% endhighlight %}

We can now check that this layer processes the embedded graphs that come out from the initial feature map:

{% highlight python %}
mpnn = MPNN(hidden_size=128, hops=8)
{% endhighlight %}

{% highlight python %}
hidden_graph = mpnn(embedded_graph)
{% endhighlight %}

{% highlight python %}
hidden_graph.node_sets['atom'].features
{% endhighlight %}
<pre class="output">
{'hidden_state': <tf.Tensor: shape=(1562, 128), dtype=float32, numpy=
array([[0.        , 0.        , 0.4671823 , ..., 0.18130356, 0.        ,
        0.        ],
       [0.        , 0.        , 0.4360276 , ..., 0.20995092, 0.        ,
        0.        ],
       [0.        , 0.        , 0.4278612 , ..., 0.20583078, 0.        ,
        0.        ],
       ...,
       [0.        , 0.00174278, 0.51690316, ..., 0.20055819, 0.        ,
        0.        ],
       [0.        , 0.01096402, 0.54201955, ..., 0.20254537, 0.        ,
        0.        ],
       [0.        , 0.0008509 , 0.5069808 , ..., 0.19695306, 0.        ,
        0.        ]], dtype=float32)>}
</pre>

{% highlight python %}
hidden_graph.edge_sets['bond'].features
{% endhighlight %}
<pre class="output">
{'hidden_state': <tf.Tensor: shape=(3370, 128), dtype=float32, numpy=
array([[0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , ..., 0.        , 0.        ,
        0.        ],
       [0.        , 0.19551635, 0.05461648, ..., 0.        , 0.06973472,
        0.        ],
       ...,
       [0.01244138, 0.        , 0.        , ..., 0.        , 0.02111641,
        0.        ],
       [0.01244138, 0.        , 0.        , ..., 0.        , 0.02111641,
        0.        ],
       [0.01244138, 0.        , 0.        , ..., 0.        , 0.02111641,
        0.        ]], dtype=float32)>}
</pre>

### <a id="mpnn_construction">3.3 Model construction</a>

We are now ready to combine both ingredients into a `tf.keras.Model` that takes a `GraphTensor` representing a molecule as input, and produces as its output another `GraphTensor` with hidden states for all the atoms. We use Keras' functional API to define a `vanilla_mpnn_model` helper function returning the desired `tf.keras.Model`:

{% highlight python %}
def vanilla_mpnn_model(graph_tensor_spec, init_states_fn, pass_messages_fn):
    """
    Chain an initialization layer and a message-passing stack to produce a `tf.keras.Model`.
    """
    graph_tensor = tf.keras.layers.Input(type_spec=graph_tensor_spec)
    embedded_graph = init_states_fn(graph_tensor)
    hidden_graph = pass_messages_fn(embedded_graph)
    return tf.keras.Model(inputs=graph_tensor, outputs=hidden_graph)
{% endhighlight %}

{% highlight python %}
model = vanilla_mpnn_model(graph_tensor_spec=graph_spec,
                           init_states_fn=graph_embedding,
                           pass_messages_fn=mpnn)
model.summary()
{% endhighlight %}
<pre class="output">
Model: "model_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [()]                      0         
                                                                 
 graph_embedding (MapFeature  ()                       5248      
 s)                                                              
                                                                 
 gat_mpnn (MPNN)             ()                        396288    
                                                                 
=================================================================
Total params: 401,536
Trainable params: 401,536
Non-trainable params: 0
_________________________________________________________________
</pre>

For later convenience, let us encapsulate all of this logic in a function we can use to get model constructors for fixed hyperparameters. The returned constructor by convention takes only the `GraphTensorSpec` of the input graphs for the model, and for good measure our constructor will also add some $$L_2$$ regularization through an `l2_coefficient` hyperparameter:

{% highlight python %}
def get_model_creation_fn(hidden_size, hops, activation='relu', l2_coefficient=1e-3):
    """
    Return a model constructor for a given set of hyperparameters.
    """
    def model_creation_fn(graph_tensor_spec):
        initial_map_features = get_initial_map_features(hidden_size=hidden_size, activation=activation)
        mpnn = MPNN(hidden_size=hidden_size, hops=hops)
        
        model = vanilla_mpnn_model(graph_tensor_spec=graph_tensor_spec,
                                   init_states_fn=initial_map_features,
                                   pass_messages_fn=mpnn)
        model.add_loss(lambda: tf.reduce_sum([tf.keras.regularizers.l2(l2=l2_coefficient)(weight) for weight in model.trainable_weights]))
        return model
    return model_creation_fn
{% endhighlight %}

{% highlight python %}
mpnn_creation_fn = get_model_creation_fn(hidden_size=128, hops=8)
{% endhighlight %}

{% highlight python %}
model = mpnn_creation_fn(graph_spec)
model.summary()
{% endhighlight %}
<pre class="output">
Model: "model_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_4 (InputLayer)        [()]                      0         
                                                                 
 graph_embedding (MapFeature  ()                       5248      
 s)                                                              
                                                                 
 gat_mpnn (MPNN)             ()                        396288    
                                                                 
=================================================================
Total params: 401,536
Trainable params: 401,536
Non-trainable params: 0
_________________________________________________________________
</pre>

<div class="section separator"></div>

## <a id="classification">4. Graph binary classification</a>

<div class="subsection separator"></div>

### <a id="classification_task">4.1 Task specification</a>

Having a GNN model at our disposal, we are now ready to apply it to the task at hand, namely the binary classification of molecules predicting their toxicity. This involves:

1. Adding a readout and prediction head, which computes logits for each class from the features computed by the GNN.
1. Defining the loss function to be minimized, which in this case should be a categorical crossentropy loss.
1. Defining the metrics we are interested in measuring during training and validation.

The [orchestrator](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md) defines the `Task` protocol to achieve these goals, and conveniently provides a pre-implemented `GraphBinaryClassification` class which complies with this protocol. While we could use it as is, for illustration purposes here we will extend its basic implementation in two ways:
- We will include the AUROC metric, which is reported by the authors of [[1]](#cardiotox) (other metrics reported there can be added in a similar fashion).
- We will generalize the readout and prediction head to include a hidden layer.

First, we define a simple wrapper around the `tf.keras.metrics.AUC` class to adapt it to our conventions:

{% highlight python %}
class AUROC(tf.keras.metrics.AUC):
    """
    AUROC metric computation for binary classification from logits.
    
    y_true: true labels, with shape (batch_size,)
    y_pred: predicted logits, with shape (batch_size, 2)
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, tf.math.softmax(y_pred, axis=-1)[:,1])
{% endhighlight %}

Next, we subclass the `GraphBinaryClassification` task and override its `adapt` and `metrics` methods:

{% highlight python %}
class GraphBinaryClassification(runner.GraphBinaryClassification):
    """
    A GraphBinaryClassification task with a hidden layer in the prediction head, and additional metrics.
    """
    def __init__(self, hidden_dim, *args, **kwargs):
        self._hidden_dim = hidden_dim
        super().__init__(*args, **kwargs)
        
    def adapt(self, model):
        hidden_state = tfgnn.pool_nodes_to_context(model.output,
                                                   node_set_name=self._node_set_name,
                                                   reduce_type=self._reduce_type,
                                                   feature_name=self._state_name)
        
        hidden_state = tf.keras.layers.Dense(units=self._hidden_dim, activation='relu', name='hidden_layer')(hidden_state)
        
        logits = tf.keras.layers.Dense(units=self._units, name='logits')(hidden_state)
        
        return tf.keras.Model(inputs=model.inputs, outputs=logits)
    
    def metrics(self):
        return (*super().metrics(), AUROC(name='AUROC'))
{% endhighlight %}

To create an instance of this class we need to specify the node set which will be used to aggregate hidden states for prediction (remember in our case there is only one, `'atom'`) and the number of classes (two, for toxic and non-toxic), as well as the new hyperparameter `hidden_dim`:

{% highlight python %}
task = GraphBinaryClassification(hidden_dim=256, node_set_name='atom', num_classes=2)
{% endhighlight %}

This instance then provides everything we will need for training, namely:
- The loss function

{% highlight python %}
task.losses()
{% endhighlight %}
<pre class="output">
(<keras.losses.SparseCategoricalCrossentropy at 0x7fe2bc861650>,)
</pre>

- The metrics

{% highlight python %}
task.metrics()
{% endhighlight %}
<pre class="output">
(<keras.metrics.SparseCategoricalAccuracy at 0x7fe2bc795550>,
 <keras.metrics.SparseCategoricalCrossentropy at 0x7fe2bc861850>,
 <__main__.AUROC at 0x7fe2bc90a590>)
</pre>

- An `adapt` method to place the readout and prediction head on top of the GNN

{% highlight python %}
classification_model = task.adapt(model)
classification_model.summary()
{% endhighlight %}
<pre class="output">
Model: "model_4"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_4 (InputLayer)           [()]                 0           []                               
                                                                                                  
 graph_embedding (MapFeatures)  ()                   5248        ['input_4[0][0]']                
                                                                                                  
 gat_mpnn (MPNN)                ()                   396288      ['graph_embedding[0][0]']        
                                                                                                  
 input.node_sets (InstancePrope  {'atom': ()}        0           ['gat_mpnn[0][0]']               
 rty)                                                                                             
                                                                                                  
 input.sizes (InstanceProperty)  (1,)                0           ['input.node_sets[0][0]']        
                                                                                                  
 tf.math.cumsum (TFOpLambda)    (1,)                 0           ['input.sizes[0][0]']            
                                                                                                  
 tf.math.reduce_sum (TFOpLambda  ()                  0           ['input.sizes[0][0]']            
 )                                                                                                
                                                                                                  
 tf.ones_like (TFOpLambda)      (1,)                 0           ['tf.math.cumsum[0][0]']         
                                                                                                  
 tf.__operators__.add (TFOpLamb  ()                  0           ['tf.math.reduce_sum[0][0]']     
 da)                                                                                              
                                                                                                  
 tf.math.unsorted_segment_sum (  (None,)             0           ['tf.ones_like[0][0]',           
 TFOpLambda)                                                      'tf.__operators__.add[0][0]',   
                                                                  'tf.math.cumsum[0][0]']         
                                                                                                  
 tf.math.cumsum_1 (TFOpLambda)  (None,)              0           ['tf.math.unsorted_segment_sum[0]
                                                                 [0]']                            
                                                                                                  
 input._get_features_ref_4 (Ins  {'hidden_state': (N  0          ['input.node_sets[0][0]']        
 tanceProperty)                 one, 128)}                                                        
                                                                                                  
 tf.__operators__.getitem (Slic  (None,)             0           ['tf.math.cumsum_1[0][0]',       
 ingOpLambda)                                                     'tf.math.reduce_sum[0][0]']     
                                                                                                  
 tf.math.unsorted_segment_mean   (1, 128)            0           ['input._get_features_ref_4[0][0]
 (TFOpLambda)                                                    ',                               
                                                                  'tf.__operators__.getitem[0][0]'
                                                                 ]                                
                                                                                                  
 hidden_layer (Dense)           (1, 256)             33024       ['tf.math.unsorted_segment_mean[0
                                                                 ][0]']                           
                                                                                                  
 logits (Dense)                 (1, 2)               514         ['hidden_layer[0][0]']           
                                                                                                  
==================================================================================================
Total params: 435,074
Trainable params: 435,074
Non-trainable params: 0
________________________________________________________________________________________________
</pre>

The resulting model then produces logits for each class, given a `GraphTensor` as its input:

{% highlight python %}
classification_model(graph_tensor)
{% endhighlight %}
<pre class="output">
<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-0.05114917, -0.23557672]], dtype=float32)>
</pre>

- A `preprocessors` method, which could be used to pre-process the graphs before reaching the GNN but will remain unused here

{% highlight python %}
task.preprocessors()
{% endhighlight %}
<pre class="output">
()
</pre>

### <a id="classification_training">4.2 Training</a>

We are now ready to train the model. First, we create a `KerasTrainer` instance which implements the [orchestrator's `Trainer` protocol](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md#training) making use of Keras' `fit` method:<a id="strategy"></a>

{% highlight python %}
trainer = runner.KerasTrainer(strategy=tf.distribute.get_strategy(), model_dir='model')
{% endhighlight %}

Next, we define a simple function conforming to [the `GraphTensorProcessorFn` protocol](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/runner.md#graphtensor-processing), which extracts the labels from the `GraphTensor` objects, for use during supervised training (this function will be mapped over the datasets that are then passed on to the `tf.keras.Model.fit` method):

{% highlight python %}
def extract_labels(graph_tensor):
    """
    Extract the toxicity class label from the `GraphTensor` representation of a molecule.
    Return a pair compatible with the `tf.keras.Model.fit` method.
    """
    return graph_tensor, graph_tensor.context['toxicity']
{% endhighlight %}

Lastly, we can put everything together and get a coffee while watching some progress bars move :-)<a id="orchestrator"></a>

{% highlight python %}
runner.run(
    train_ds_provider=train_dataset_provider,
    valid_ds_provider=valid_dataset_provider,
    feature_processors=[extract_labels],
    model_fn=get_model_creation_fn(hidden_size=128, hops=8),
    task=task,
    trainer=trainer,
    epochs=50,
    optimizer_fn=tf.keras.optimizers.Adam,
    gtspec=graph_spec,
    global_batch_size=128
)
{% endhighlight %}
<pre class="output">
Epoch 1/50
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_7/node_set_update_7/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_7/node_set_update_7/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_7/node_set_update_7/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_7/node_set_update_7/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_7/node_set_update_7/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_7/node_set_update_7/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_6/node_set_update_6/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_6/node_set_update_6/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_6/node_set_update_6/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_6/node_set_update_6/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_6/node_set_update_6/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_6/node_set_update_6/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_5/node_set_update_5/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_5/node_set_update_5/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_5/node_set_update_5/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_5/node_set_update_5/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_5/node_set_update_5/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_5/node_set_update_5/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_4/node_set_update_4/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_4/node_set_update_4/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_4/node_set_update_4/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_4/node_set_update_4/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_4/node_set_update_4/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_4/node_set_update_4/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_3/node_set_update_3/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_3/node_set_update_3/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_3/node_set_update_3/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_3/node_set_update_3/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_3/node_set_update_3/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_3/node_set_update_3/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_2/node_set_update_2/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_2/node_set_update_2/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_2/node_set_update_2/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_2/node_set_update_2/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_2/node_set_update_2/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_2/node_set_update_2/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_1/node_set_update_1/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_1/node_set_update_1/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_1/node_set_update_1/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_1/node_set_update_1/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_1/node_set_update_1/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_1/node_set_update_1/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_0/node_set_update/gat_v2_conv/Reshape_3:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_0/node_set_update/gat_v2_conv/Reshape_2:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_0/node_set_update/gat_v2_conv/Cast:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/indexed_slices.py:446: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_0/node_set_update/gat_v2_conv/Reshape_6:0", shape=(None,), dtype=int32), values=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_0/node_set_update/gat_v2_conv/Reshape_5:0", shape=(None, 1, 1), dtype=float32), dense_shape=Tensor("gradient_tape/model_7/gat_mpnn/message_passing_0/node_set_update/gat_v2_conv/Cast_1:0", shape=(3,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "shape. This may consume a large amount of memory." % value)
50/50 [==============================] - 10s 89ms/step - loss: 0.5478 - sparse_categorical_accuracy: 0.7352 - sparse_categorical_crossentropy: 0.5478 - AUROC: 0.6414 - val_loss: 0.5037 - val_sparse_categorical_accuracy: 0.7311 - val_sparse_categorical_crossentropy: 0.5037 - val_AUROC: 0.7434
Epoch 2/50
50/50 [==============================] - 3s 50ms/step - loss: 0.4676 - sparse_categorical_accuracy: 0.7645 - sparse_categorical_crossentropy: 0.4676 - AUROC: 0.7896 - val_loss: 0.4758 - val_sparse_categorical_accuracy: 0.7520 - val_sparse_categorical_crossentropy: 0.4758 - val_AUROC: 0.7870
Epoch 3/50
50/50 [==============================] - 2s 50ms/step - loss: 0.4404 - sparse_categorical_accuracy: 0.7889 - sparse_categorical_crossentropy: 0.4404 - AUROC: 0.8219 - val_loss: 0.4469 - val_sparse_categorical_accuracy: 0.7884 - val_sparse_categorical_crossentropy: 0.4469 - val_AUROC: 0.8204
Epoch 4/50
50/50 [==============================] - 2s 49ms/step - loss: 0.4192 - sparse_categorical_accuracy: 0.8055 - sparse_categorical_crossentropy: 0.4192 - AUROC: 0.8423 - val_loss: 0.4276 - val_sparse_categorical_accuracy: 0.8027 - val_sparse_categorical_crossentropy: 0.4276 - val_AUROC: 0.8406
Epoch 5/50
50/50 [==============================] - 3s 58ms/step - loss: 0.4015 - sparse_categorical_accuracy: 0.8191 - sparse_categorical_crossentropy: 0.4015 - AUROC: 0.8577 - val_loss: 0.4257 - val_sparse_categorical_accuracy: 0.8079 - val_sparse_categorical_crossentropy: 0.4257 - val_AUROC: 0.8441
Epoch 6/50
50/50 [==============================] - 3s 50ms/step - loss: 0.3943 - sparse_categorical_accuracy: 0.8202 - sparse_categorical_crossentropy: 0.3943 - AUROC: 0.8631 - val_loss: 0.4186 - val_sparse_categorical_accuracy: 0.8099 - val_sparse_categorical_crossentropy: 0.4186 - val_AUROC: 0.8497
Epoch 7/50
50/50 [==============================] - 2s 50ms/step - loss: 0.3878 - sparse_categorical_accuracy: 0.8286 - sparse_categorical_crossentropy: 0.3878 - AUROC: 0.8675 - val_loss: 0.4138 - val_sparse_categorical_accuracy: 0.8132 - val_sparse_categorical_crossentropy: 0.4138 - val_AUROC: 0.8529
Epoch 8/50
50/50 [==============================] - 3s 54ms/step - loss: 0.3860 - sparse_categorical_accuracy: 0.8322 - sparse_categorical_crossentropy: 0.3860 - AUROC: 0.8687 - val_loss: 0.4108 - val_sparse_categorical_accuracy: 0.8145 - val_sparse_categorical_crossentropy: 0.4108 - val_AUROC: 0.8522
Epoch 9/50
50/50 [==============================] - 2s 50ms/step - loss: 0.3823 - sparse_categorical_accuracy: 0.8327 - sparse_categorical_crossentropy: 0.3823 - AUROC: 0.8717 - val_loss: 0.4095 - val_sparse_categorical_accuracy: 0.8125 - val_sparse_categorical_crossentropy: 0.4095 - val_AUROC: 0.8551
Epoch 10/50
50/50 [==============================] - 3s 52ms/step - loss: 0.3759 - sparse_categorical_accuracy: 0.8350 - sparse_categorical_crossentropy: 0.3759 - AUROC: 0.8762 - val_loss: 0.4053 - val_sparse_categorical_accuracy: 0.8171 - val_sparse_categorical_crossentropy: 0.4053 - val_AUROC: 0.8586
Epoch 11/50
50/50 [==============================] - 2s 47ms/step - loss: 0.3726 - sparse_categorical_accuracy: 0.8363 - sparse_categorical_crossentropy: 0.3726 - AUROC: 0.8780 - val_loss: 0.4177 - val_sparse_categorical_accuracy: 0.8145 - val_sparse_categorical_crossentropy: 0.4177 - val_AUROC: 0.8496
Epoch 12/50
50/50 [==============================] - 2s 49ms/step - loss: 0.3692 - sparse_categorical_accuracy: 0.8389 - sparse_categorical_crossentropy: 0.3692 - AUROC: 0.8800 - val_loss: 0.3957 - val_sparse_categorical_accuracy: 0.8184 - val_sparse_categorical_crossentropy: 0.3957 - val_AUROC: 0.8635
Epoch 13/50
50/50 [==============================] - 2s 48ms/step - loss: 0.3590 - sparse_categorical_accuracy: 0.8458 - sparse_categorical_crossentropy: 0.3590 - AUROC: 0.8871 - val_loss: 0.3963 - val_sparse_categorical_accuracy: 0.8216 - val_sparse_categorical_crossentropy: 0.3963 - val_AUROC: 0.8652
Epoch 14/50
50/50 [==============================] - 3s 54ms/step - loss: 0.3579 - sparse_categorical_accuracy: 0.8466 - sparse_categorical_crossentropy: 0.3579 - AUROC: 0.8877 - val_loss: 0.3913 - val_sparse_categorical_accuracy: 0.8294 - val_sparse_categorical_crossentropy: 0.3913 - val_AUROC: 0.8689
Epoch 15/50
50/50 [==============================] - 3s 54ms/step - loss: 0.3519 - sparse_categorical_accuracy: 0.8519 - sparse_categorical_crossentropy: 0.3519 - AUROC: 0.8916 - val_loss: 0.3798 - val_sparse_categorical_accuracy: 0.8359 - val_sparse_categorical_crossentropy: 0.3798 - val_AUROC: 0.8759
Epoch 16/50
50/50 [==============================] - 2s 48ms/step - loss: 0.3451 - sparse_categorical_accuracy: 0.8536 - sparse_categorical_crossentropy: 0.3451 - AUROC: 0.8961 - val_loss: 0.3811 - val_sparse_categorical_accuracy: 0.8372 - val_sparse_categorical_crossentropy: 0.3811 - val_AUROC: 0.8737
Epoch 17/50
50/50 [==============================] - 2s 49ms/step - loss: 0.3378 - sparse_categorical_accuracy: 0.8580 - sparse_categorical_crossentropy: 0.3378 - AUROC: 0.9010 - val_loss: 0.3823 - val_sparse_categorical_accuracy: 0.8294 - val_sparse_categorical_crossentropy: 0.3823 - val_AUROC: 0.8750
Epoch 18/50
50/50 [==============================] - 2s 48ms/step - loss: 0.3436 - sparse_categorical_accuracy: 0.8573 - sparse_categorical_crossentropy: 0.3436 - AUROC: 0.8964 - val_loss: 0.3918 - val_sparse_categorical_accuracy: 0.8294 - val_sparse_categorical_crossentropy: 0.3918 - val_AUROC: 0.8753
Epoch 19/50
50/50 [==============================] - 2s 48ms/step - loss: 0.3439 - sparse_categorical_accuracy: 0.8562 - sparse_categorical_crossentropy: 0.3439 - AUROC: 0.8969 - val_loss: 0.3861 - val_sparse_categorical_accuracy: 0.8294 - val_sparse_categorical_crossentropy: 0.3861 - val_AUROC: 0.8746
Epoch 20/50
50/50 [==============================] - 2s 47ms/step - loss: 0.3337 - sparse_categorical_accuracy: 0.8612 - sparse_categorical_crossentropy: 0.3337 - AUROC: 0.9032 - val_loss: 0.3891 - val_sparse_categorical_accuracy: 0.8210 - val_sparse_categorical_crossentropy: 0.3891 - val_AUROC: 0.8750
Epoch 21/50
50/50 [==============================] - 2s 49ms/step - loss: 0.3286 - sparse_categorical_accuracy: 0.8661 - sparse_categorical_crossentropy: 0.3286 - AUROC: 0.9056 - val_loss: 0.3782 - val_sparse_categorical_accuracy: 0.8464 - val_sparse_categorical_crossentropy: 0.3782 - val_AUROC: 0.8756
Epoch 22/50
50/50 [==============================] - 3s 51ms/step - loss: 0.3286 - sparse_categorical_accuracy: 0.8658 - sparse_categorical_crossentropy: 0.3286 - AUROC: 0.9066 - val_loss: 0.3853 - val_sparse_categorical_accuracy: 0.8294 - val_sparse_categorical_crossentropy: 0.3853 - val_AUROC: 0.8775
Epoch 23/50
50/50 [==============================] - 2s 47ms/step - loss: 0.3229 - sparse_categorical_accuracy: 0.8692 - sparse_categorical_crossentropy: 0.3229 - AUROC: 0.9090 - val_loss: 0.3849 - val_sparse_categorical_accuracy: 0.8346 - val_sparse_categorical_crossentropy: 0.3849 - val_AUROC: 0.8746
Epoch 24/50
50/50 [==============================] - 2s 48ms/step - loss: 0.3130 - sparse_categorical_accuracy: 0.8747 - sparse_categorical_crossentropy: 0.3130 - AUROC: 0.9151 - val_loss: 0.3758 - val_sparse_categorical_accuracy: 0.8398 - val_sparse_categorical_crossentropy: 0.3758 - val_AUROC: 0.8801
Epoch 25/50
50/50 [==============================] - 2s 50ms/step - loss: 0.3132 - sparse_categorical_accuracy: 0.8756 - sparse_categorical_crossentropy: 0.3132 - AUROC: 0.9146 - val_loss: 0.3712 - val_sparse_categorical_accuracy: 0.8431 - val_sparse_categorical_crossentropy: 0.3712 - val_AUROC: 0.8833
Epoch 26/50
50/50 [==============================] - 3s 66ms/step - loss: 0.3085 - sparse_categorical_accuracy: 0.8766 - sparse_categorical_crossentropy: 0.3085 - AUROC: 0.9181 - val_loss: 0.3681 - val_sparse_categorical_accuracy: 0.8470 - val_sparse_categorical_crossentropy: 0.3681 - val_AUROC: 0.8848
Epoch 27/50
50/50 [==============================] - 3s 51ms/step - loss: 0.2998 - sparse_categorical_accuracy: 0.8797 - sparse_categorical_crossentropy: 0.2998 - AUROC: 0.9225 - val_loss: 0.3674 - val_sparse_categorical_accuracy: 0.8496 - val_sparse_categorical_crossentropy: 0.3674 - val_AUROC: 0.8859
Epoch 28/50
50/50 [==============================] - 2s 49ms/step - loss: 0.3059 - sparse_categorical_accuracy: 0.8789 - sparse_categorical_crossentropy: 0.3059 - AUROC: 0.9190 - val_loss: 0.3640 - val_sparse_categorical_accuracy: 0.8548 - val_sparse_categorical_crossentropy: 0.3640 - val_AUROC: 0.8892
Epoch 29/50
50/50 [==============================] - 3s 52ms/step - loss: 0.2951 - sparse_categorical_accuracy: 0.8828 - sparse_categorical_crossentropy: 0.2951 - AUROC: 0.9240 - val_loss: 0.3736 - val_sparse_categorical_accuracy: 0.8483 - val_sparse_categorical_crossentropy: 0.3736 - val_AUROC: 0.8828
Epoch 30/50
50/50 [==============================] - 2s 49ms/step - loss: 0.2938 - sparse_categorical_accuracy: 0.8825 - sparse_categorical_crossentropy: 0.2938 - AUROC: 0.9254 - val_loss: 0.3759 - val_sparse_categorical_accuracy: 0.8470 - val_sparse_categorical_crossentropy: 0.3759 - val_AUROC: 0.8827
Epoch 31/50
50/50 [==============================] - 2s 49ms/step - loss: 0.2848 - sparse_categorical_accuracy: 0.8856 - sparse_categorical_crossentropy: 0.2848 - AUROC: 0.9301 - val_loss: 0.3668 - val_sparse_categorical_accuracy: 0.8451 - val_sparse_categorical_crossentropy: 0.3668 - val_AUROC: 0.8877
Epoch 32/50
50/50 [==============================] - 3s 58ms/step - loss: 0.2885 - sparse_categorical_accuracy: 0.8861 - sparse_categorical_crossentropy: 0.2885 - AUROC: 0.9275 - val_loss: 0.3867 - val_sparse_categorical_accuracy: 0.8294 - val_sparse_categorical_crossentropy: 0.3867 - val_AUROC: 0.8832
Epoch 33/50
50/50 [==============================] - 2s 48ms/step - loss: 0.2904 - sparse_categorical_accuracy: 0.8831 - sparse_categorical_crossentropy: 0.2904 - AUROC: 0.9269 - val_loss: 0.3913 - val_sparse_categorical_accuracy: 0.8372 - val_sparse_categorical_crossentropy: 0.3913 - val_AUROC: 0.8765
Epoch 34/50
50/50 [==============================] - 2s 47ms/step - loss: 0.2891 - sparse_categorical_accuracy: 0.8834 - sparse_categorical_crossentropy: 0.2891 - AUROC: 0.9276 - val_loss: 0.3857 - val_sparse_categorical_accuracy: 0.8444 - val_sparse_categorical_crossentropy: 0.3857 - val_AUROC: 0.8809
Epoch 35/50
50/50 [==============================] - 2s 50ms/step - loss: 0.2780 - sparse_categorical_accuracy: 0.8894 - sparse_categorical_crossentropy: 0.2780 - AUROC: 0.9333 - val_loss: 0.3794 - val_sparse_categorical_accuracy: 0.8470 - val_sparse_categorical_crossentropy: 0.3794 - val_AUROC: 0.8894
Epoch 36/50
50/50 [==============================] - 3s 57ms/step - loss: 0.2670 - sparse_categorical_accuracy: 0.8945 - sparse_categorical_crossentropy: 0.2670 - AUROC: 0.9382 - val_loss: 0.4007 - val_sparse_categorical_accuracy: 0.8197 - val_sparse_categorical_crossentropy: 0.4007 - val_AUROC: 0.8775
Epoch 37/50
50/50 [==============================] - 3s 52ms/step - loss: 0.2598 - sparse_categorical_accuracy: 0.8964 - sparse_categorical_crossentropy: 0.2598 - AUROC: 0.9422 - val_loss: 0.3599 - val_sparse_categorical_accuracy: 0.8542 - val_sparse_categorical_crossentropy: 0.3599 - val_AUROC: 0.8921
Epoch 38/50
50/50 [==============================] - 3s 55ms/step - loss: 0.2585 - sparse_categorical_accuracy: 0.8959 - sparse_categorical_crossentropy: 0.2585 - AUROC: 0.9428 - val_loss: 0.3738 - val_sparse_categorical_accuracy: 0.8529 - val_sparse_categorical_crossentropy: 0.3738 - val_AUROC: 0.8895
Epoch 39/50
50/50 [==============================] - 2s 47ms/step - loss: 0.2507 - sparse_categorical_accuracy: 0.9002 - sparse_categorical_crossentropy: 0.2507 - AUROC: 0.9458 - val_loss: 0.3730 - val_sparse_categorical_accuracy: 0.8561 - val_sparse_categorical_crossentropy: 0.3730 - val_AUROC: 0.8937
Epoch 40/50
50/50 [==============================] - 2s 49ms/step - loss: 0.2546 - sparse_categorical_accuracy: 0.8975 - sparse_categorical_crossentropy: 0.2546 - AUROC: 0.9446 - val_loss: 0.3717 - val_sparse_categorical_accuracy: 0.8535 - val_sparse_categorical_crossentropy: 0.3717 - val_AUROC: 0.8954
Epoch 41/50
50/50 [==============================] - 2s 50ms/step - loss: 0.2463 - sparse_categorical_accuracy: 0.9020 - sparse_categorical_crossentropy: 0.2463 - AUROC: 0.9484 - val_loss: 0.3772 - val_sparse_categorical_accuracy: 0.8470 - val_sparse_categorical_crossentropy: 0.3772 - val_AUROC: 0.8899
Epoch 42/50
50/50 [==============================] - 2s 47ms/step - loss: 0.2490 - sparse_categorical_accuracy: 0.9028 - sparse_categorical_crossentropy: 0.2490 - AUROC: 0.9459 - val_loss: 0.3963 - val_sparse_categorical_accuracy: 0.8477 - val_sparse_categorical_crossentropy: 0.3963 - val_AUROC: 0.8955
Epoch 43/50
50/50 [==============================] - 2s 48ms/step - loss: 0.2403 - sparse_categorical_accuracy: 0.9038 - sparse_categorical_crossentropy: 0.2403 - AUROC: 0.9498 - val_loss: 0.3779 - val_sparse_categorical_accuracy: 0.8522 - val_sparse_categorical_crossentropy: 0.3779 - val_AUROC: 0.9005
Epoch 44/50
50/50 [==============================] - 2s 48ms/step - loss: 0.2527 - sparse_categorical_accuracy: 0.8994 - sparse_categorical_crossentropy: 0.2527 - AUROC: 0.9447 - val_loss: 0.3837 - val_sparse_categorical_accuracy: 0.8405 - val_sparse_categorical_crossentropy: 0.3837 - val_AUROC: 0.8924
Epoch 45/50
50/50 [==============================] - 3s 53ms/step - loss: 0.2330 - sparse_categorical_accuracy: 0.9084 - sparse_categorical_crossentropy: 0.2330 - AUROC: 0.9539 - val_loss: 0.3765 - val_sparse_categorical_accuracy: 0.8652 - val_sparse_categorical_crossentropy: 0.3765 - val_AUROC: 0.8977
Epoch 46/50
50/50 [==============================] - 3s 52ms/step - loss: 0.2286 - sparse_categorical_accuracy: 0.9127 - sparse_categorical_crossentropy: 0.2286 - AUROC: 0.9541 - val_loss: 0.3828 - val_sparse_categorical_accuracy: 0.8535 - val_sparse_categorical_crossentropy: 0.3828 - val_AUROC: 0.8976
Epoch 47/50
50/50 [==============================] - 2s 48ms/step - loss: 0.2172 - sparse_categorical_accuracy: 0.9156 - sparse_categorical_crossentropy: 0.2172 - AUROC: 0.9594 - val_loss: 0.4157 - val_sparse_categorical_accuracy: 0.8457 - val_sparse_categorical_crossentropy: 0.4157 - val_AUROC: 0.8968
Epoch 48/50
50/50 [==============================] - 2s 48ms/step - loss: 0.2285 - sparse_categorical_accuracy: 0.9142 - sparse_categorical_crossentropy: 0.2285 - AUROC: 0.9549 - val_loss: 0.4125 - val_sparse_categorical_accuracy: 0.8509 - val_sparse_categorical_crossentropy: 0.4125 - val_AUROC: 0.8921
Epoch 49/50
50/50 [==============================] - 3s 52ms/step - loss: 0.2226 - sparse_categorical_accuracy: 0.9144 - sparse_categorical_crossentropy: 0.2226 - AUROC: 0.9574 - val_loss: 0.3726 - val_sparse_categorical_accuracy: 0.8620 - val_sparse_categorical_crossentropy: 0.3726 - val_AUROC: 0.9007
Epoch 50/50
50/50 [==============================] - 2s 48ms/step - loss: 0.2209 - sparse_categorical_accuracy: 0.9155 - sparse_categorical_crossentropy: 0.2209 - AUROC: 0.9583 - val_loss: 0.3729 - val_sparse_categorical_accuracy: 0.8600 - val_sparse_categorical_crossentropy: 0.3729 - val_AUROC: 0.9016
2022-06-23 19:33:36.568640: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
/opt/conda/lib/python3.7/site-packages/tensorflow/python/saved_model/nested_structure_coder.py:524: UserWarning: Encoding a StructuredValue with type tensorflow_gnn.GraphTensorSpec; loading this StructuredValue will require that this type be imported and registered.
  "imported and registered." % type_spec_class_name)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/saved_model/nested_structure_coder.py:524: UserWarning: Encoding a StructuredValue with type tensorflow_gnn.ContextSpec.v2; loading this StructuredValue will require that this type be imported and registered.
  "imported and registered." % type_spec_class_name)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/saved_model/nested_structure_coder.py:524: UserWarning: Encoding a StructuredValue with type tensorflow_gnn.NodeSetSpec; loading this StructuredValue will require that this type be imported and registered.
  "imported and registered." % type_spec_class_name)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/saved_model/nested_structure_coder.py:524: UserWarning: Encoding a StructuredValue with type tensorflow_gnn.EdgeSetSpec; loading this StructuredValue will require that this type be imported and registered.
  "imported and registered." % type_spec_class_name)
/opt/conda/lib/python3.7/site-packages/tensorflow/python/saved_model/nested_structure_coder.py:524: UserWarning: Encoding a StructuredValue with type tensorflow_gnn.AdjacencySpec; loading this StructuredValue will require that this type be imported and registered.
  "imported and registered." % type_spec_class_name)
</pre>

### <a id="classification_metrics">4.3 Metric visualization</a>

A straightforward way to visualize the various metrics collected during training and validation is to use TensorBoard. Ideally, the following magic should work in your system:

{% highlight python %}
%load_ext tensorboard
%tensorboard --logdir model --bind_all
{% endhighlight %}

For reference, I have uploaded the logs of a local run to [TensorBoard.dev](https://tensorboard.dev/experiment/Y85x93T8Rl6p5QqyajLFeQ/#scalars). There you should see that the loss decreases until we start to overfit (orange line is training, blue line is validation):

{% include figure.html path="assets/img/blog/tfgnn-intro/epoch_loss.png" class="img-fluid rounded z-depth-1" %}

The accuracy reaches approximately 87%

{% include figure.html path="assets/img/blog/tfgnn-intro/accuracy.png" class="img-fluid rounded z-depth-1" %}

And the AUROC increases up to approximately 0.9

{% include figure.html path="assets/img/blog/tfgnn-intro/auroc.png" class="img-fluid rounded z-depth-1" %}

<div class="section separator"></div>

## <a id="conclusions">Conclusions</a>

In this notebook we have seen how to train a GNN model for graph binary classification using TF-GNN in an end-to-end fashion. The final [cell running the orchestrator](#orchestrator) brings together all the elements we introduced along the way, namely:

- The `DatasetProvider`-compliant objects `train_dataset_provider` and `valid_dataset_provider` constructed in [§2](#dataprep) to provide the data
- The model constructor function `get_model_creation_fn` built in [§3.3](#mpnn_construction) with the components of [§3.1](#mpnn_embedding) and [§3.2](#mpnn_stack) to assemble the GNN
- The `GraphBinaryClassification` task defined in [§4.1](#classification_task) to specify the readout and prediction head, as well as the loss and metrics.
- The `KerasTrainer` and target feature extractor created in [§4.2](#classification_training) for supervised training

While it may not be immediately obvious from such a small example, TF-GNN helped at each step by providing not just the underlying operations we need to perform on graphs, but also many useful protocols and helper functions taking care of much of the boilerplate code we would have otherwise required. Using these in tandem with the orchestrator means that all of the components are easily extendable and/or replaceable. Moreover, it allows us at least in principle to easily scale the various moving parts independently and without unnecessary pain. For example, introducing a non-trivial strategy in the [trainer](#strategy) we could distribute our training across multiple GPUs or, eventually, TPUs, while also parallelizing our input pipeline through the `InputContext` that is [passed on to the `DatasetProvider`](#input_context).

---

The results we obtained for the drug cardiotoxicity dataset are good, but not impressive. This was to be expected given the very simple GAT-based model that we implemented and the fact that we did no hyperparameter optimization or principled architectural choices. For comparison, we partially quote here [Table 1 from [1]](#cardiotox), where we see that our AUROC results are essentially consistent with the GNN baseline considered there:

{% include figure.html path="assets/img/blog/tfgnn-intro/table.png" class="img-fluid rounded z-depth-1" %}

The reader is encouraged to try various modifications of the GNN architecture to improve the performance of the model, as well as study its behavior on the out-of-distribution test sets `test1_dataset_provider` and `test2_dataset_provider` that we prepared in [§2](#dataprep) but did not use here.

<div class="section separator"></div>

### <a id="conclusions_resources">5.1 Resources and acknowledgements</a>

There are a number of resources that have inspired or been used in this notebook, some of which are cited along the way. They are also collected here for reference purposes:

- The paper [[1]](#cardiotox) introducing the CardioTox dataset digs deeper in the data we have used, and builds better classification models for it

- The papers [[2]](#gat) and [[3]](#gat) introduced and popularized Graph Attention Networks

- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) is a good reference to start learning more about GNNs and their applications

- For those looking to dig deeper into GNNs, the [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/) by W. L. Hamilton is fairly up-to-date and more comprehensive

- The [TensorFlow GNN guide](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/intro.md) is a good place to start learning how to use TF-GNN, and some of this notebook's code was inspired by the examples provided there

- **[July 5, 2022 UPDATE]** Some official notebooks with TF-GNN examples are now available, see _e.g._ [Molecular Graph Classification](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/intro_mutag_example.ipynb), [Solving OGBN-MAG end-to-end](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/ogbn_mag_e2e.ipynb#scrollTo=udvGTpefWRE_) and [Learning shortest paths with GraphNetworks](https://colab.research.google.com/github/tensorflow/gnn/blob/master/examples/notebooks/graph_network_shortest_path.ipynb)