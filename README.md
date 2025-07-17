# TopicVI: Prior Gene Programs Motivated Cell Subtype or State Definition by Topic Modeling

TopicVI is a Python package that implements topic modeling to define cell subtypes or states based on prior gene programs. It uses the variational autoencoder (VAE) framework and non-negative matrix factorization (NMF) to discover topics in single-cell RNA-seq data, allowing for the identification of distinct cell populations and their functional characteristics.

The key idea of TopicVI is to leverage existing biologicaxl knowledge, in the form of prior gene programs, to guide the topic modeling process. By optimal transport algorithm, TopicVI is able to improve the interpretability and biological relevance of the discovered topics.

The built-in prior gene programs include cell markers, cell states, cell functions, and more. These programs can be used to define cell subtypes or states, and to annotate the discovered topics with biological meaning.

| Content                                    | Source                                                                     | Category       |
| ------------------------------------------ | -------------------------------------------------------------------------- | -------------- |
| Human / Mouse cell markers                 | CellMarkerDB v2.0                                    | Cell Markers   |
| Cell markers of different Human tissue     | ScType                                     | Cell Markers   |
| Meta-Program of Immune and Malignant cells | [Gavish el al.](doi.org/10.1038/s41586-023-06130-4)                                           | Cell States    |
| Tissue Specific Geneset                    | DAVID Database (2024)                                    | Cell States    |
| Gene Ontology Items                        | The Gene Ontology Resource (2017) | Cell Functions |
| KEGG Pathways                              | KEGG DB (2021, Human)                               | Cell Functions |
| Reactome Pathways                          | Reactome DB (2022)                                | Cell Functions |
| Hallmarks                                  | MSigDB HallMark (2020)                         | Cell Functions |

For more details, please reefer to our paper.

## Installation

by source code:

Notice to the pytorch version and your CUDA version. We highly recommend you should install the correct pytorch version first, then install TopicVI.

```bash
git clone https://github.com/gx-cai/topicvi.git
cd topicvi
pip install -e .
```

by pip: [Currently not available]

```bash
pip install topicvi
```


## Documentation

See [Wiki page](https://github.com/gx-cai/topicvi/wiki) for more details.

## Citation

If you use TopicVI in your research, please cite the following paper:

[Currently not available]
