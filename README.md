# Structure-Conditional Minimum Bayes Risk Decoding

## Structural Variation Datasets

We provide several structural variation datasets derived from DailyDialog and Alpaca:

- [dataset/dialogue-act.json](dataset/dialogue-act.json)
- [dataset/emotion.json](dataset/emotion.json)
- [dataset/response-structure.json](dataset/response-structure.json)

Each dataset contains conversational or instructional contexts, multiple generated candidate responses, and structural labels (e.g. dialogue act, emotion, or response format). See [DATA_LICENSE](DATA_LICENSE) for details on dataset licensing and attribution.

## Structure-Conditional MBR 

To install
```
git clone git@github.com:Roxot/structure-conditional-mbr.git
cd structure-conditional-mbr
pip install .
```

First precompute utility values to use for training and testing SC-MBR methods.
```
scmbr-utility -s train -u bleurt -d dialogue-act
scmbr-utility -s validation -u bleurt -d dialogue-act
scmbr-utility -s test -u bleurt -d dialogue-act
```

Train the structure embeddings clustering model:
```
scmbr-train -d dialogue-act -m cluster -n structure-embeddings-model
```

Obtain thresholds:
```
scmbr-train -d dialogue-act -m cutoff-bleurt
scmbr-train -d dialogue-act -m structure-embeddings-bleurt --cluster_model structure-embeddings-model
```

Evaluate structural optimality:
```
scmbr-eval -d dialogue-act -s test -m cutoff-bleurt --threshold 0.5
scmbr-eval -d dialogue-act -s test -m cluster-bleurt --cluster_model structure-embeddings-model
scmbr-eval -d dialogue-act -s test -m structure-embeddings-bleurt --threshold 0.8 --cluster_model structure-embeddings-model
```

Generate outputs:
```
scmbr-generate -d dialogue-act -s test -m cutoff-bleurt --threshold 0.5 --out cutoff-generations.jsonl
scmbr-generate -d dialogue-act -s test -m cluster-bleurt --cluster_model structure-embeddings-model --out cluster-generations.jsonl
scmbr-generate -d dialogue-act -s test -m structure-embeddings-bleurt --threshold 0.8 --cluster_model structure-embeddings-model --out structure-embeddings-generations.jsonl
```

## Cite
If you use these datasets or the code, please cite:
```
@inproceedings{eikema-etal-2025-structureconditional,
  title     = {Structure-Conditional Minimum Bayes Risk Decoding},
  author    = {Bryan Eikema and Anna Rutkiewicz and Mario Giulianelli},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  month     = nov,
  year      = {2025},
  address   = {Suzhou},
  publisher = {Association for Computational Linguistics}
}
```

## Acknowledgements
<img src="eu_flag.png" alt="EU flag" width="60" align="left" />

This work was partly funded by the European Union's Horizon Europe (HE) Research and Innovation programme under Grant Agreement No 101070631 and from the UK Research and Innovation (UKRI) under the UK government's HE funding grant No 10039436.

## License
- Code: [MIT License](LICENSE)  
- Datasets: [DATA_LICENSE](DATA_LICENSE)  
