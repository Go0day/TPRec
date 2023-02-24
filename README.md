# TPRec
This repository contains the source code of the TOIS 2022 paper "[Time-aware Path Reasoning on Knowledge Graph for Recommendation](https://arxiv.org/pdf/2108.02634.pdf)" [1].

## Datasets
We obtain the data from PGPR[2,3]. 
Three Amazon datasets used in this paper can be download [here](https://drive.google.com/drive/folders/1xtmfHeZ_LAW-RkYdu_6_HEDXCFzgi0nw?usp=share_link).

## Requirements
- Python >= 3.6
- PyTorch = 1.0

## How to run the code
1. Preprocess the temporal information.
```bash
python GMM_process.py --dataset <dataset_name> --cluster_num <num> --cluster_feature <temporal_feature>
```
"<dataset_name>" should be one of "beauty", "cloth", "cell" (refer to utils.py).
"temporal_feature" should be one of "all", "w-stru", "w-stat".


2. Preprocess the data:
```bash
python preprocess.py --dataset <dataset_name>
```


3. Train Time-aware Collaborative Knowledge Graph embeddings:
```bash
python train_transe_model.py --dataset <dataset_name>
```
In order to reduce the training time, it is better to put the embedding without time information into the "tmp/<dataset_name>/init_embedding/" folder in advance, which can be be obtained from the third step of PGPR[2,3], or downloaded from [here](https://drive.google.com/drive/folders/1xtmfHeZ_LAW-RkYdu_6_HEDXCFzgi0nw?usp=share_link).

4. Train RL agent:
```bash
python train_agent.py --dataset <dataset_name>
```

5. Evaluation
```bash
python test_agent.py --dataset <dataset_name>
```


## References
[1] Yuyue Zhao, Xiang Wang, Jiawei Chen, Wei Tang, Yashen Wang, Xiangnan He, Haiyong Xie. Time-aware Path Reasoning on Knowledge Graph for Recommendation. arXiv preprint arXiv:2108.02634, 2021.

[2] Yikun Xian, Zuohui Fu, S. Muthukrishnan, Gerard de Melo, Yongfeng Zhang. "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation." In Proceedings of SIGIR. 2019.

[3] The backbone implementation is reference to https://github.com/orcax/PGPR .
