# NK-GAD
Neighbor Knowledge-enhanced Unsupervised Graph Anomaly Detection. DASFAA2026

Environment
```
python==3.9.20
pytorch==2.1.0
pytorch-cuda==11.8
torch-cluster==1.6.3+pt21cu118 
torch-geometric==2.6.1
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-spline-conv==1.2.2
torchvision==0.16.0
pygod==1.1.0
tqdm==4.67.0
scikit-learn==1.5.2
scipy==1.13.1
networkx==3.2.1
numpy==1.25.0
matplotlib==3.9.2
dgl==2.4.0
```

Install requirements.txt
```
conda install --file requirements.txt
```

Use the following command to run the main script with configuration options:

```bash
python3 main.py --dataset $dataset_name --device $gpu_id
```
`$dataset_name: inj_cora, inj_amazon, weibo, reddit, disney, books, enron`

We support using datasets from [data](https://github.com/pygod-team/data). You can download these datasets and place them in the data directory `data`. 
Alternatively, you can run the code directly, and the dataset will be downloaded automatically.
