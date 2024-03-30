<h3 align="left"> EiVPR: Efficient integration of global retrieval and reranking for visual place recognition via Transformer </h3>
The data and codes that support the findings of this study are available here.

## Catalogs

- Motivation of this work
- Environments
- How to run this project

### Motivation: This figure shows our motivation for this study.
![motivation.png](images%2Fmotivation.png)

### Environments
1. pytorch==2.1.0
2. pytorch-cuda==12.1
2. pytorch-lightning==2.1.2
3. faiss-gpu==1.7.2

### How to run this project
All code and preprocess data are including this project.

Everything will be saved easily by Tensorboard.

You just need to download train and benchmark datasets, put them to correct path. 

Then run main.py.

```sh
python main.py
```