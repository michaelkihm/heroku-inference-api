# Inference model deployed with REST API

## Getting started
```
conda env create -f environment.yaml
conda activate heroku-app
uvicorn api.main:app --reload
```