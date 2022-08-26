# REST API to inference model trained on Census Income Data Set

## Getting started
```
conda env create -f environment.yaml
conda activate heroku-app
uvicorn api.main:app --reload
```
Usaually the server will listen to http://127.0.0.1:8000/
## Run Test
```
conda activate heroku-app
pytest
```

## Train model
```
conda activate heroku-app
python model_training/train_model.py
```

## API URL
https://inferenceapi0.herokuapp.com/docs

## Required Files
- [continuous_deloyment.png](screenshots/continuous_deloyment.png)
- [example.png](screenshots/example.png)
- [live_get.png](screenshots/live_get.png)
- [live_post.png](screenshots/live_post.png)
- [slice_output.txt](models/slice_output.txt)
- [continuous_integration.png](screenshots/continuous_integration.png)