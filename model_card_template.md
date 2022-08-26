# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
RandomForest classifier using sklean [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The parameters of the model or mostly the default ones. The only changed parameters are:
- max_depth=10
## Intended Use
The model is a binary classifier which predicts if a person has an income below or higher then 50k per year. The classifier predicts 
the salary class based on the follwing features:
- age
- workclass
- education
- maritalStatus
- occupation
- relationship
- race
- sex
- hoursPerWeek
- nativeCountry
## Training Data
The model is trained on the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income).
## Evaluation Data
We used 20% of the dataset for model evaluation
## Metrics
The current model has the following performance metrices:
- precision: 0.7403141361256544
- recall: 0.47802569303583503
- fbeta: 0.5809367296631059

## Ethical Considerations
The dataset contains two critical features (race, gender) which could lead to potential discrimination of people.
## Caveats and Recommendations
The current model does not achieve a very high score. One could perform the following steps to optimize the model:
- Do hyperparameter optimization
- Chose another kind of classifier like for example a[ Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html)
