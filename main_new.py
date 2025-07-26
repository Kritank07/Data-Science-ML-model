from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import joblib
import os

MODEL_FILE = 'housing_model.pkl'
PIPELINE_FILE = 'housing_pipelines.pkl'

def build_pipelines(num_attribs, cat_attribs):
    # Numerical attributes pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('standardization', StandardScaler()),
    ])

    # Categorical attributes pipeline
    cat_pipeline = Pipeline([
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine the numerical and categorical pipelines
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Train the model
    housing = pd.read_csv("housing.csv")

    # 2. stratified split
    housing["income_cat"] = pd.cut(housing["median_income"],
                                        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                        labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv('input.csv', index = False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    pipeline = build_pipelines(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)
    random_predictions = model.predict(housing_prepared)
    random_rmses = -cross_val_score(model, housing_prepared, housing_labels,
                            scoring="neg_root_mean_squared_error", cv=10)
    print(pd.Series(random_rmses).describe())
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

else:
    # Let's do the inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv('output.csv', index=False)
    print('inference is complete, results saved to output.csv. Enjoy!')