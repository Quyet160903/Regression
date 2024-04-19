import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")

target = "math score"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

nums_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

education_levels = ["master's degree", "bachelor's degree", "associate's degree", "some college",
                    "high school", "some high school"]
lunch_values = x_train["lunch"].unique()
gender_values = x_train["gender"].unique()
test_prep_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_levels, gender_values, lunch_values, test_prep_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

# result = ord_transformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
# for i, j in zip(x_train[["parental level of education", "gender", "lunch", "test preparation course"]].values, result):
#     print("Before: {}. After: {}".format(i, j))

preprocessor = ColumnTransformer(transformers=[
    ("num_features", nums_transformer, ["reading score", "writing score"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"])
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])
# x_train = reg.fit_transform(x_train)
# x_test = reg.transform(x_test)

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__criterion": ["squared_error", "absolute_error", "poisson"],
    "model__max_depth": [None, 2, 5, 10],
    "preprocessor__num_features__imputer__strategy": ["mean", "median"]
}
grid_search = RandomizedSearchCV(reg, param_distributions=param_grid, n_iter=30, cv=5, verbose=1, n_jobs=6, scoring="r2")
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
y_predict = grid_search.predict(x_test)

# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print("Predicted: {}. Actual: {}".format(i, j))

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("Mean absolute error: {}".format(mae))
print("Mean square error: {}".format(mse))
print("R2: {}".format(r2))

# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(models)