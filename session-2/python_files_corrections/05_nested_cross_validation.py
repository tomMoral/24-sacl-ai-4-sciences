# %% [markdown]
#
# # Hyperparameter tuning with nested cross-validation
#
# In this notebook, we will illustrate how to perform hyperparameter tuning
# using nested cross-validation.
#
# We start be defining the predictive model that we created in the previous
# notebook.

# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins.csv")
df

# %%
feature_names = [
    "Region",
    "Island",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
]
target_name = "Species"
X = df[feature_names]
y = df[target_name]

categorical_columns = X.select_dtypes(include="object").columns
X[categorical_columns] = X[categorical_columns].astype("category")

# %%
from skrub import tabular_learner
from sklearn.linear_model import LogisticRegression

logistic_regression = tabular_learner(LogisticRegression())
logistic_regression

# %% [markdown]
#
# In the past, we saw how to evaluate this model with cross-validation.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(logistic_regression, X, y, cv=5, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %% [markdown]
#
# However, this machine learning pipeline has some hyperparameters that are set to
# default values and not necessarily optimal for our problem.

# %%
logistic_regression.get_params()

# %% [markdown]
#
# We can tune these hyperparameters using a search strategy.

# %%
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "simpleimputer__strategy": ["mean", "median", "most_frequent"],
    "logisticregression__C": loguniform(1e-3, 1e3),
}
tuned_model = RandomizedSearchCV(
    logistic_regression,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    random_state=0,
)
tuned_model

# %%
# set `return_estimator=True` to access the best model found during the search
cv_results = cross_validate(tuned_model, X, y, cv=5, return_estimator=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
#
# We can check the best estimator that have been for each fold.

# %%
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)

# %% [markdown]
#
# However, the previous approach is not the best way to tune hyperparameters.
# We used the default score that is the accuracy here. However, if we have a
# probabilistic model, we should instead optimized a proper scoring function instead
# of a threshold-based metric. So we can optimize the log loss instead of the accuracy.

# %%
cv_results = cross_validate(
    tuned_model, X, y, cv=5, scoring="neg_log_loss", return_estimator=True
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)

# %%
