# %% [markdown]
#
# # Working with heterogeneous data
#
# In this notebook, we will present how to handle heterogeneous data. Usually, examples
# only show how to deal with numerical data but in practice, we often have to deal with
# a mix of numerical and categorical data.
#
# So let's look at the entire penguins dataset this time.

# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins.csv")
df

# %% [markdown]
#
# We see that we have some strings and numbers in this dataset. Let's set up a
# classification problem: we want to predict the species of the penguins given some
# numerical and categorical features.

# %%
feature_names = [
    "Region",
    "Island",
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
]
target_name = "Species"
X = df[feature_names]
y = df[target_name]

# %% [markdown]
#
# Before to evaluate model through cross-validation, we will first look at model
# using a single split.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
#
# ### Exercise
#
# Fit a `LogisticRegression` model on the training data.

# %%
from sklearn.linear_model import LogisticRegression

# %% [markdown]
#
# So life is difficult. So let's start by looking at the numerical part.

# %%
X_numeric = X.select_dtypes(include="number")

# %% [markdown]
#
# ### Exercise
#
# Fit a `LogisticRegression` model on the numerical data.

# %%

# %% [markdown]
#
# Does it work?

# %%
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

logistic_regression = make_pipeline(SimpleImputer(), LogisticRegression())
logistic_regression.fit(X_numeric, y)

# %% [markdown]
#
# It works but we get a convergence warning. Let's check how many iterations were
# performed.

# %%
logistic_regression[-1].n_iter_

# %% [markdown]
#
# Indeed, we reached the maximum number of iterations. We can increase the number of
# iterations. Let's check which parameters we can set with the `get_params` method.

# %%
logistic_regression.get_params()

# %% [markdown]
#
# We can set the `max_iter` parameter to a higher value through the variable
# `logisticregression__max_iter`.

# %%
logistic_regression.set_params(logisticregression__max_iter=10_000)
logistic_regression.fit(X_numeric, y)

# %%
logistic_regression[-1].n_iter_

# %% [markdown]
#
# Now, the model converged but it required almost 2,500 iterations. The warning message
# mentioned that we could try to scale the data. Let's try to scale the data using a
# `StandardScaler`. We can then check if the convergence is faster.

# %%
from sklearn.preprocessing import StandardScaler

logistic_regression = make_pipeline(
    StandardScaler(), SimpleImputer(), LogisticRegression()
)
logistic_regression.fit(X_numeric, y)
logistic_regression[-1].n_iter_

# %% [markdown]
#
# It only requires 11 iterations. We can now evaluate the model using cross-validation.

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    logistic_regression, X_numeric, y, cv=10, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %% [markdown]
#
# Now, let's only consider the categorical data.

# %%
X_categorical = X.select_dtypes(exclude="number")
X_categorical

# %% [markdown]
#
# ### Exercise
#
# Think of a way to transform the string category into numerical data.
# Come with your own transform and evaluate the model using cross-validation.

# %% [markdown]
#
# We need to find a strategy to "encode" the categorical data into numerical data. The
# simplest strategy is to use an ordinal encoding.

# %%
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder().set_output(transform="pandas")
X_encoded = ordinal_encoder.fit_transform(X_categorical)
X_encoded

# %% [markdown]
#
# It replace a category by an integer. However, with linear models, it means that we
# would assume that the difference between two categories is the same. Also, there is
# an ordering imposed by this transformation.
#
# If this modelling assumption is not desired, we can use a one-hot encoding.

# %%
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
X_encoded = one_hot_encoder.fit_transform(X_categorical)
X_encoded

# %% [markdown]
#
# In this case, we create independent binary columns for each category. We therefore
# have an individual coefficient for each category. Usually, this is a more appropriate
# encoding for linear models.
#
# Let's use this encoding and evaluate the model using cross-validation.

# %%
logistic_regression = make_pipeline(OneHotEncoder(), LogisticRegression())
cv_results = cross_validate(
    logistic_regression, X_categorical, y, cv=10, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %% [markdown]
#
# We get an error for one of the split. This is due to the fact that some categories are
# not present in the test set. We can handle this issue by ignoring the unknown
# categories. This is given by a parameter in the `OneHotEncoder`.

# %%
logistic_regression.get_params()

# %%
logistic_regression.set_params(onehotencoder__handle_unknown="ignore")

# %%
cv_results = cross_validate(
    logistic_regression, X_categorical, y, cv=10, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %% [markdown]
#
# Now, we need to combine both numerical and categorical preprocessing and feed the
# output to a single linear model. The `ColumnTransformer` class is designed for this
# purpose: we provide a list of columns such that it will be transformed by a specific
# transformer (or a pipeline of transformers). This `ColumnTransformer` can be used as
# a preprocessing stage of a pipeline containing a linear model as the final stage.

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns = selector(dtype_include="number")
numerical_columns(X)

# %%
from sklearn.compose import make_column_transformer

numerical_selector = selector(dtype_include="number")
categorical_selector = selector(dtype_exclude="number")
preprocessor = make_column_transformer(
    (make_pipeline(StandardScaler(), SimpleImputer()), numerical_selector),
    (OneHotEncoder(handle_unknown="ignore"), categorical_selector),
)
logistic_regression = make_pipeline(preprocessor, LogisticRegression())
logistic_regression

# %%
cv_results = cross_validate(logistic_regression, X, y, cv=10, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
#
# We gave basic preprocessing steps for linear model. However, there is another group
# of models that can handle heterogeneous data: tree-based models.
#
# ### Exercise
#
# Looking at the documentation, create and evaluate a `HistGradientBoostingClassifier`
# model on the penguins dataset. You are free to create any preprocessing steps you
# want.
