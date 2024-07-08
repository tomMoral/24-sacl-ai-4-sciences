# %% [markdown]
#
# # Evaluate a predictive model
#
# In this first notebook, we show how to evaluate a predictive model.
#
# Let's consider a simple regression problem by loading a subset of the penguins
# dataset. Let's check what are the data available in this dataset.

# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins_regression.csv")
df

# %% [markdown]
#
# In this dataset, we observe that we have two variables: the flipper length and the
# body mass of the penguins. The objective here is to create a predictive model allowing
# us to predict the body mass of a penguin based on its flipper length.
#
# First, we can have a look at the relationship between the flipper length and the body
# mass of the penguins.

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
_ = ax.set_title("Penguin Flipper Length vs Body Mass")

# %% [markdown]
#
# Looking at this plot, we observe that there is a kind of linear relationship between
# the flipper length and the body mass of the penguins. We will start by fitting a
# linear regression model to this data.
#
# To do so, we will first prepare the data by creating the input data `X` and the target
# data `y`.

# %%
X = df[["Flipper Length (mm)"]]
y = df["Body Mass (g)"]

# %%
print(f"The dimensions of X are: {X.shape}")
print(f"The dimensions of y are: {y.shape}")

# %% [markdown]
#
# Here, the matrix `X` only contains a single feature. However, in the future, we might
# want to add several features that allow use to predict the target `y`. The target `y`
# is a one-dimensional array here meaning that we only predict a single target. Note
# that in some cases, it is possible that the target to be predicted is a
# multi-dimensional array.
#
# Also, here we try to predict a continuous target. This is why we are in a regression
# setting. In other cases, we might want to predict a categorical target. This is called
# a classification problem.
#
# Let's start to fit a scikit-learn model that is a simple linear regression model.

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X, y)

# %% [markdown]
#
# In scikit-learn, the method `fit` is used to train a model. In this case, it allows us
# to find the best parameters of the linear regression model to fit the data. These
# parameters are stored in the attributes `coef_` and `intercept_` of the instance.

# %%
linear_regression.coef_, linear_regression.intercept_

# %% [markdown]
#
# Let's now use this model to predict the body mass of the penguins based on their
# flipper length. We create a synthetic dataset of potential flipper length values and
# predict the body mass of the penguins using our model.

# %%
import numpy as np

X_to_infer = pd.DataFrame({"Flipper Length (mm)": np.linspace(175, 230, 100)})
y_pred = linear_regression.predict(X_to_infer)

# %% [markdown]
#
# The method `predict` allow us to get the prediction of the model on new data. Now,
# we plot the obtained values.

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.set_title("Penguin Flipper Length vs Body Mass")
ax.plot(X_to_infer, y_pred, linewidth=3, color="tab:orange", label="Linear Regression")
_ = ax.legend()

# %% [markdown]
#
# This `LinearRegression` model is known to minimize the mean squared error. We can
# compute this metric to know what would be the error on the same dataset that we used
# to train the model.

# %%
from sklearn.metrics import mean_squared_error

error = mean_squared_error(y, linear_regression.predict(X))
print(f"The mean squared error is: {error:.2f}")

# %% [markdown]
#
# ### Exercise
#
# Let's repeat the previous experiment by fitting again a linear model but this model
# is known as a quantile regression. You can import it from
# `sklearn.linear_model.QuantileRegressor`. Let's fit the median (look at the
# documentation to know which parameter to set).

# %%
from sklearn.linear_model import QuantileRegressor

# %% [markdown]
#
# Plot the prediction of the quantile regression model on the same plot as the linear
# regression model to have a quantitative comparison.
# Compute the mean squared error and compare it to the `LinearRegression` model.
# Compute the median absolute error and compare it to the `LinearRegression` model.
# Can you provide some insights.

# %% [markdown]
#
# Up to now, we have been evaluating the model on the same dataset but it does not tell
# us how well the model will generalize to new data. Let's imagine that we have a more
# complex model that make some data engineering. We can use a polynomial feature
# expansion to create a more complex model.
#
# Let's first demonstrate how to create a polynomial feature expansion with
# scikit-learn transformers.

# %%
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

polynomial_features = PolynomialFeatures(degree=5).set_output(transform="pandas")
standard_scaler = StandardScaler().set_output(transform="pandas")
X_scaled = standard_scaler.fit_transform(X)
X_poly = polynomial_features.fit_transform(X_scaled)
X_poly

# %% [markdown]
#
# Scikit-learn transformers are models that have a `fit` and `transform` methods. The
# `fit` method will compute the required statistics to transform the data. The
# `transform` method will apply the transformation to the data.
#
# ### Exercise
#
# Fit a `LinearRegression` model on the `X_poly` data and predict the body mass of the
# penguins. Plot the prediction on the same plot as the linear regression model and the
# quantile regression model. Compute the mean squared error and compare it to the
# previous models.

# %%
linear_regression.fit(X_poly, y)
X_to_infer_poly = polynomial_features.transform(standard_scaler.transform(X_to_infer))
y_pred_poly = linear_regression.predict(X_to_infer_poly)

# %% [markdown]
#
# Up to now, we have no way to compare the quality of this model with the previous
# model. To do so, we need to put ourself in a situation where we have a training set
# and a testing set. The training set is the set used to create the model while the
# testing set is used to evaluate the model on unseen data.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %%
from sklearn.metrics import mean_absolute_error

linear_regression.fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)
print(
    "Mean absolute error on the training set: "
    f"{mean_absolute_error(y_train, linear_regression.predict(X_train)):.2f}"
)
print(
    "Mean absolute error on the testing set: "
    f"{mean_absolute_error(y_test, linear_regression.predict(X_test)):.2f}"
)

# %% [markdown]
#
# Now, by computing the mean absolute error on the testing set, we have an estimate of
# potential generalization power of our models. Eventually, we could keep the best model
# that leads to the lowest error on the testing set.
#
# However, we the results above, we have no idea of the variability of the error. We
# might have been lucky while creating the training and testing set. To have a better
# estimate of the error, we can use cross-validation: we will repeat the splitting of
# the data into training and testing set several times.

# %%
from sklearn.model_selection import cross_validate, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_results = cross_validate(
    linear_regression,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
#
# The `cross_validate` function allows us to make this cross-validation and store the
# important results in a Python dictionary. Note that scikit-learn uses a "score"
# convention: the higher the score, the better the model. Since we used error metrics,
# this convention will force us to use the negative values of the error.
#
# To access these metrics, we can pass a string to the `scoring` parameter. For error,
# we need to add the `neg_` prefix and it leads to negative values in the report.
#
# We negate the values to get back meaningful values.

# %%
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]
cv_results[["train_score", "test_score"]].describe()

# %% [markdown]
#
# So now, we have an estimate of the mean absolute error and its variability. We can
# compare it with the model using the polynomial features.

# %%
cv_results = cross_validate(
    linear_regression,
    X_poly,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]
cv_results[["train_score", "test_score"]].describe()

# %% [markdown]
#
# We observe that the model using polynomial features has a lower mean absolute error
# than the linear model.
#
# However, we have an issue with the pattern used here. By scaling the full dataset and
# computing the polynomial features on the full dataset, we leak information from the
# the testing set to the training set. Therefore, the scores obtained might be too
# optimistic.
#
# We should therefore make the split before scaling the data and computing the
# polynomial features. Scikit-learn provides a `Pipeline` class that allows
# to chain several transformers and a final estimator.
#
# In this way, we can declare a pipeline that do not require any data during its
# declaration.

# %%
from sklearn.pipeline import make_pipeline

linear_regression_poly = make_pipeline(
    StandardScaler(), PolynomialFeatures(degree=5), LinearRegression()
)
linear_regression_poly

# %% [markdown]
#
# This sequence of transformers and final learner provide the same API as the final
# learner. Under the hood, it will call the proper methods when we call `fit` and
# `predict` methods.

# %%
linear_regression_poly.fit(X_train, y_train)
linear_regression_poly[-1].coef_, linear_regression_poly[-1].intercept_

# %%
print(
    "Mean absolute error on the training set: "
    f"{mean_absolute_error(y_train, linear_regression_poly.predict(X_train)):.2f}"
)
print(
    "Mean absolute error on the testing set: "
    f"{mean_absolute_error(y_test, linear_regression_poly.predict(X_test)):.2f}"
)

# %% [markdown]
#
# So now, we can safely use this model in the `cross_validate` function and pass the
# original data that will be transformed on-the-fly.

# %%
cv_results = cross_validate(
    linear_regression_poly,
    X,
    y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results["test_score"] = -cv_results["test_score"]
cv_results["train_score"] = -cv_results["train_score"]
cv_results[["train_score", "test_score"]].describe()

# %% [markdown]
#
# ### Exercise
#
# Use a `sklearn.model_selection.RepeatedKFold` cross-validation strategy to evaluate
# the performance of the linear regression model and the polynomial regression model.
#
# The idea is to repeat several times to be able to plot a distribution of the test
# scores.
