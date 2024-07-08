# %% [markdown]
#
# # Cross-validation strategies
#
# The previous notebooks introduced how to evaluate a model and how to create a
# specific preprocessing pipeline depending of the last model.
#
# In this notebook, we will check a bit more some details regarding the cross-validation
# strategies and some of the pitfalls that we can encounter.
#
# Let's take iris dataset and evaluate a logistic regression model.

# %%
from sklearn.datasets import load_iris

df, target = load_iris(as_frame=True, return_X_y=True)

# %%
df

# %%
target

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

logistic_regression = make_pipeline(StandardScaler(), LogisticRegression())
logistic_regression

# %% [markdown]
#
# ### Exercise
#
# Evaluate the previous `LogisticRegression` model using a 3-fold cross-validation
# (i.e. `sklearn.model_selection.KFold`). What do you observe?

# %%

# %% [markdown]
#
# We observe that the training score is always zero that is really surprising. We can
# check the target to understand why.

# %% [markdown]
#
# We can use a `StratifiedKFold` object to ensure that the class distribution is
# preserved in each fold. A side effect will be that all classes will be present in the
# training set and testing set.

# %%
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=3)
cv_results = cross_validate(
    logistic_regression, df, target, cv=cv, return_train_score=True
)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %%
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        f"Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts()}"
    )
    print(
        f"Class counts on the test set:\n" f"{target.iloc[test_indices].value_counts()}"
    )
    print()

# %% [markdown]
#
# This is particularly useful when we have imbalanced classes. Let's check the class
# distribution of the breast cancer dataset.

# %%
from sklearn.datasets import load_breast_cancer

df, target = load_breast_cancer(as_frame=True, return_X_y=True)

# %%
target.value_counts(normalize=True)

# %% [markdown]
#
# Here, we see that the proportion of the two classes is not equal. We can check the
# class distribution in each fold using a `KFold` object.

# %%
cv = KFold(n_splits=3, shuffle=True, random_state=0)
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        "Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts(normalize=True)}\n"
    )
    print(
        f"Class counts on the test set:\n"
        f"{target.iloc[test_indices].value_counts(True)}"
    )
    print()

# %% [markdown]
#
# We observe that the class distribution is not preserved in each fold. We can use a
# `StratifiedKFold` object to ensure that the class distribution is preserved in each
# fold.

# %%
cv = StratifiedKFold(n_splits=3)
for cv_fold_idx, (train_indices, test_indices) in enumerate(cv.split(df, target)):
    print(f"Fold {cv_fold_idx}:\n")
    print(
        "Class counts on the train set:\n"
        f"{target.iloc[train_indices].value_counts(normalize=True)}\n"
    )
    print(
        f"Class counts on the test set:\n"
        f"{target.iloc[test_indices].value_counts(True)}"
    )
    print()

# %% [markdown]
#
# Now, let's check the documentation of the `cross_validate` function to see if this
# function was already providing a way to stratify the data.

# %%
help(cross_validate)

# %% [markdown]
#
# Now, we will look at the notion of `groups` in cross-validation. We will use the
# digits dataset and group the samples by writer.

# %%
from sklearn.datasets import load_digits

df, target = load_digits(return_X_y=True)

# %% [markdown]
#
# We create a simple model that is a logistic regression model with a scaling of the
# data.

# %%
from sklearn.preprocessing import MinMaxScaler

logistic_regression = make_pipeline(MinMaxScaler(), LogisticRegression())

# %% [markdown]
#
# ### Exercise
#
# Make an evaluation and compare 2 different strategies:
# - using a `KFold` object with 13 splits without shuffling the data;
# - using a `KFold` object with 13 splits with shuffling the data.
#
# What do you observe? What are the causes of the differences?

# %% [markdown]
#
# Here, we provide a `groups` array that mentioned the writer ID for each sample.

# %%
from itertools import count
import numpy as np

# defines the lower and upper bounds of sample indices
# for each writer
writer_boundaries = [
    0,
    130,
    256,
    386,
    516,
    646,
    776,
    915,
    1029,
    1157,
    1287,
    1415,
    1545,
    1667,
    1797,
]
groups = np.zeros_like(target)
lower_bounds = writer_boundaries[:-1]
upper_bounds = writer_boundaries[1:]

for group_id, lb, up in zip(count(), lower_bounds, upper_bounds):
    groups[lb:up] = group_id

# %%
import matplotlib.pyplot as plt

plt.plot(groups)
plt.yticks(np.unique(groups))
plt.xticks(writer_boundaries, rotation=90)
plt.xlabel("Target index")
plt.ylabel("Writer index")
_ = plt.title("Underlying writer groups existing in the target")

# %% [markdown]
#
# We can use this information to properly evaluate our model. We need to use the
# `GroupKFold` object and pass the `groups` parameter to the `cross_validate` function.

# %%
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=13)
cv_results = cross_validate(logistic_regression, df, target, groups=groups, cv=cv)
print(
    f"Mean test score: {cv_results['test_score'].mean():.3f} +/- "
    f"{cv_results['test_score'].std():.3f}"
)

# %% [markdown]
#
# We observe that the mean test score is even lower but certainly closer to the true
# performance of the model.
