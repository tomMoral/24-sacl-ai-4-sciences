# %% [markdown]
#
# # Notes related to classification metrics
#
# This notebook goes a bit deeper on classification metrics. We are going to get back
# to the penguins dataset.
#
# Let's start by crafting a machine learning pipeline that we used in a previous
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

# %% [markdown]
#
# In addition, we will simplify the problem to a binary classification task by keeping
# only two classes.

# %%
classes_to_keep = [
    "Adelie Penguin (Pygoscelis adeliae)",
    "Chinstrap penguin (Pygoscelis antarctica)",
]
X = X[y.isin(classes_to_keep)]
y = y[y.isin(classes_to_keep)]

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hist_gradient_boosting = HistGradientBoostingClassifier(
    categorical_features="from_dtype"
)

# %% [markdown]
#
# We used in the previous notebook the `cross_validate` function to evaluate the
# performance of the model.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(hist_gradient_boosting, X, y, cv=5, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %% [markdown]
#
# The score used by default in this case is the accuracy. However, accuracy is not
# always the best metric to evaluate the performance of a classifier. It is actually
# an issue with imbalanced datasets. We can check the ratio of the class on our dataset.

# %%
y.value_counts(normalize=True)

# %% [markdown]
#
# So we already see if we predict always the majority class, we will get an accuracy of
# around 0.7. Therefore, when interpreting the accuracy obtained from the
# cross-validation, we should keep this in mind.
#
# Otherwise, we have some other metrics that we could use to evaluate the performance of
# a classifier. To check these metrics, we can check the `classification_report`
# function.

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
hist_gradient_boosting.fit(X_train, y_train)

print(classification_report(y_test, hist_gradient_boosting.predict(X_test)))

# %% [markdown]
#
# These metrics are computed from the confusion matrix. To provide an example, we have
# the following confusion matrix:

# %%
from sklearn.metrics import ConfusionMatrixDisplay

display = ConfusionMatrixDisplay.from_predictions(
    hist_gradient_boosting.predict(X_test), y_test
)
_ = display.ax_.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Adelie", "Chinstrap"],
    yticklabels=["Adelie", "Chinstrap"],
)

# %% [markdown]
#
# An important point to notice is that we used the method `predict` to compute these
# metrics. However, when dealing with classification, we could use the probabilistic
# properties of our classifier. Here, we will illustrate what are the probabilities
# output provided by the classifier and how those are transformed into a class.
#
# The method `predict_proba` provides the probabilities of each class.

# %%
y_pred = hist_gradient_boosting.predict_proba(X_test)
pd.DataFrame(y_pred, columns=hist_gradient_boosting.classes_).head()

# %% [markdown]
#
# We can convince ourselves that the sum of the probabilities is equal to 1.

# %%
y_pred.sum(axis=1)

# %% [markdown]
#
# The method `predict` is using a threshold of 0.5 to convert the probabilities into
# classes. This could be done by taking the `argmax` of the probabilities.

# %%
hist_gradient_boosting.classes_[y_pred.argmax(axis=1)]

# %%
(
    hist_gradient_boosting.classes_[y_pred.argmax(axis=1)]
    == hist_gradient_boosting.predict(X_test)
)

# %% [markdown]
#
# So it means that the threshold of 0.5 is completely arbitrary. We could change this
# threshold and compute the confusion matrix and subsequent metrics. It will therefore
# create a curve.
#
# In classification, there is two main curves that are known: the ROC curve and the
# precision-recall curve. They usually show some trade-off regarding our classifier.
# Let's start to look at the precision-recall curve.

# %%
from sklearn.metrics import PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %% [markdown]
#
# We can plot the value for the default threshold of 0.5.

# %%
from sklearn.metrics import precision_score, recall_score

precision = precision_score(
    y_test,
    hist_gradient_boosting.predict(X_test),
    pos_label=hist_gradient_boosting.classes_[1],
)
recall = recall_score(
    y_test,
    hist_gradient_boosting.predict(X_test),
    pos_label=hist_gradient_boosting.classes_[1],
)

# %%
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Fixed threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %% [markdown]
#
# But we could try any other threshold and observe that we are moving on this curve.
# We use the `FixedThresholdClassifier` by setting the threshold to 0.1.

# %%
from sklearn.model_selection import FixedThresholdClassifier

classifier = FixedThresholdClassifier(hist_gradient_boosting, threshold=0.1)
classifier.fit(X_train, y_train)
precision = precision_score(
    y_test, classifier.predict(X_test), pos_label=classifier.classes_[1]
)
recall = recall_score(
    y_test, classifier.predict(X_test), pos_label=classifier.classes_[1]
)
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Fixed threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %% [markdown]
#
# We could look at another trade-off by plotting the ROC curve.

# %%
from sklearn.metrics import RocCurveDisplay

display = RocCurveDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
_ = display.ax_.set(
    xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC curve"
)

# %% [markdown]
#
# The usage of these curves are chosen depending on the problem at end and the
# community.
#
# One important point concept to understand here is that these curves are evaluating
# the discriminative power of a classifier or in other words the ranking of the
# predictions.
#
# However, we don't evaluate if the probability estimates of the classifier are
# precise as known as well-calibrated. We can check the calibration of a classifier
# using the `CalibrationDisplay`. This is also known as a reliability diagram.

# %%
from sklearn.calibration import CalibrationDisplay

display = CalibrationDisplay.from_estimator(
    hist_gradient_boosting, X, y, n_bins=5, pos_label=hist_gradient_boosting.classes_[1]
)
_ = display.ax_.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")

# %% [markdown]
#
# Our classifier seems to be quite well calibrated. Let's create a model that is not
# well calibrated to see the difference.

# %%
from sklearn.linear_model import LogisticRegression
from skrub import tabular_learner

logistic_regression = tabular_learner(LogisticRegression(C=1e-2)).fit(X_train, y_train)
logistic_regression

# %%
display = CalibrationDisplay.from_estimator(
    hist_gradient_boosting, X, y, n_bins=5, pos_label=hist_gradient_boosting.classes_[1]
)
CalibrationDisplay.from_estimator(
    logistic_regression,
    X,
    y,
    n_bins=5,
    pos_label=logistic_regression.classes_[1],
    ax=display.ax_,
    name="LogisticRegression",
)
_ = display.ax_.set(xlabel="Mean predicted probability", ylabel="Fraction of positives")

# %% [markdown]
#
# Here, the logistic regression does not follow the diagonal line. It means that the
# probabilities are not well calibrated.
#
# In practice, there are two metrics that could provide information regarding the
# quality of the probabilities: the log loss and the Brier score.

# %%
from sklearn.metrics import log_loss

log_loss_hgbdt = log_loss(y_test, hist_gradient_boosting.predict_proba(X_test))
log_loss_rf = log_loss(y_test, logistic_regression.predict_proba(X_test))

print(f"Log loss of the HistGradientBoostingClassifier: {log_loss_hgbdt:.2f}")
print(f"Log loss of the LogisticRegression: {log_loss_rf:.2f}")

# %%
from sklearn.metrics import brier_score_loss


brier_score_hgbdt = brier_score_loss(
    y_test,
    hist_gradient_boosting.predict_proba(X_test)[:, 1],
    pos_label=hist_gradient_boosting.classes_[1],
)
brier_score_rf = brier_score_loss(
    y_test,
    logistic_regression.predict_proba(X_test)[:, 1],
    pos_label=logistic_regression.classes_[1],
)

print(f"Brier score of the HistGradientBoostingClassifier: {brier_score_hgbdt:.2f}")
print(f"Brier score of the LogisticRegression: {brier_score_rf:.2f}")

# %% [markdown]
#
# We observed that the log loss and the Brier score are lower for the
# `HistGradientBoostingClassifier` than for the `LogisticRegression`.
#
# # Bonus section
#
# We will do this part if we have time. In the previous section, we saw that the
# `FixedThresholdClassifier` is a way to change the threshold of the classifier.
# However, we might want to find the best threshold that maximizes a metric.
#
# This is the job of the `TunedThresholdClassifierCV`.

# %%
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %% [markdown]
#
# Let's find on this curve, which model is maximizing the F1-score.

# %%
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

# we need to pass pos_label because we have a binary classifier with classes other than
# 0 and 1 or -1 and 1
scorer = make_scorer(f1_score, pos_label=hist_gradient_boosting.classes_[1])
tuned_threshold_classifier = TunedThresholdClassifierCV(
    hist_gradient_boosting,
    cv=3,
    scoring=scorer,
).fit(X_train, y_train)

# %%
tuned_threshold_classifier.best_threshold_

# %%
precision = precision_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
recall = recall_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Tuned threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %% [markdown]
#
# Here, we therefore found the threshold that maximizes the F1-score. We could do the
# same for other metrics.
#
# Actually, we can go a bit further by maximizing a metric while having a constraint on
# another metric. For instance, we might want to maximize the recall while having a
# precision greater than 0.5.
#
# In this case, we need to define a custom scorer.

# %%
import numpy as np


def max_recall_at_min_precision(y_true, y_pred, min_precision, pos_label):
    precision = precision_score(y_true.tolist(), y_pred, pos_label=pos_label)
    recall = recall_score(y_true.tolist(), y_pred, pos_label=pos_label)
    if precision < min_precision:
        return -np.inf
    return recall


scorer = make_scorer(
    max_recall_at_min_precision,
    min_precision=0.5,
    pos_label=hist_gradient_boosting.classes_[1],
)
tuned_threshold_classifier.set_params(scoring=scorer, store_cv_results=True).fit(
    X_train, y_train
)

# %%
tuned_threshold_classifier.best_threshold_

# %%
precision = precision_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
recall = recall_score(
    y_test,
    tuned_threshold_classifier.predict(X_test),
    pos_label=tuned_threshold_classifier.classes_[1],
)
display = PrecisionRecallDisplay.from_estimator(hist_gradient_boosting, X_test, y_test)
display.ax_.plot(recall, precision, marker="o", label="Tuned threshold classifier")
display.ax_.legend()
_ = display.ax_.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")

# %%
