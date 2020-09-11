from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from ensemble_utils import train_test_split, clf_accuracy, best_selection_trees, early_stopping_trees
import xgboost


# Instead of taking one training model and working to optimize this individual result, taking many different training models with different errors and
# # applying majority vote with their results can yield better performance. This ensemble learning is most effective with indepenedent models.
X, y = make_moons(n_samples=10000, noise=0.15, shuffle=True, random_state=42)
X_train, y_train, X_test, y_test = train_test_split(9000, X, y)
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_rbf_clf = SVC(probability=True)

# Another method beyond using different models and aggregating is using different samples of the same dataset on the same model and aggregating over those models
# This is referred to as bagging if you sample with replacement or pasting if not
bag_dt_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, bootstrap_features=True, n_jobs=-1)
# Below is a more robust Random Forest Implementation as well as the bagging equivalent
rnd_clf = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
bag_rf_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16), n_estimators=500,
    max_samples=100, bootstrap=True, bootstrap_features=True, n_jobs=-1)
# Lastly Extra-Trees can be used in order to speed up the result for more bias by removing the retrival of optimal feature update
ext_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

# Generate the voting ensemble and record accuracy discrepencies
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc_rbf', svm_rbf_clf), ('bag_dt', bag_dt_clf), ('bag_rf', bag_rf_clf), ('ext_clf', ext_clf)],
    voting='soft')

# Another method that is useful is boosting, which incrementally updates models based on the previous models performance
# Below are ADABoost (update through weights of misclassification) and GradientBoost (update through residuals of previous model)
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME.R", learning_rate=0.5)
gbclf = GradientBoostingClassifier(max_depth=2, n_estimators=120, learning_rate=1.0)
# Self implementation of best selection
best_n = best_selection_trees(gbclf.fit(X_train, y_train), X_train, y_train, X_test, y_test)
best_gbclf = GradientBoostingClassifier(max_depth=2, n_estimators=best_n, learning_rate=1.0)
# Self implementation of early stop
gbclf = GradientBoostingClassifier(max_depth=2, n_estimators=120, learning_rate=1.0, warm_start=True)
es_gbclf = early_stopping_trees(gbclf, X_train, y_train, X_test, y_test, n_estimators=120)

# XGBoost Regression with Early Stop
xgb_clf = xgboost.XGBClassifier()
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=2)

# Stacking can also be done as a better mixer than voting from before
sclf = StackingClassifier(classifiers=[svm_rbf_clf, ext_clf, ada_clf, xgb_clf], meta_classifier=LogisticRegression())

# Output results from all models
models = (log_clf, rnd_clf, svm_rbf_clf, bag_dt_clf, bag_rf_clf, ext_clf, voting_clf, ada_clf, best_gbclf, es_gbclf, xgb_clf, sclf)
clf_accuracy(models, X_train, y_train, X_test, y_test)