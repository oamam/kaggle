import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, accuracy_score

train = pd.read_csv('preprocessed_train.csv')
X_train = train.drop('Survived', axis=1)
y_train = train.Survived

X_test = pd.read_csv('preprocessed_test.csv')
sample = pd.read_csv('gender_submission.csv')
y_test = sample.Survived


def get_optimum_model(do_grid_search=False):
    if do_grid_search is False:
        return RandomForestClassifier(
            max_depth=None,
            max_leaf_nodes=10,
            min_samples_leaf=1,
            min_samples_split=50,
            n_estimators=100,
            n_jobs=-1
        )
    param_grid = [{
        'max_depth': [None, 10],
        'max_leaf_nodes': [10, 50, 100],
        'min_samples_leaf': [1, 5, 10],
        'min_samples_split': [10, 50],
        'n_estimators': [50, 100]
    }]
    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=3,
        scoring=make_scorer(accuracy_score)
    )
    return grid_search

do_grid_search = False
model = get_optimum_model(do_grid_search)
if (do_grid_search):
    re = model.fit(X_train, y_train)
    print('Best socre: {}'.format(re.best_score_))
    print('Best params: {}'.format(re.best_params_))
    print('Feature importances: {}'.format(
        re.best_estimator_.feature_importances_
    ))

else:
    model.fit(X_train, y_train)
predict = model.predict(X_test)
print('Train score: {}'.format(model.score(X_train, y_train)))
print('Test score: {}'.format(model.score(X_test, y_test)))
df = pd.DataFrame(data={
    'PassengerId': sample.PassengerId,
    'Survived': predict}
)
df = df.set_index('PassengerId')
df.to_csv('random_forest_predict.csv')
