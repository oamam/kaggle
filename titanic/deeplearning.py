import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('preprocessed_train.csv')
X_train = train.drop('Survived', axis=1)
y_train = train.Survived

X_test = pd.read_csv('preprocessed_test.csv')
y_test = pd.read_csv('gender_submission.csv').Survived


def create_model(activation="relu", optimizer="adam", out_dim=10, layer_num=1):
    model = Sequential()
    model.add(Dense(out_dim, activation=activation, input_dim=17))

    for _ in range(0, layer_num):
        model.add(Dense(out_dim, activation=activation))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


def get_optimum_model(X_train, y_train):
    activation = ["relu", "sigmoid"]
    optimizer = ["adam", "adagrad"]
    out_dim = [10, 20]
    nb_epoch = [400, 600]
    batch_size = [32, 64]
    layer_num = [5, 10]
    model = KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = dict(
        activation=activation,
        optimizer=optimizer,
        out_dim=out_dim,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        layer_num=layer_num
    )
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train)
    print(grid_result.best_score_)
    print(grid_result.best_params_)

# get_optimum_model(X_train, y_train)

epochs = 400
batch_size = 32

model = create_model('relu', 'adagrad', 10, 10)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

loss_and_metrics = model.evaluate(X_test, y_test)
classies = model.predict_classes(X_test)
prob = model.predict_proba(X_test)
print(loss_and_metrics)


def output(classies):
    predict = []
    for v in classies:
        predict.extend(v)
    data = {
        'PassengerId': pd.read_csv('gender_submission.csv').PassengerId,
        'Survived': predict
    }
    df = pd.DataFrame(data=data)
    df = df.set_index('PassengerId')
    df.to_csv('deeplearning_predict.csv')
output(classies)
