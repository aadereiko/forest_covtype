import keras.losses
import keras_tuner
import numpy as np
import pandas as pd
from joblib import dump

from keras import layers, optimizers

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

column_names = [
    'elevation',
    'aspect',
    'slope',
    'horizontal_distance_hydrology',
    'vertical_distance_hydrology',
    'horizontal_distance_roadways',
    'hillshade_9am',
    'hillshade_Noon',
    'hillshade_3pm',
    'horizontal_distance_to_fire_points',
    'wilderness_1',
    'wilderness_2',
    'wilderness_3',
    'wilderness_4',
    'soil_1',
    'soil_2',
    'soil_3',
    'soil_4',
    'soil_5',
    'soil_6',
    'soil_7',
    'soil_8',
    'soil_9',
    'soil_10',
    'soil_11',
    'soil_12',
    'soil_13',
    'soil_14',
    'soil_15',
    'soil_16',
    'soil_17',
    'soil_18',
    'soil_19',
    'soil_20',
    'soil_21',
    'soil_22',
    'soil_23',
    'soil_24',
    'soil_25',
    'soil_26',
    'soil_27',
    'soil_28',
    'soil_29',
    'soil_30',
    'soil_31',
    'soil_32',
    'soil_33',
    'soil_34',
    'soil_35',
    'soil_36',
    'soil_37',
    'soil_38',
    'soil_39',
    'soil_40',
    'cover_type'
]

df = pd.read_csv('./covtype.data',
                 header=None,
                 names=column_names
                 )

cover_types_unique = df['cover_type'].nunique()
encoder = LabelEncoder()

df['cover_type'] = encoder.fit_transform(df['cover_type'])


def data_overview():
    print("Type counts:", df['cover_type'].value_counts())
    # the problem is the data is imbalanced


def get_cut_points(data, number_of_points):
    split_data = np.array_split(np.sort(data), number_of_points)
    split_points = [points[-1] for points in split_data]

    return split_points


cut_points = get_cut_points(df['elevation'], cover_types_unique)


def search_insert_position(data, element):
    """
    returns the index where a given element should be put into a sorted array
    works for O(log n)
    """
    left, right = 0, len(data) - 1

    while left <= right:
        mid = (left + right) // 2

        if data[mid] == element:
            return mid
        elif data[mid] < element:
            left = mid + 1
        else:
            right = mid - 1

    return left


def get_cover_type_by_elevation(row):
    elevation = row['elevation']
    idx = search_insert_position(cut_points, elevation)

    return idx + 1


def task2():
    """
    Implement a very simple heuristic that will classify the data (It doesn't need to be accurate)
     - The implemented heuristic is based on the elevation column.
     We divide elevation values into equal parts the amount of which is equal to the quantity of cover types (7)
     and according to it bringing up the class
    """

    df['elevation_heuristic_cover_type'] = df.apply(get_cover_type_by_elevation, axis=1)
    print(np.sum(df['elevation_heuristic_cover_type'] == df['cover_type']) / len(df),
          '- the portion of correct answers')


def get_cover_type_heuristic(inputs):
    return [search_insert_position(cut_points, x[0]) + 1 for x in inputs]


# also has the task 5 inside
def task3():
    """
    Task 3.
    Use Scikit-learn library to train two simple Machine Learning models
    Choose models that will be useful as a baseline

    Since we have more than 100k and we have a task of classification,
    we could use SGD Classifier, kernel approximation
    """

    def predict_with_model(clf, X_train, y_train, X_test, y_test, address_to_save):
        print("Started training")
        clf.fit(X_train, y_train)
        print("Finished training")

        dump(clf, address_to_save)

        score = clf.score(X_train, y_train)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

        print("Score: ", score)
        print("Cross Validation scores", cv_scores)

        # Prediction
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix
        print("Confusion matrix", cm)

        cr = classification_report(y_test, y_pred)
        print("Classification Report", cr)

    X_train, X_test, y_train, y_test = train_test_split(df.drop('cover_type', axis=1),
                                                        df['cover_type'],
                                                        train_size=0.75,
                                                        random_state=21)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Logistic Regression
    logistic_clf = LogisticRegression()
    predict_with_model(logistic_clf, X_train, y_train, X_test, y_test, './models/logistic_regression.joblib')

    # SGD Classifier
    sgdc_clf = SGDClassifier()
    predict_with_model(sgdc_clf, X_train, y_train, X_test, y_test, './models/sgd.joblib')

    # Kernel approximation
    # nystroem = Nystroem(n_components=1000)
    # X_train_approx = nystroem.fit_transform(X_train)
    # X_test_approx = nystroem.transform(X_test)
    # sgdc_kernel_clf = SGDClassifier()

    # predict_with_model(sgdc_kernel_clf, X_train_approx, y_train, X_test_approx, y_test, './models/kernel.joblib')


# also has the task 5 inside
def task4():
    """
    Use TensorFlow library to train a neural network that will classify the data
        * Create a function that will find a good set of hyperparameters for the NN
        * Plot training curves for the best hyperparameters
    """
    X_train, X_test, y_train, y_test = train_test_split(df.drop('cover_type', axis=1),
                                                        df['cover_type'],
                                                        test_size=0.2,
                                                        random_state=42)

    def build_nn_model(hp):
        model = keras.Sequential()

        model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                               input_dim=X_train.shape[1]))
        model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(layers.Dense(units=7, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    tuner = keras_tuner.RandomSearch(
        build_nn_model,
        objective='val_accuracy',
        max_trials=5,
        directory='my_dir',
        project_name='forest_cover_type')

    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=2, validation_split=0.2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.save('./models/nn_model.h5')

    history = model.fit(X_train, y_train,
                        epochs=10,
                        validation_split=0.2,
                        verbose=0)

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print('Test accuracy:', test_acc)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    data_overview()
    task2()
    task3()
    task4()

