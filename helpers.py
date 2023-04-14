from keras.models import load_model
from joblib import load
from flask import abort


def get_file_name_by_ml_model_name(name):
    # if name == "kernel":
    #     return "./models/kernel.joblib"
    if name == "logistic":
        return "./models/logistic_regression.joblib"
    if name == "sgd":
        return "./models/sgd.joblib"

    return ""


def handle_ml_request(data, model_name):
    clf = load(get_file_name_by_ml_model_name(model_name))

    if not clf:
        abort(404, "The model is not trained :(")
        return

    y_predict = clf.predict(data['input'])

    return y_predict.tolist()


def handle_nn_request(data):
    model = load_model('./models/nn_model.h5')
    if not model:
        abort(404, "The model is not trained :(")
        return

    y_predict = model.predict(data['input'])

    return y_predict.tolist()
