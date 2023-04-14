from flask import Flask, request, jsonify, abort
from keras.models import load_model
from joblib import load

app = Flask(__name__)

MODEL_NAMES = {
    # "kernel": "Kernel SGD Classifier",
    "logistic": "Logistic Regression",
    "nn": "Neural Network",
    "sgd": "SGD Classifier"
}

ML_MODEL_NAMES = ['logistic', 'sgd']


@app.route('/alive', methods=['GET'])
def get_is_alive():
    return jsonify({"is_alive": True})


@app.route('/ml/names', methods=['GET'])
def get_ml_names():
    return jsonify({"model_names": ['Logistic Regression', 'SGD Classifier']})


def get_file_name_by_ml_model_name(name):
    # if name == "kernel":
    #     return "./models/kernel.joblib"
    if name == "logistic":
        return "./models/logistic_regression.joblib"
    if name == "sgd":
        return "./models/sgd.joblib"

    return ""


@app.route('/ml/<string:model_name>/predict', methods=['POST'])
def predict_ml(model_name):
    if model_name not in ML_MODEL_NAMES:
        abort(400, "ML models might be taken from the list [kernel, logistic, sgd]")
        return

    clf = load(get_file_name_by_ml_model_name(model_name))

    if not clf:
        abort(404, "The model is not trained :(")
        return

    data = request.get_json(force=True)

    y_predict = clf.predict(data['input'])

    return jsonify({model_name: MODEL_NAMES[model_name], "model_id": model_name, "predict": y_predict.tolist()})


@app.route('/input/example', methods=['GET'])
def get_ml_input_example():
    return jsonify({"example": [[2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0]]})


if __name__ == '__main__':
    app.run(debug=True)
