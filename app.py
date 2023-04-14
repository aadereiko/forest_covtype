from flask import Flask, request, jsonify, abort
from main import get_cover_type_heuristic
from helpers import handle_ml_request, handle_nn_request

app = Flask(__name__)

MODEL_NAMES = {
    # "kernel": "Kernel SGD Classifier",
    "logistic": "Logistic Regression",
    "nn": "Neural Network",
    "sgd": "SGD Classifier"
}

ML_MODEL_NAMES = ['logistic', 'sgd']
NN_MODEL_NAMES = ['nn']
HEURISTICS_NAMES = ['heuristics']

MODEL_NAMES = ML_MODEL_NAMES + NN_MODEL_NAMES + HEURISTICS_NAMES


@app.route('/', methods=['GET'])
def get_main_page():
    return jsonify({"is_alive": True})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    model_name, input_data = data["model_name"], data["input"]

    if model_name not in MODEL_NAMES:
        abort(400, "Model might be taken from the list [logistic, sgd, nn, heuristics]")
        return

    if not input_data:
        abort(400, "No input data")
        return

    if model_name in ML_MODEL_NAMES:
        return jsonify({
            "model_type": "machine_learning",
            "predict": handle_ml_request(data, model_name)
        })

    if model_name in NN_MODEL_NAMES:
        y_predict = handle_nn_request(data)

        return jsonify({"predict": y_predict, "model_type": "neural networks"})

    if model_name in HEURISTICS_NAMES:
        y_predict = get_cover_type_heuristic(data['input'])

        return jsonify({"predict": y_predict, "model_type": "no model. heuristics"})

    abort(400)


@app.route('/input/example', methods=['GET'])
def get_ml_input_example():
    return jsonify({"example": [[2596, 51, 3, 258, 0, 510, 221, 232, 148, 6279, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0]]})


if __name__ == '__main__':
    app.run(debug=True)
