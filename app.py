from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/alive', methods=['GET'])
def get_is_alive():
    return jsonify({"is_alive": True})


if __name__ == '__main__':
    app.run(debug=True)