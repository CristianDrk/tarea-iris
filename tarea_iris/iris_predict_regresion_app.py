import pickle
from flask import Flask, jsonify, request
from iris_predict_service import predict_single

app = Flask('churn-predict')

with open('models/iris-regresion-model.pck', 'rb') as f:
    dv, model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        customer = request.get_json()["datos"]
        predictions = [predict_single(c, dv, model) for c in customer]

        result = {
            'probabilities': predictions,
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)  