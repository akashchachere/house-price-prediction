from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        features = [float(request.form[key]) for key in
                    ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]

        # Convert to NumPy array
        input_features = np.array([features]).reshape(1, -1)

        # Predict price
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction_text=f"Estimated House Price: ${round(prediction, 2)}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
