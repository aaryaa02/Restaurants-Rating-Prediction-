from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings

# Initialize the Flask app
app = Flask(__name__)

# Load the machine learning model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))  # Adjust the path to your pickle file

# Suppress warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    online_order = int(request.form.get('online_order', 0))
    book_table = int(request.form.get('book_table', 0))
    votes = float(request.form.get('votes', 0))  # Changed to float
    location = int(request.form.get('location', 0))
    rest_type = int(request.form.get('rest_type', 0))
    review_sentimen= int(request.form.get('review_sentimen',0))
    cuisines = int(request.form.get('cuisines', 0))
    costfor2 = float(request.form.get('costfor2', 0))  # Changed to float
    type = int(request.form.get('type', 0))
    city = int(request.form.get('city', 0))  # 10th feature

    # Create a feature array for prediction
    features = np.array([[online_order, book_table, votes, location, rest_type, review_sentimen, cuisines, costfor2, type, city]])

    # Predict using the loaded model
    prediction = model.predict(features)

    # Render the same template with the prediction result
    return render_template('index.html', prediction_text='Predicted Zomato Rating: {:.2f}'.format(prediction[0]))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)