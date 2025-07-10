from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Load model and vectorizer
model = pickle.load(open('viral.pkl', 'rb'))
vectorizer = pickle.load(open('vector.pkl', 'rb'))

# Load one-hot column lists and ensure they're flat lists
category_cols = list(pickle.load(open('category.pkl', 'rb')))
day_cols = list(pickle.load(open('days.pkl', 'rb')))
month_cols = list(pickle.load(open('month.pkl', 'rb')))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    text = request.form['Text_cleaned']
    followers = int(request.form['Followers'])
    likes = int(request.form['Likes'])
    category = request.form['Category']
    day_of_week = request.form['DayOfWeek']
    month = request.form['Month']
    time_str = request.form['time']  # Expected format: HH:MM:SS

    # Transform text using TF-IDF
    X_text = vectorizer.transform([text]).toarray()

    # Numeric metadata
    meta = np.array([[followers, likes]])

    # One-hot encode category
    cat_df = pd.DataFrame(
        [[1 if col == f"cat_{category}" else 0 for col in category_cols]],
        columns=category_cols
    )

    # One-hot encode day
    day_df = pd.DataFrame(
        [[1 if col == f"day_{day_of_week}" else 0 for col in day_cols]],
        columns=day_cols
    )

    # One-hot encode month
    month_df = pd.DataFrame(
        [[1 if col == f"month_{month}" else 0 for col in month_cols]],
        columns=month_cols
    )

    # Parse time
    try:
        time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
        time_features = np.array([[time_obj.hour, time_obj.minute, time_obj.second]])
    except ValueError:
        return render_template('index.html', prediction_text='❌ Invalid time format. Use HH:MM:SS')

    # Final feature array
    final_features = np.hstack((
        X_text,
        meta,
        cat_df.values,
        day_df.values,
        month_df.values,
        time_features
    ))

    # Predict
    prediction = model.predict(final_features)
    output = 'Viral Tweet 🚀' if prediction[0] == 1 else 'Not Viral Tweet 😐'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == '__main__':
    app.run(debug=True)
