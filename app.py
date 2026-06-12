import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

model = joblib.load('models/model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
feature_names = joblib.load('models/feature_names.pkl')

def encode_input(data):
    df = pd.DataFrame([data])
    for col, le in label_encoders.items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()

        input_data = {
            'age': int(form_data.get('age', 18)),
            'Medu': int(form_data.get('Medu', 2)),
            'Fedu': int(form_data.get('Fedu', 2)),
            'traveltime': int(form_data.get('traveltime', 1)),
            'studytime': int(form_data.get('studytime', 2)),
            'failures': int(form_data.get('failures', 0)),
            'famrel': int(form_data.get('famrel', 4)),
            'freetime': int(form_data.get('freetime', 3)),
            'goout': int(form_data.get('goout', 3)),
            'Dalc': int(form_data.get('Dalc', 1)),
            'Walc': int(form_data.get('Walc', 2)),
            'health': int(form_data.get('health', 4)),
            'absences': int(form_data.get('absences', 0)),
            'sex': form_data.get('sex', 'F'),
            'address': form_data.get('address', 'U'),
            'famsize': form_data.get('famsize', 'LE3'),
            'Pstatus': form_data.get('Pstatus', 'T'),
            'Mjob': form_data.get('Mjob', 'other'),
            'Fjob': form_data.get('Fjob', 'other'),
            'reason': form_data.get('reason', 'other'),
            'guardian': form_data.get('guardian', 'mother'),
            'schoolsup': form_data.get('schoolsup', 'no'),
            'famsup': form_data.get('famsup', 'no'),
            'paid': form_data.get('paid', 'no'),
            'activities': form_data.get('activities', 'no'),
            'nursery': form_data.get('nursery', 'yes'),
            'higher': form_data.get('higher', 'yes'),
            'internet': form_data.get('internet', 'no'),
            'romantic': form_data.get('romantic', 'no'),
        }

        df = encode_input(input_data)

        df = df[feature_names]

        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)
        prediction = max(0, min(20, prediction))

        return render_template('result.html', prediction=prediction, data=form_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('result.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        df = encode_input(data)
        df = df[feature_names]
        prediction = model.predict(df)[0]
        prediction = round(float(prediction), 2)
        prediction = max(0, min(20, prediction))
        return jsonify({'prediction': prediction, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
