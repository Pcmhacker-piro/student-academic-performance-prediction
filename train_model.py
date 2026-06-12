import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.makedirs('models', exist_ok=True)

stud = pd.read_csv('student-mat.csv')

label_encoders = {}
categorical_col_names = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                         'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                         'famsup', 'paid', 'activities', 'nursery', 'higher',
                         'internet', 'romantic']

for col_name in categorical_col_names:
    le = LabelEncoder()
    stud[col_name] = stud[col_name].astype(str)
    stud[col_name] = le.fit_transform(stud[col_name])
    label_encoders[col_name] = le

stud = stud.drop(['school', 'G1', 'G2'], axis='columns')

most_correlated = stud.corr().abs()['G3'].sort_values(ascending=False)
most_correlated = most_correlated[:9]
top_features_with_target = most_correlated.index.tolist()
stud = stud.loc[:, top_features_with_target]

feature_names = [f for f in top_features_with_target if f != 'G3']
X = stud[feature_names]
y = stud['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'models/model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f'Model trained successfully!')
print(f'Training R² score: {train_score:.4f}')
print(f'Testing R² score: {test_score:.4f}')
print(f'Features used: {feature_names}')
