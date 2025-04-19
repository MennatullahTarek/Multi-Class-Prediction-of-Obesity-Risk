import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to handle unseen categories safely during transformation
def safe_transform(encoder, df, feature, column):
    try:
        # Try to transform the feature using the encoder
        return encoder.transform(df[[feature]])
    except ValueError as e:
        # If an unseen category is found, handle it by using the first category or a default value
        st.warning(f"⚠️ Unseen category detected in {column}. Using default value.")
        
        # If the encoder is a LabelEncoder, we can use its classes_
        if hasattr(encoder, 'classes_'):
            default_value = encoder.classes_[0]  # Default to the first class
            return encoder.transform([default_value])
        else:
            # Fallback if classes_ attribute is not available
            return encoder.transform([encoder.categories_[0][0]])  # Using the first category from the encoder

# Sample data loading (make sure you replace with your actual data loading process)
# For this example, assuming the dataset has been pre-processed
df = pd.read_csv('your_dataset.csv')  # Adjust the path to your dataset

# Initializing LabelEncoder and OrdinalEncoder
label_gender = LabelEncoder()
label_FAVC = LabelEncoder()
label_SCC = LabelEncoder()
label_smoke = LabelEncoder()

encoder_CAEC = OrdinalEncoder()
encoder_MTRANS = OrdinalEncoder()
encoder_history = LabelEncoder()

# Fit the LabelEncoders on columns with categorical data
label_gender.fit(df['Gender'])
label_FAVC.fit(df['FAVC'])
label_SCC.fit(df['SCC'])
label_smoke.fit(df['SMOKE'])
encoder_CAEC.fit(df[['CAEC']])  # For ordinal columns
encoder_MTRANS.fit(df[['MTRANS']])
encoder_history.fit(df['family_history_with_overweight'])

# Transform categorical columns using the safe_transform function
df['family_history_with_overweight'] = safe_transform(encoder_history, df, 'family_history_with_overweight', 'Family History with Overweight')
df['Gender'] = safe_transform(label_gender, df, 'Gender', 'Gender')
df['CALC'] = safe_transform(encoder_CAEC, df, 'CALC', 'Alcohol Consumption')
df['FAVC'] = safe_transform(label_FAVC, df, 'FAVC', 'Frequent High-Calorie Food')
df['SCC'] = safe_transform(label_SCC, df, 'SCC', 'Calorie Monitoring')
df['SMOKE'] = safe_transform(label_smoke, df, 'SMOKE', 'Do you smoke?')

# Split the dataset into features (X) and target (y)
X = df.drop(columns=['target_column'])  # Replace 'target_column' with the actual column name for the target
y = df['target_column']  # The target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (RandomForestClassifier as an example)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.success(f"🎉 Model Accuracy: {accuracy * 100:.2f}%")

# Display results
label = label_FAVC.inverse_transform(y_pred)  # Using LabelEncoder to convert prediction back to original labels
st.success(f"🎉 Your Predicted Obesity Risk Level is: **{label[0]}**")
st.balloons()

# If needed, display the dataframe to check the transformations
st.dataframe(df.head())  # Display the top rows of the dataframe
