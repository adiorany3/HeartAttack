# Heart Attack Prediction Model

This project implements a heart attack prediction model using logistic regression. The model is trained on a dataset containing various health-related features and aims to predict the likelihood of a heart attack.

## Project Structure

- **Jantung.py**: Contains the heart attack prediction model, including data loading, feature selection, data splitting, feature standardization, model training, and evaluation metrics.
- **streamlit_app.py**: The main entry point for the Streamlit application. It imports the necessary libraries, loads the model from Jantung.py, creates a user interface for inputting data, and displays the prediction results along with the model evaluation metrics.
- **requirements.txt**: Lists the dependencies required for the project, including Streamlit and other libraries used in Jantung.py.
- **README.md**: Documentation for the project, including setup instructions and an overview of the heart attack prediction model.

## Setup Instructions

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```
   streamlit run streamlit_app.py
   ```

5. Open your web browser and go to the URL provided in the terminal to access the application.

## Model Overview

The heart attack prediction model uses the following features:

- Age
- Gender
- Cholesterol
- Blood Pressure
- Heart Rate
- Smoker
- Diabetes
- Hypertension
- Family History
- Stress Level

The model outputs the predicted outcome and provides evaluation metrics such as accuracy and a classification report.