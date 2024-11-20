# Disease Prediction System 🩺💉

## Project Description
This project aims to predict the likelihood of diseases based on user inputs, utilizing machine learning models and exploratory data analysis (EDA). The system takes various health parameters and outputs a prediction for potential diseases. It helps users take preventive actions and seek timely medical attention.

---

## **Features of the System**

### **Data Collection**
- Data sourced from healthcare datasets containing information on various health parameters and disease history.
- Includes features such as age, weight, blood pressure, cholesterol, and more, depending on the type of disease being predicted.

### **Data Cleaning**
- Handled missing values and outliers.
- Standardized the input features for better model performance.
- Encoded categorical variables (e.g., gender, smoking history, etc.).

### **Feature Engineering**
- Created additional derived features based on existing data, such as Body Mass Index (BMI) from weight and height.
- Handled feature scaling using techniques like normalization or standardization for model accuracy.

### **Visualization**
- Distribution plots to visualize the spread of numerical features like age, cholesterol, etc.
- Correlation heatmap to identify relationships between features and the target disease.
- Boxplots to check for outliers in important health indicators.

### **Modeling**
- Applied classification algorithms like Logistic Regression, Decision Trees, Random Forests, or Support Vector Machines (SVM) for disease prediction.
- Model evaluation using metrics like accuracy, precision, recall, and F1-score.

---

## **Tools and Libraries Used**
- **Programming Language**: Python
- **Data Analysis**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Web App Deployment**: Streamlit (for real-time predictions)

---

## **Key Insights**
- Health parameters like age, cholesterol, and smoking history are significant factors influencing disease prediction.
- Log transformations helped normalize skewed features such as BMI and cholesterol.
- The model can accurately predict the likelihood of diseases based on key health indicators, assisting in early detection and prevention.

---

## **Visuals**
The analysis and model include:
- **Histograms and Boxplots**: To explore the distribution and outliers of health indicators.
- **Bar Charts**: For categorical feature comparisons (e.g., gender, smoking history).
- **Correlation Heatmap**: To understand the relationships between health parameters and the predicted disease.

---

## **How to Run**

1. Clone the repository:
   git clone https://github.com/yourusername/disease-prediction.git
   
2.Ensure the required files are in place:
dataset.csv: The health dataset used for model training.
model.pkl: The trained machine learning model for predictions.

3.Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn streamlit

4.Run the Streamlit application:
streamlit run app.py

---

## **Example Workflow**

User Input:
Input health parameters such as age, blood pressure, cholesterol levels, BMI, etc.

Prediction:
The model predicts the likelihood of specific diseases (e.g., diabetes, heart disease, etc.) based on the input parameters.

Output:
A probability score indicating the likelihood of the user contracting a disease, along with relevant recommendations.

---
## **Future Scope**

Integrate additional datasets for a broader range of diseases.

Implement deep learning models for improved prediction accuracy.

Expand the web interface with more user-friendly features like visual explanations of the predictions.

Deploy the model as a REST API for integration with healthcare applications.

---

## **Dataset Information**

The dataset used in this project is from publicly available healthcare datasets (e.g., Diabetes, Heart Disease datasets). The dataset contains health-related attributes and disease outcomes.

