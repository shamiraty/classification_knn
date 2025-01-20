# KNN ALGORITHM AND PREDICTION WITH MYSQL

## Main Objective

The main objective of this project is to implement the **K-Nearest Neighbors (KNN)** algorithm to predict future customer types based on features such as **Age** and **Income**.

### Key Goals:
1. **Data Training**: The model will be trained using data stored in a **MySQL database**, which is collected from research observations. The data includes customer details like **Age**, **Income**, and **Customer Type**.
2. **Prediction**: Once trained, the model will allow predictions to be made on new customer records, and the predicted customer type will be saved back into the MySQL database for future reference and analysis.

___

## Simple Analytics

### Customer Type Analysis
The project provides an analysis of the customer dataset, focusing on the following key metrics:

1. **Customer Type Count**: Displays the total count of customers in each customer type category, showing the distribution of customer types.
2. **Age Insights**: Key statistics on the age, including the maximum age, minimum age, and the age range within each customer type, along with the standard deviation (age variation) to understand how much the ages vary.
3. **Overall Customer Insights**: Basic statistics about the entire dataset, including the total number of unique customers, the maximum and minimum ages, and the overall age range.

These analytics provide valuable insights into the dataset, helping to interpret the distribution of customer types and better understand the underlying patterns based on age and income.

---

## MySQL Database Connection and Data Handling

1. **MySQL Connection Setup**: Establishes a connection to the MySQL database to read and write customer data.
2. **Read Data**: Retrieves customer data (Age, Income, Customer Type) from the database for use in model training and predictions.
3. **Save Predictions**: After making predictions, the model saves the predicted customer type back into the database for future use and analysis.

---

## Customer Analytics

The app groups the customer data by **Customer_Type** and calculates the following metrics:
- **Customer Count** by category.
- **Maximum Age**, **Minimum Age**, **Age Range**, and **Age Variation** (Standard Deviation).

The results are displayed using Streamlit's metrics and tables.

---

## KNN Model Training and Prediction

1. **Feature Selection**: The features used for model training include **Age** and **Income**, while the target variable is **Customer_Type**.
2. **Data Split**: The dataset is split into training and test sets (80% training and 20% testing).
3. **KNN Classifier**: The K-Nearest Neighbors (KNN) model is trained with `k=3` neighbors.
4. **Model Accuracy**: The accuracy of the model is calculated and displayed, showing how well the model performs on the test data.
5. **Prediction**: The model is used to predict the customer type for each record in the dataset.

---

## New Customer Prediction

A sidebar form allows users to input new records (Age and Income) to predict the customer type using the trained KNN model. The predicted customer type is saved to the MySQL database.

---

## Displaying Results

1. **Dataset View**: The user can view the original dataset with actual and predicted customer types.
2. **Updated Dataset**: After a prediction is made, the dataset is updated with the new predicted values.
3. **Model Accuracy**: The app displays the accuracy of the model as a success message.

---

## Research Interpretation

1. **Dataset with Actual and Predicted Values**: The dataset shows both actual and predicted customer types. The column `Actual_Customer_Type` refers to the real values, while `Predicted_Customer_Type` contains the model's predictions.
2. **Prediction for New Record**: The model predicts the customer type for new data based on age and income. The prediction for records outside the training data (e.g., Age=100, Income=100,000) may not be reliable.
3. **Model Accuracy**: The accuracy of the model is displayed, showing the percentage of correct predictions made by the model.

---

## Conclusion

This Streamlit application implements a K-Nearest Neighbors model to predict customer types based on age and income. The app provides both analytics for customer types and real-time prediction functionalities, along with the ability to save results in a MySQL database for future analysis.
