# KNN ALGORITHM AND PREDICTION WITH MYSQL

## Live Demo
You can view the live demo of this project at:  
[**K-Nearest Neighbors Prediction & Trends**](https://k-means-algorithm.streamlit.app/)

## Technologies

| Technology     | Description                                                                                          |
|----------------|------------------------------------------------------------------------------------------------------|
| **Python**     | The core programming language used for implementing the algorithm and handling the data.            |
| **Streamlit**  | A framework used to create an interactive web application for visualizing the model's predictions and analytics. |
| **SQL Statements** | Used to interact with the MySQL database to read and write customer data.                           |
| **MySQL**      | A relational database management system used to store customer data for training and predictions.     |

## Applied Domains

| Role                                | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| **Statisticians**                   | Professionals who apply statistical techniques to collect, analyze, and interpret data for various purposes. |
| **Healthcare and Medicine**         | Medical professionals and researchers who use data analysis to improve patient care and medical research. |
| **Finance and Economics**           | Experts who use data and statistical models to make financial and economic decisions. |
| **Marketing and Customer Analytics**| Professionals who analyze consumer data to optimize marketing strategies and enhance customer experiences. |
| **Environmental Scientist**         | Scientists who apply data analysis, modeling, and statistical methods to study environmental changes, conservation efforts, and sustainability initiatives. |
| **Data Scientist**                  | Specialists in analyzing and interpreting complex data to provide actionable insights across various domains. |
| **Operational Research**            | Analysts who apply advanced mathematical and statistical models to help organizations solve complex problems. |

## Main Objective

The main objective of this project is to implement the **K-Nearest Neighbors (KNN)** algorithm to segment customers into distinct tiers (VIP, Regular, Gold, Premium) based on their predicted behavior. This segmentation will enable targeted service delivery, such as SMS notifications, tailored to each tier.

### Key Goals:
1. **Data Training**: The model will be trained using historical customer data (e.g., purchase history, website activity, engagement metrics) stored in a **MySQL database**. This data will be used to identify patterns and predict future customer behavior.
2. **Prediction**: The trained KNN model will then be used to classify new and existing customers into the appropriate tiers (VIP, Regular, Gold, Premium). This classification will facilitate personalized services and communication strategies.

___

### Simple Analytics and Tier-Specific Insights

- **Customer Tier Analysis**: The project provides an analysis of the customer dataset, focusing on key metrics within each tier:
    - **Tier Distribution**: The number of customers assigned to each tier (VIP, Regular, Gold, Premium), providing an overview of the customer base composition.
    - **Behavioral Insights**: Analysis of key behavioral metrics (e.g., average purchase value, frequency of purchases, website visits) within each tier. This helps understand the characteristics of each customer segment.
    - **Demographic Insights (If Available):** If demographic data (e.g., age, location) is available, analysis of these features within each tier can provide further insights into customer demographics.

- **Overall Customer Insights**: Basic statistics about the entire dataset, including the total number of unique customers and overall distribution of behavioral metrics.

These analytics provide valuable insights into customer segmentation, enabling targeted service delivery and personalized experiences based on predicted customer behavior and tier assignment.

---

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

## Interpretation

1. **Dataset with Actual and Predicted Values**: The dataset shows both actual and predicted customer types. The column `Actual_Customer_Type` refers to the real values, while `Predicted_Customer_Type` contains the model's predictions.
2. **Prediction for New Record**: The model predicts the customer type for new data based on age and income. The prediction for records outside the training data (e.g., Age=100, Income=100,000) may not be reliable.
3. **Model Accuracy**: The accuracy of the model is displayed, showing the percentage of correct predictions made by the model.

---

## Conclusion

This Streamlit application implements a K-Nearest Neighbors model to predict customer types based on age and income. The app provides both analytics for customer types and real-time prediction functionalities, along with the ability to save results in a MySQL database for future analysis.

**My Contacts**

**WhatsApp**  
+255675839840  
+255656848274

**YouTube**  
[Visit my YouTube Channel](https://www.youtube.com/channel/UCjepDdFYKzVHFiOhsiVVffQ)

**Telegram**  
+255656848274  
+255738144353

**PlayStore**  
[Visit my PlayStore Developer Page](https://play.google.com/store/apps/dev?id=7334720987169992827&hl=en_US&pli=1)

**GitHub**  
[Visit my GitHub](https://github.com/shamiraty/)
