import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from numerize import numerize

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("KNN ALGORITHM AND PREDICTION WITH MYSQL")

# Main objective
st.subheader("Main Objective")
st.markdown("""
- **The main objective of this project** is to implement the K-Nearest Neighbors (KNN) algorithm to predict future customer types based on features like Age and Income.
- The model will be trained using data stored in a MySQL database, which is collected from observations conducted in research. The data includes customer details like Age, Income, and Customer Type.
- Once trained, the model will allow predictions to be made on new customer records, and the predicted customer type will be saved back into the MySQL database for future reference and analysis.

___
### Simple Analytics

- **Customer Type Analysis**: The project also provides an analysis of the customer dataset, focusing on key metrics such as:
    - **Customer Type Count**: The total count of customers in each customer type category, showing the distribution of customer types.
    - **Age Insights**: Key statistics on age, including the maximum age, minimum age, and age range within each customer type, along with the standard deviation (age variation) to understand how much the ages vary.
    - **Overall Customer Insights**: Basic statistics about the entire dataset, including the total number of unique customers, the maximum and minimum ages, and the overall age range.

These analytics provide valuable insights into the dataset, helping to interpret the distribution of customer types and better understand the underlying patterns based on age and income.
""")

# MySQL connection setup
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="knn"
    )

# Read data from MySQL database
def read_data_from_database():
    connection = get_mysql_connection()
    query = "SELECT Age, Income, Customer_Type FROM large_customer_data"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# Save new prediction to the database
def save_prediction_to_database(age, income, predicted_value):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = "INSERT INTO large_customer_data (Age, Income, Customer_Type) VALUES (%s, %s, %s)"
    cursor.execute(query, (age, income, predicted_value))
    connection.commit()
    connection.close()

# Read the data from the database
df = read_data_from_database()



# Simple Analytics
st.subheader("Customer Analytics")

# Group by customer type and calculate the requested metrics
analytics_df = df.groupby('Customer_Type').agg(
    customer_count=('Customer_Type', 'count'),
    max_age=('Age', 'max'),
    min_age=('Age', 'min'),
    age_range=('Age', lambda x: x.max() - x.min()),
    age_variation=('Age', 'std')
).reset_index()

# Display the metrics using columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Customer Categories", df["Customer_Type"].nunique())

with col2:
    st.metric("Max Age", df["Age"].max())

with col3:
    st.metric("Min Age", df["Age"].min())

with col4:
    st.metric("Age Range", df["Age"].max() - df["Age"].min())

with col5:
    st.metric("Total Records", numerize.numerize(df.shape[0]))

# Display analytics in a table
st.subheader("Customer Type Analytics by Category")
st.dataframe(analytics_df, use_container_width=True)




# Features and Target
X = df[["Age", "Income"]]
y = df["Customer_Type"]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict for all rows in the dataset
df["Predicted_Customer_Type"] = knn.predict(X)

# Model accuracy
accuracy = accuracy_score(y_test, knn.predict(X_test))

# Sidebar for new record input and prediction
with st.sidebar.form("prediction_form"):
    st.subheader("Enter New Record for Prediction")
    age = st.number_input("Age", min_value=0)
    income = st.number_input("Income", min_value=0)
    submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Predict for the new record
        new_record = [[age, income]]
        new_prediction = knn.predict(new_record)

        # Save to database
        save_prediction_to_database(age, income, new_prediction[0])

        # Show success message
        st.success(f"The predicted Customer Type for Age={age} and Income={income} is: {new_prediction[0]}")

        # Display the updated dataset in the sidebar
        updated_df = read_data_from_database()
        updated_df.rename(columns={"Customer_Type": "Actual_Customer_Type"}, inplace=True)
        updated_df["Predicted_Customer_Type"] = knn.predict(updated_df[["Age", "Income"]])

        # Save the updated dataframe in the session state so it's available globally
        st.session_state.updated_df = updated_df

# Use the updated dataframe from the session state if available
if "updated_df" in st.session_state:
    updated_df = st.session_state.updated_df
else:
    updated_df = df  # Use the original dataframe if no prediction was made yet

# Display the main dataset with Actual and Predicted Values at the top of the page in an expander
with st.expander("Dataset with Actual and Predicted Values"):
    show_data = st.multiselect('Filter: ', df.columns, default=df.columns.tolist(), key="main_table_filter")
    st.dataframe(df[show_data], use_container_width=True)

# Display Model Accuracy
st.subheader("Model Accuracy")
st.success(f"Model accuracy: {accuracy:.2f}")

# Interpretation of Results (simplified)
st.subheader("Research Interpretation")
st.success("1. **Dataset with Actual and Predicted Values**: The dataset shows both actual and predicted customer types. The column 'Actual_Customer_Type' refers to the real values, while 'Predicted_Customer_Type' contains the model's predictions.")
st.success("2. **Prediction for New Record**: The model predicts the customer type for Age=100 and Income=100,000. This prediction might not be reliable as the input is far outside the data used for training.")
st.success(f"3. **Model Accuracy**: The model accuracy is {accuracy:.2f}, meaning it correctly predicted {accuracy * 100}% of the test data. A higher accuracy indicates a better model.")

# Display tables with updated dataset after prediction in expander
with st.expander("View Updated Dataset with Predicted Values"):
    show_updated_data = st.multiselect('Filter: ', updated_df.columns, default=updated_df.columns.tolist(), key="updated_table_filter")
    st.dataframe(updated_df[show_updated_data], use_container_width=True)






