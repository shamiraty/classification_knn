import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from numerize import numerize

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("K-NEIGHBORS ALGORITHM: PREDICTION & TRENDS WITH MYSQL")
st.warning("Predicting Future Customer Behaviors Starts Here")
# Main objective
st.subheader("Main Objective")
st.markdown("""
- **The main objective of this project** is to implement the K-Nearest Neighbors (KNN) algorithm to segment customers into distinct tiers (VIP, Regular, Gold, Premium) based on their predicted behavior. This segmentation will enable targeted service delivery, such as SMS notifications, tailored to each tier.
- The model will be trained using historical customer data (e.g., purchase history, website activity, engagement metrics) stored in a MySQL database. This data will be used to identify patterns and predict future customer behavior.
- The trained KNN model will then be used to classify new and existing customers into the appropriate tiers (VIP, Regular, Gold, Premium). This classification will facilitate personalized services and communication strategies.

___
### Simple Analytics and Tier-Specific Insights

- **Customer Tier Analysis**: The project provides an analysis of the customer dataset, focusing on key metrics within each tier:
    - **Tier Distribution**: The number of customers assigned to each tier (VIP, Regular, Gold, Premium), providing an overview of the customer base composition.
    - **Behavioral Insights**: Analysis of key behavioral metrics (e.g., average purchase value, frequency of purchases, website visits) within each tier. This helps understand the characteristics of each customer segment.
    - **Demographic Insights (If Available):** If demographic data (e.g., age, location) is available, analysis of these features within each tier can provide further insights into customer demographics.

- **Overall Customer Insights**: Basic statistics about the entire dataset, including the total number of unique customers and overall distribution of behavioral metrics.

These analytics provide valuable insights into customer segmentation, enabling targeted service delivery and personalized experiences based on predicted customer behavior and tier assignment.
""")


# Load CSV file
@st.cache_data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

# Read the data from the CSV file
csv_file_path = "large_customer_data.csv"  # Replace with the path to your CSV file
df = load_csv_data(csv_file_path)

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

        # Show success message
        st.success(f"The predicted Customer Type for Age={age} and Income={income} is: {new_prediction[0]}")

        # Append the new prediction to the dataset
        new_entry = {"Age": age, "Income": income, "Customer_Type": new_prediction[0]}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

# Display the main dataset with Actual and Predicted Values at the top of the page in an expander
with st.expander("Dataset with Actual and Predicted Values"):
    show_data = st.multiselect('Filter: ', df.columns, default=df.columns.tolist(), key="main_table_filter")
    st.dataframe(df[show_data], use_container_width=True)

# Display Model Accuracy
st.subheader("Model Accuracy")
st.success(f"Model accuracy: {accuracy:.2f}")

# Interpretation of Results (simplified)
st.subheader("Research Interpretation")
st.success("1. **Dataset with Actual and Predicted Values**: The dataset shows both actual and predicted customer types. The column 'Customer_Type' refers to the real values, while 'Predicted_Customer_Type' contains the model's predictions.")
st.success("2. **Prediction for New Record**: The model predicts the customer type for Age=100 and Income=100,000. This prediction might not be reliable as the input is far outside the data used for training.")
st.success(f"3. **Model Accuracy**: The model accuracy is {accuracy:.2f}, meaning it correctly predicted {accuracy * 100}% of the test data. A higher accuracy indicates a better model.")

# Display tables with updated dataset after prediction in expander
with st.expander("View Updated Dataset with Predicted Values"):
    show_updated_data = st.multiselect('Filter: ', df.columns, default=df.columns.tolist(), key="updated_table_filter")
    st.dataframe(df[show_updated_data], use_container_width=True)
