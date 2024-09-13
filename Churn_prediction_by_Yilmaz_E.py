# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

matplotlib.use('QT5Agg')
from matplotlib import pyplot
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

# To see all columns at once
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

#####################################################################################################################
###################################### Load and Examine the Dataset #################################################
#####################################################################################################################

df = pd.read_csv('C://Users//erdem//OneDrive//Masaüstü//github_Projects//Churn_prediction_by_Yilmaz_E//'
                 'Telco_customer_churn.csv')

def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

# Renaming the columns 'Gender' to 'Gender' and 'Tenure' to 'Tenure'
df.rename(columns={'gender': 'Gender', 'tenure': 'Tenure'}, inplace=True)


'''
1. customerID: Customer ID

2. Gender: Whether the customer is a male or a female

3. SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)

4. Partner: Whether the customer has a partner or not (Yes, No)

5. Dependents: Whether the customer has dependents or not (Yes, No)

6. Tenure: Number of months the customer has stayed with the company

7. PhoneService: Whether the customer has a phone service or not (Yes, No)

8. MultipleLines: Whether the customer has multiple lines or not (Yes, No, Phone service)

9. InternetService: Customer's internet service provider (DSL, Fiber optic, No)

10. OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)

11. OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)

12. DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)

13. TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)

14. StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)

15. StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)

16. Contract: The contract term of the customer (Month-to-month, One  year, Two year)

17. PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)

18. PaymentMethod: The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), 
    Credit Card)
    
19. MonthlyCharges: The amount charged to customer monthly

20. TotalCharges: The total amount charged to the customer

21. Churn: Whether customer churned or not (Yes or No)
'''

check_data(df)
df.head(30)
df.tail(30)
'''
Shape of the data set is: (7043, 21).

Types of the variables are object(18), int64(2), float(1) but SeniorCitizen needs to be converted categorical while
TotalCharges needs to be float64.
    
There seems to be no unusual entries in first and last observations.

There is no missing data.
'''


####################################################################################################################

# Since Customer ID is irrelevant in this context, we directly drop it.
df.drop('customerID', axis=1, inplace=True)

# Assuming 'SeniorCitizen' is already in your DataFrame
df['SeniorCitizenCategory'] = df['SeniorCitizen'].apply(lambda x: 'Non-Senior' if x == 0 else 'Senior')

# Check the transformation
print(df[['SeniorCitizen', 'SeniorCitizenCategory']].head())

# Drop the variable
df.drop('SeniorCitizen', axis=1, inplace=True)

df.head()


######################################## Examining Numerical Variables #############################################

# Fill missing values in 'TotalCharges' with the mean of the column
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Verify if there are any more missing values in 'TotalCharges'
print(df['TotalCharges'].isnull().sum())


# Replace any non-numeric values (like empty strings) with NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convert the column to float64 if not already
df['TotalCharges'] = df['TotalCharges'].astype('float64')

# Check the transformation
print(df['TotalCharges'].dtype)


####################################################################################################################

# First, analyze numerical columns with visualizations.
# Check for missing or non-numeric values in numerical columns
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing values in numerical columns for better visualization
cleaned_data = df.dropna(subset=['Tenure', 'MonthlyCharges', 'TotalCharges'])

# Plot histograms for each numerical column
numerical_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges']

plt.figure(figsize=(12, 8))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 1, i)
    plt.hist(cleaned_data[column], bins=30, edgecolor='k')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

'''
It seems like TotalCharges skewed to the right so logarithmic transformation is going to be applied.
Tenure and Monthly charges can be categorized since there are accumulations in several points.
'''


####################################################################################################################

# More visualization for numerical values.
# Create additional visualizations for Tenure, MonthlyCharges, and TotalCharges

plt.figure(figsize=(15, 12))

# Boxplots for each numerical column
plt.subplot(3, 2, 1)
plt.boxplot(cleaned_data['Tenure'])
plt.title('Boxplot of Tenure')
plt.ylabel('Tenure (months)')

plt.subplot(3, 2, 3)
plt.boxplot(cleaned_data['MonthlyCharges'])
plt.title('Boxplot of Monthly Charges')
plt.ylabel('Monthly Charges')

plt.subplot(3, 2, 5)
plt.boxplot(cleaned_data['TotalCharges'])
plt.title('Boxplot of Total Charges')
plt.ylabel('Total Charges')

# KDE plots for each numerical column
plt.subplot(3, 2, 2)
cleaned_data['Tenure'].plot(kind='kde', title='KDE of Tenure')
plt.xlabel('Tenure (months)')

plt.subplot(3, 2, 4)
cleaned_data['MonthlyCharges'].plot(kind='kde', title='KDE of Monthly Charges')
plt.xlabel('Monthly Charges')

plt.subplot(3, 2, 6)
cleaned_data['TotalCharges'].plot(kind='kde', title='KDE of Total Charges')
plt.xlabel('Total Charges')

plt.tight_layout()
plt.show()

'''
There are no outliers for all 3 numerical variables.
In KDE for each, we can see distributions in more detail and skewness became more clear for TotalCharges.
'''


####################################################################################################################

# Select only the numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numerical variables
corr_matrix = numerical_df.corr()

# Plot the correlation matrix using seaborn's heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Variables')
plt.show()

'''
Tenure-MonthlyCharges has correlation as .25
Tenure-TotalCharges has correlation as .83
TotalCharges-MonthlyCharges has correlation as .65
There seems genuinely high correlations for Tenure-TotalCharges, and also TotalCharges-MonthlyCharges 
'''

######################################## Examining Categorical Variables ###########################################

# Select the categorical columns (those with data type 'object' or 'category')
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Print the categorical columns
print("Categorical Columns:", categorical_cols)


####################################################################################################################

# Loop through categorical columns and plot
for col in categorical_cols:
    # Get the value counts for the categorical column
    value_counts = df[col].value_counts()

    # Plot using matplotlib
    plt.figure(figsize=(8, 4))
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x labels if needed
    plt.show()


# Cross Tabulation with the Target Variables ########################################################################

for col in categorical_cols:

    # Optionally, normalize the values to get percentages
    cross_tab_churn_normalized = pd.crosstab(df[col], df['Churn'], normalize='index')

    # Display the normalized cross-tabulation
    print(cross_tab_churn_normalized)
    print('\n*************************************')


# Grouping and Aggregating #########################################################################################

for i in categorical_cols:

    for j in numerical_columns:

        # Grouping by a categorical column and calculating the mean of a numerical column
        grouped_data = df.groupby(i)[j].mean()

        # Display the grouped data
        print(grouped_data)
        print('\n*************************************')


cross_tab = pd.crosstab(df['Contract'], df['PaymentMethod'])
# Visualizing the cross-tabulation between Contract and PaymentMethod
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='d')
plt.title('Cross-tabulation of Contract vs PaymentMethod')
plt.show()

'''
Not feasible but desired relation can be found by searching here.
'''


# Chi-Square Test for Independence ##################################################################################

for i in categorical_cols:

    for j in categorical_cols:

        if i != j:
            # Create a contingency table
            contingency_table = pd.crosstab(df[i], df[j])

            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Check p-value to interpret the result
            print(f"Chi-square test result: p-value = {p}")
            print('\n*************************************')


df.head()


#####################################################################################################################
###################################### Feature Engineering ##########################################################
#####################################################################################################################

df.head()


#####################################################################################################################
####################################### Encoding ####################################################################
#####################################################################################################################
'''
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
# Convert the column to float64 if not already
df['Churn'] = df['Churn'].astype('float64')
df.head()'''

# Apply one-hot encoding to all categorical variables including 'Churn'

df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

# Check the transformed DataFrame
print(df_encoded.head())
df = df_encoded
df['Churn']

#####################################################################################################################
###################################### Training the Model ###########################################################
#####################################################################################################################


# Convert 'Churn' to binary if not already (ensure it's encoded as 0 and 1)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Apply one-hot encoding to all categorical variables
df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Re-identify numerical columns after encoding
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling only to numerical variables
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Train the model on the training data
logreg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)











