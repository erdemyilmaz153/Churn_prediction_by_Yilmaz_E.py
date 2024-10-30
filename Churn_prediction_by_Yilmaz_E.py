# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import chi2_contingency

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

# Renaming the columns 'gender' to 'Gender' and 'tenure' to 'Tenure'
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
    
There seems to be no unusual entries in first and last observations except SeniorCitizen column needs to be categorical
and TotalCharges numerical.

There is no missing data.
'''


########## Define numerical and categorical variables ##########

# Get categorical columns (usually 'object' or 'category' types)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns:", categorical_columns)

# Get numerical columns (usually 'int64' or 'float64' types)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numerical columns:", numerical_columns)


########## A quick look ##########

# Since Customer ID is irrelevant in this context, we can directly drop it.
df.drop('customerID', axis=1, inplace=True)

# Converting 'SeniorCitizen' to categorical
df['SeniorCitizen'] = df['SeniorCitizen'].apply(lambda x: 'Non-Senior' if x == 0 else 'Senior')

# Drop missing values of 'TotalCharges'
df = df.dropna(subset=['TotalCharges'])


######################################## Examining Numerical Variables #############################################

# Replace any non-numeric values (like empty strings) with NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convert the column to float64 if not already
df['TotalCharges'] = df['TotalCharges'].astype('float64')
print(df['TotalCharges'].dtype)

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Histogram Graphs for Numerical Variables #
print("Numerical columns:", numerical_columns)

for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=10, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(f'{col}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

'''
TotalCharges seems to have skewness.
'''

# Calculate Skewness of the Numerical Variables #
skewness_values = df[numerical_columns].skew()

# Display the skewness values
skewness_values

'''
After double-check the skewness of 'TotalCharges' it is safe to say it is skewed with the value of 0.962.
'''

# Solve the issue of skewness

# Apply square root transformation
df['TotalCharges'] = np.sqrt(df['TotalCharges'])

# Check the result
print(df['TotalCharges'].head())


# Creating a new column from 'MonthlyCharges' indicating whether charges are lower or higher than 30
df['MonthlyCharges_category'] = df['MonthlyCharges'].apply(lambda x: 'Lower than 30' if x < 30 else 'Higher than 30')

# Creating a new column from 'Tenure' with three categories: lower than 8, between 9 and 65, and higher than 65
df['Tenure_category'] = (df['Tenure'].apply
                         (lambda x: 'Lower than 9' if x < 9 else ('Middle (8-65)' if x <= 65 else 'Higher than 65')))

# Display the result
df[['MonthlyCharges', 'MonthlyCharges_category', 'Tenure', 'Tenure_category']].head(22)


##### Correlation Heatmap for the Numerical Variables #####
correlation_matrix = df[numerical_columns].corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# Display the correlation matrix
correlation_matrix

'''
'TotalCharges' and 'Tenure' has really high positive correlation with 0.826
Additionally, 'MonthlyCharges' and 'TotalCharges' has high positive correlation with 0.651.
Lastly, 'Tenure' and 'Monthly' charges positively correlated with the value of 0.248.

'''

##### Box-and-Whisker plot for the Numerical Columns #####
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[column])
    plt.title(f'Box-and-Whisker Plot for {column}')
    plt.show()
'''
There is no outliers for numerical variables.
'''


##################################### Examining Categorical Variables ############################################

##### Plot Cramér's V to see relation between categorical variables #####

categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Function to calculate Cramér's V
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

# Function to create Cramér's V matrix for a DataFrame of categorical columns
def cramers_v_matrix(df, categorical_columns):
    matrix = pd.DataFrame(np.zeros((len(categorical_columns), len(categorical_columns))),
                          columns=categorical_columns, index=categorical_columns)

    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 != col2:
                confusion_matrix = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(confusion_matrix)
            else:
                matrix.loc[col1, col2] = 1  # Same columns will have perfect association
    return matrix

# Calculate the Cramér's V matrix
cramers_v_mat = cramers_v_matrix(df, categorical_columns)

# Visualize the Cramér's V matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cramers_v_mat, annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Cramér's V Matrix for Categorical Columns")
plt.show()

'''
There are significant relationships between 'InternetService' and these: 'OnlineSecurity', 'OnlineBackup', 
'DeviceProtection', 'TechSupport', 'StreamingTV', and 'StreamingMovies' around 0.72. Additionally, it is correlated
with 'MultipleLines' and 'PhoneService' around 0.42.

There is almost no relation with target 'Churn' and these: 'Gender', 'PhoneService', and 'MultipleLines'. However,
others are relatively highly correlated and accumulated around 0.30.

Gender seems has nothing to do with other variables as well. It can directly be dropped but it is kept since it can
be useful for feature engineering.

'Dependents' and 'Partner' are correlated with 0.45
'''

##### Distributions of Categorical Variables on the Target #####

# Loop through each categorical column and create a report
for column in categorical_columns:

    if column != 'Churn':

        # Cross-tabulation between the categorical variable and Churn (raw counts)
        crosstab = pd.crosstab(df[column], df['Churn'])

        # Cross-tabulation for percentages
        crosstab_pct = pd.crosstab(df[column], df['Churn'], normalize='index') * 100

        # Combine raw counts and percentages
        combined = crosstab.copy()

        for col in crosstab.columns:
            combined[col] = combined[col].astype(str) + " (" + crosstab_pct[col].round(2).astype(str) + "%)"

        print(f"Distribution of {column} with respect to 'Churn':")
        print(combined)
        print("\n" + "=" * 60 + "\n")


# Plot the distribution
plt.figure(figsize=(8, 6))
# Loop through each categorical column to create visualizations
for column in categorical_columns:

    if column != 'Churn':
        # Cross-tabulation for each categorical column vs 'Churn'
        crosstab = pd.crosstab(df[column], df['Churn'], normalize='index') * 100

        # Print the cross-tabulation
        print(f"Churn Distribution by {column}:")
        print(crosstab)
        print("\n" + "=" * 40 + "\n")


        crosstab.plot(kind='bar', stacked=True)
        plt.title(f"Churn Distribution by {column}")
        plt.ylabel("Percentage")
        plt.xlabel(column)
        plt.xticks(rotation=0)
        plt.legend(title="Churn", loc='upper right')
        plt.show()


'''
'ElectronicCheck' has high negative impact on the target than other 'PaymentMethod's. others almost euqal
Those who pay as 'PaperlessBilling' are more prone to leave.
In 'Contract', people leave less when the time of contract increases
Those who has no internet service while 'StreamingMovies' are less prone to leave than others. StreamingTV same
'TechSupport', no > yes > noI
'DeviceProtection', no > yes > noI
'OnlineBackup', no > yes > noI
'OnlineSecurity', no > yes > noI
'InternetService', fiber optic > DLS > no
'MultipleLines', no = yes = no phone service
'PhoneService', yes = no
'Dependents', yes > no
'Partner', no > yes
'SenirCitizen'> Senior > Non-senior
'''


# List of potentially redundant columns
redundant_columns = [
    'TechSupport', 'DeviceProtection', 'OnlineBackup', 'OnlineSecurity', 'StreamingMovies', 'TechSupport']


# Function to calculate Cramér's V between two variables

# Create a Cramér's V matrix for these variables
redundancy_matrix = pd.DataFrame(np.zeros((len(redundant_columns), len(redundant_columns))),
                                 columns=redundant_columns, index=redundant_columns)

for col1 in redundant_columns:
    for col2 in redundant_columns:
        if col1 != col2:
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            redundancy_matrix.loc[col1, col2] = cramers_v(confusion_matrix)
        else:
            redundancy_matrix.loc[col1, col2] = 1  # perfect correlation with itself

# Visualize the Cramér's V matrix to check for high correlations
plt.figure(figsize=(8, 6))
sns.heatmap(redundancy_matrix, annot=True, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Cramér's V Matrix for Redundant Features")
plt.show()


# Dropping the specified columns from the DataFrame
df_dropped = df.drop([
    'OnlineSecurity', 'OnlineBackup', 'StreamingTV', 'DeviceProtection', 'StreamingMovies', 'TechSupport'], axis=1)


###################################################################################################################

# Define numerical and categorical variables
# Get categorical columns (usually 'object' or 'category' types)
categorical_columns = df_dropped.select_dtypes(include=['object', 'category']).columns.tolist()

# Get numerical columns (usually 'int64' or 'float64' types)
numerical_columns = df_dropped.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print the results
print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

####################################################################################################################

df_dropped.head()
# Assuming you have a list of categorical columns
# Exclude 'Churn' from the list
categorical_columns = [col for col in categorical_columns if col != 'Churn']

# Apply one-hot encoding to the remaining categorical columns
df_encoded = pd.get_dummies(df_dropped, columns=categorical_columns, drop_first=True)

# Display the first few rows of the updated DataFrame
print(df_encoded.head())

df_encoded = df_encoded.dropna(subset=['TotalCharges'])


#####################################################################################################################
###################################### Training Several Models ######################################################
#####################################################################################################################

######################################## Logistic Regression ########################################

df_encoded['Churn'] = df_encoded['Churn'].str.strip().map({'No': 0, 'Yes': 1})

df_encoded = df_encoded.copy()

# Step 1: Splitting the data into features and target
X = df_encoded.drop('Churn', axis=1)  # Features (drop the target 'Churn')
y = df_encoded['Churn']  # Target

# Step 2: Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Step 3: Scaling the numerical features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train a classification model (Logistic Regression)
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train_scaled, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

########## 0.8010, f1=0.54


########################################## Random Forest ########################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Step 1: Define the Random Forest model
rf = RandomForestClassifier(random_state=18)

# Step 2: Define the hyperparameter grid for tuning
rf_param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Step 3: RandomizedSearchCV for hyperparameter tuning with 10-fold cross-validation
rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_grid,
    n_iter=50,
    cv=10,  # Changed cv from 3 to 10
    verbose=2,
    random_state=18,
    n_jobs=-1
)

# Step 4: Fit the model on the training data
rf_random_search.fit(X_train_scaled, y_train)

# Step 5: Retrieve the best model
rf_best_model = rf_random_search.best_estimator_

# Step 6: Evaluate the best model using 10-fold cross-validation on the training data
cv_scores = cross_val_score(
    rf_best_model,
    X_train_scaled,
    y_train,
    cv=10,
    scoring='accuracy',
    n_jobs=-1
)

print("Cross-validation Accuracy Scores:", cv_scores)
print(f"Mean Cross-validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Step 7: Predict and evaluate the model on the test data
y_pred_rf = rf_best_model.predict(X_test_scaled)

# Step 8: Evaluate the model on the test data
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Set Accuracy: {accuracy_rf:.4f}")
print("Random Forest Classification Report on Test Set:")
print(classification_report(y_test, y_pred_rf))

# Accuracy: 0.8038
# f1 score: 0.58

from sklearn.metrics import roc_auc_score

# After Step 5: Make predictions on the test set
# Get the predicted probabilities for the positive class
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Compute ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Output the results
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print(f"ROC AUC Score: {roc_auc:.4f}")

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# After obtaining y_pred_proba
# Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Compute ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


######################################## Gradient Boosting ##################################################

from sklearn.ensemble import GradientBoostingClassifier

# Step 1: Define the Gradient Boosting model
gb = GradientBoostingClassifier(random_state=18)

# Step 2: Define the hyperparameter grid for tuning
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.13, 0,16, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Step 3: RandomizedSearchCV for hyperparameter tuning
gb_random_search = RandomizedSearchCV(estimator=gb, param_distributions=gb_param_grid,
                                      n_iter=50, cv=10, verbose=2, random_state=18, n_jobs=-1)

# Step 4: Fit the model on the training data
gb_random_search.fit(X_train_scaled, y_train)

# Step 5: Predict and evaluate the model on the test data
gb_best_model = gb_random_search.best_estimator_
y_pred_gb = gb_best_model.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb:.4f}")
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gb))

#### 0.8081, f1=0.59

# Step 1: Extract feature importance from the best Random Forest model
feature_importance = rf_best_model.feature_importances_

# Step 2: Create a DataFrame to hold feature names and their corresponding importance
features = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
})

# Step 3: Sort the DataFrame by importance in descending order
features_sorted = features.sort_values(by='Importance', ascending=False)

# Step 4: Visualize feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=features_sorted)
plt.title('Feature Importance in Random Forest Model')
plt.show()


############################################### XGBoost #####################################################

from xgboost import XGBClassifier

# Check if the conversion was successful
print(df_encoded['Churn'].unique())
df_encoded.head()
# Proceed with model training

# Step 1: Define the XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=18)

# Step 2: Define the hyperparameter grid for tuning
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.03, 0.6, 0.1, 0.13, 0.16, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Step 3: RandomizedSearchCV for hyperparameter tuning
xgb_random_search = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_grid,
                                       n_iter=50, cv=3, verbose=2, random_state=18, n_jobs=-1)

# Step 4: Fit the model on the training data
xgb_random_search.fit(X_train_scaled, y_train)

# Step 5: Predict and evaluate the model on the test data
xgb_best_model = xgb_random_search.best_estimator_
y_pred_xgb = xgb_best_model.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Accuracy: 0.7960
# f1 score: 0.55


####################################### Neural Networks #################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Define the Sequential model
model = Sequential()

# Input Layer and First Hidden Layer
model.add(Dense(units=32, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Second Hidden Layer
model.add(Dense(units=8, activation='relu'))

# Optionally, add Dropout to prevent overfitting (if necessary)
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(units=1, activation='sigmoid'))

# Step 2: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Add Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Step 4: Train the model
history = model.fit(X_train_scaled, y_train,
                    epochs=100, batch_size=64,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=[early_stopping])

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_class = (y_pred > 0.5).astype("int32")

# 0.8060


#################################### SMOTE with XGB ################################################

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Step 1: Split the data into features and target
X = df_encoded.drop('Churn', axis=1)  # Features
y = df_encoded['Churn']  # Target (binary: 0 or 1)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

# Step 3: Apply SMOTE to the training data
smote = SMOTE(random_state=18)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 4: Train the XGBoost model on the SMOTE-applied data
xgb_model = XGBClassifier(random_state=18)
xgb_model.fit(X_train_smote, y_train_smote)

# Step 5: Make predictions on the test data
y_pred = xgb_model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy after SMOTE: {accuracy:.4f}")
print("Classification Report after SMOTE:")
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

# Before SMOTE
sns.countplot(x=y_train)
plt.title("Class Distribution Before SMOTE")
plt.show()

# After SMOTE
sns.countplot(x=y_train_smote)
plt.title("Class Distribution After SMOTE")
plt.show()

# Accuracy: 0.7633
# f1 socre: 0.57