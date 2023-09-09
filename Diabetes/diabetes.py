""" 1. Data Preprocessing: Check for missing values in the dataset and handle them appropriately (e.g., by imputation or removing rows with missing values). Standardize or normalize the input features to ensure that they have similar scales."""

import pandas as pd

df             = pd.read_csv("diabetes.csv")
print(" \n")
"""
print("glimpse of the data: \n", df.head(10))						# glimpse of the data
print(" \n")
print("describe parameters like count, mean, std, and max \n", df.describe())		# check count, mean, std, and max
print(" \n")
print("Shape of the dataset \n", df.shape)						# rows and columns
print(" \n")
outcome_counts = df['Outcome'].value_counts()						# Count the occurrences of outcome values
print("Count of data with Outcome 1 (DIABETIC):", outcome_counts[1])			# Display the counts
print("Count of data with Outcome 0 (NON-DIABETIC):", outcome_counts[0])
print(" \n")
"""
rows_to_drop   = df[(df.iloc[:,: -1]==0).any(axis=1)].index										# Rows with missing values
prepro_data    = df.drop(rows_to_drop)													# Removing rows with missing values
prepro_data.to_csv(" datapro_diabetes.csv", index=False)	
"""									# saving new data as csv
print("Rows with 0 entries in any column except for 'Outcome' is removed and saved to 'datapro_diabetes.csv'.\n")
print("glimpse of the data without missing values: \n", prepro_data.head(10))			# glimpse of the data
print(" \n")
outcome_counts = prepro_data['Outcome'].value_counts()						# Count the occurrences of outcome values
print("Count of data with Outcome 1 (DIABETIC):", outcome_counts[1])			# Display the counts
print("Count of data with Outcome 0 (NON-DIABETIC):", outcome_counts[0])
print(" \n")
"""
from sklearn.preprocessing import MinMaxScaler
normalized_data = MinMaxScaler().fit_transform(prepro_data)				# for normalization
normalized_df   = pd.DataFrame(data=normalized_data, columns=prepro_data.columns)
normalized_df.to_csv("normalized_diabetes.csv", index=False)
print("The input features are normalized and saved to 'normalized_diabetes.csv'.\n")
"""print("glimpse of the normalized data: \n", normalized_df.head(10))				# glimpse of the data"""
print(" \n")


"""2. Exploratory Data Analysis (EDA): Explore the distribution of each feature in the dataset using histograms or density plots. Calculate and visualize the correlation matrix of the features using a heatmap to identify potential relationships."""

import matplotlib.pyplot as plt
import seaborn as sns

normalized_df.drop(columns=["Outcome"]).hist(figsize=(10, 8), bins=20)
plt.suptitle("Feature Distribution (Normalized Data)")
plt.show()

correlation_matrix = normalized_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap (Normalized Data)")
plt.show()


"""3. Feature Visualization: Create pair plots (scatter plots) for a subset of features to visualize relationships between pairs of features, possibly using a different color for each class."""

feature_subset = ["Glucose", "BMI", "Age", "Insulin", "Pregnancies", "BloodPressure", "SkinThickness", "DiabetesPedigreeFunction"]
sns.pairplot(data=normalized_df, vars=feature_subset, hue="Outcome", palette="husl")
plt.suptitle("Pair Plots of Selected Features")
plt.show()

"""4. Feature Importance: Train a simple classifier (e.g., Decision Tree or Random Forest) and analyze feature importance to understand which features contribute the most to predicting the target class."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

X = normalized_df.drop(columns=["Outcome"])
y = normalized_df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(random_state=42)   # Decision Tree classifier
decision_tree.fit(X_train, y_train)

feature_importances   = decision_tree.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

print("Feature Importances:")
print(feature_importance_df)
print(" \n")


"""5. Model Selection and Training: Split the dataset into training and testing sets. Choose and train multiple classification algorithms (e.g., Logistic Regression, Decision Tree, Random Forest, Support Vector Machine) and evaluate their performance on the test set."""
"""6. Model Evaluation: Calculate and display classification metrics such as accuracy, precision, recall, F1-score, and confusion matrix for each model. Visualize the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC) for the best-performing model."""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X = normalized_df.iloc[:, :-1]
y = normalized_df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV

# Function to display evaluation metrics
def display_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("Confusion Matrix:\n", cm)
    
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print(" \n")
print("Logistic Regression:")
display_metrics(y_test, y_pred_logreg)

# Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)
print("Decision Tree:")
display_metrics(y_test, y_pred_dtree)

# Random Forest
rforest = RandomForestClassifier(random_state=42)
rforest.fit(X_train, y_train)
y_pred_rforest = rforest.predict(X_test)
print("Random Forest:")
display_metrics(y_test, y_pred_rforest)

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Support Vector Machine:")
display_metrics(y_test, y_pred_svm)


"""best performing model evaluated using cross valoidation score."""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Dictionary to store AUC values for each model
auc_values_dict = {}

# Calculate AUC for Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
auc_values_dict['Logistic Regression'] = roc_auc_logreg

# Calculate AUC for Decision Tree
fpr_dtree, tpr_dtree, _ = roc_curve(y_test, dtree.predict_proba(X_test)[:, 1])
roc_auc_dtree = auc(fpr_dtree, tpr_dtree)
auc_values_dict['Decision Tree'] = roc_auc_dtree

# Calculate AUC for Random Forest
fpr_rforest, tpr_rforest, _ = roc_curve(y_test, rforest.predict_proba(X_test)[:, 1])
roc_auc_rforest = auc(fpr_rforest, tpr_rforest)
auc_values_dict['Random Forest'] = roc_auc_rforest

# Calculate AUC for Support Vector Machine
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm.decision_function(X_test))
roc_auc_svm = auc(fpr_svm, tpr_svm)
auc_values_dict['Support Vector Machine'] = roc_auc_svm

# Find the best-performing model based on AUC
best_model = max(auc_values_dict, key=auc_values_dict.get)
best_auc = auc_values_dict[best_model]

# Visualize the ROC curve for the best-performing model
plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, color='red', lw=2, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
plt.plot(fpr_dtree, tpr_dtree, color='darkorange', linestyle='--', lw=2, marker='o', markevery=0.2, label=f'Decision Tree (AUC = {roc_auc_dtree:.2f})')
plt.plot(fpr_rforest, tpr_rforest, color='blue', linestyle='-.', lw=2, marker='s', markevery=0.2, label=f'Random Forest (AUC = {roc_auc_rforest:.2f})')
plt.plot(fpr_svm, tpr_svm, color='green', linestyle=':', lw=2, marker='*', markevery=0.2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print the best-performing model and its AUC
print("Best-Performing Model:", best_model)
print("AUC for Best Model:", best_auc)



"""7. Hyperparameter Tuning: Select the best model and perform hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV to find optimal hyperparameters"""

# Hyperparameter tuning for Random Forest
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid for Logistic Regression

# Define the hyperparameter grid for Logistic Regression 
param_grid_logreg = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  	# Regularization parameter
    'penalty': ['l2']  				# Regularization penalty ('l2' for L2 regularization/ 'lbfgs' does not support the 'l1' penalty)
}

# Create the Logistic Regression classifier
logreg = LogisticRegression(solver='lbfgs')

# Instantiate GridSearchCV with the Logistic Regression classifier and hyperparameter grid
grid_search_logreg = GridSearchCV(estimator=logreg, param_grid=param_grid_logreg, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search_logreg.fit(X_train, y_train)

# Get the best-performing Logistic Regression model with the optimal hyperparameters
best_logreg_model = grid_search_logreg.best_estimator_

# Evaluate the best model on the test set
y_pred_best_logreg = best_logreg_model.predict(X_test)

# Display evaluation metrics for the best Logistic Regression model
print("Best Logistic Regression Model (Hyperparameter-Tuned):")
display_metrics(y_test, y_pred_best_logreg)

# Get the best hyperparameters for Logistic Regression
best_params_logreg = grid_search_logreg.best_params_
print("Best Hyperparameters for Logistic Regression:")
print(best_params_logreg)


"""8. Final Model Visualization: Visualize the decision boundary of the final model by plotting the data points and using color gradients to indicate class probabilities.""" 
import numpy as np
from itertools import combinations

# Get all possible combinations of 2 features out of the 8 available
feature_combinations = list(combinations(X_train.columns, 2))

# Calculate the number of grids
num_grids = len(feature_combinations) // 4 + 1

# Loop through all feature combinations and create a grid for each 4 combinations
for grid_idx in range(num_grids):
    start_idx = grid_idx * 4
    end_idx = min(start_idx + 4, len(feature_combinations))
    
    # Create a grid with 4 subplots
    num_cols = min(4, end_idx - start_idx)
    
    if num_cols > 0:
        fig, axs = plt.subplots(1, num_cols, figsize=(16, 4))
        fig.subplots_adjust(wspace=0.5)

        # Loop through 4 feature combinations in the grid
        for i in range(start_idx, end_idx):
            feature1, feature2 = feature_combinations[i]

            # Create a mesh grid for the selected pair of features
            feature1_min, feature1_max = X_train[feature1].min() - 1, X_train[feature1].max() + 1
            feature2_min, feature2_max = X_train[feature2].min() - 1, X_train[feature2].max() + 1
            xx, yy = np.meshgrid(np.arange(feature1_min, feature1_max, 0.01), np.arange(feature2_min, feature2_max, 0.01))

            # Fit a new Logistic Regression model with only the selected features
            X_train_selected = X_train[[feature1, feature2]]
            logreg_selected = LogisticRegression()
            logreg_selected.fit(X_train_selected, y_train)

            # Use the trained Logistic Regression model to make predictions on the mesh grid
            Z = logreg_selected.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            # Plot the decision boundary and data points for the selected pair of features
            col = i % num_cols
            axs[col].contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
            axs[col].scatter(X_train[feature1], X_train[feature2], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')
            axs[col].set_xlabel(feature1)
            axs[col].set_ylabel(feature2)
            axs[col].set_title(f'({feature1} vs. {feature2})')

plt.show()


"""9. Model Interpretability: Use techniques like SHAP (SHapley Additive exPlanations) or feature importance plots to explain individual predictions made by the model."""

import matplotlib.pyplot as plt
import seaborn as sns

# Get the number of entries in X_test
num_entries = X_test.shape[0]
print("Number of entries in X_test:", num_entries)

# Calculate feature importances for the best Logistic Regression model
feature_importances = best_logreg_model.coef_[0]

# Function to explain an individual prediction
def explain_individual_prediction(model, feature_importances, data_point, feature_names):
    # Calculate the importance scores for each feature
    importance_scores = feature_importances * data_point
    
    # Create a DataFrame to store the importance scores
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
    plt.title("Feature Importance for Individual Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

# Loop through individual_prediction_index from 0 to 8
for individual_prediction_index in range(9):
    # Get the individual data point from X_test
    individual_data_point = X_test.iloc[individual_prediction_index]

    # Explain the individual prediction
    print(f"Explaining prediction at index {individual_prediction_index}:")
    explain_individual_prediction(best_logreg_model, feature_importances, individual_data_point, X_train.columns)


"""10. Discussion and Conclusion: Summarize the results obtained from different models and discuss the trade-offs between precision and recall. Conclude by highlighting the importance of feature engineering, data preprocessing, and model selection in building a successful machine learning pipeline."""






