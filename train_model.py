import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load the dataset
df = pd.read_csv('diabetes4.csv')

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights for imbalance handling
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: w for i, w in zip(np.unique(y), class_weights)}

# Define hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize and tune the model
base_model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"✅ Accuracy: {round(accuracy * 100, 2)}%")
print(f"✅ Precision: {round(precision * 100, 2)}%")
print(f"✅ Recall: {round(recall * 100, 2)}%")
print("✅ Confusion Matrix:")
print(conf_matrix)

# Save the best model
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Tuned model trained and saved as diabetes_model.pkl!")
