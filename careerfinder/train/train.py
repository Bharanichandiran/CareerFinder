import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import pickle
# Load the data
data = pd.read_csv('bot_data.csv', delimiter=',')

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Drop rows with missing 'user_input'
data.dropna(subset=['user_input'], inplace=True)

# Verify there are no missing values in 'user_input'
if data['user_input'].isnull().any():
    print("There are still missing values in 'user_input'.")
else:
    print("No missing values in 'user_input'.")

# Class distribution check
print("Initial class distribution:")
print(data['intent'].value_counts())

# Optionally, remove classes with fewer than a minimum threshold of samples
min_samples = 3
class_counts = data['intent'].value_counts()
classes_to_remove = class_counts[class_counts < min_samples].index
data = data[~data['intent'].isin(classes_to_remove)]

# Check class distribution after removal
print("Class distribution after removing infrequent classes:")
print(data['intent'].value_counts())

# Split the data into features and labels
X = data['user_input']
y = data['intent']

# Vectorize text data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy='auto', k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning with GridSearchCV for the RandomForest model
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_resampled, y_resampled)

# Best parameters
print("Best parameters found:", grid_search.best_params_)

# Train the optimized Random Forest model
model = grid_search.best_estimator_
model.fit(X_resampled, y_resampled)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Cross-validation score for robustness
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
with open('model_and_vectorizer.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)