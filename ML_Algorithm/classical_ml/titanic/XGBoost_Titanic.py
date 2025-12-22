import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# --------------------------
# 1. Data Engineering
# --------------------------
print("Loading Data....")
# Load raw data
df = sns.load_dataset('titanic')

# Select useful features
# 'pclass': Rich/Poor (1st/3rd)
# 'sex': Biological priority
# 'age': Children priority
# 'fare': Correlation with class
# 'sibsp': Family siz
features = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'survived']
df = df[features].copy()

# CLEANING:
# Fill missing Age with Median (Standard practice)
df['age'] = df['age'].fillna(df['age'].median())
# Drop rows with other missing vals
df.dropna(inplace=True)

# ENCODING (The "Translator"):
# Convert 'sex' (male/female) to 0/1
df['sex'] = df['sex'].map({'male':0, 'female':1})

# Split X and y
X = df.drop('survived', axis=1)
y = df['survived']

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --------------------------
# 2. Hyperparameter Tuning (GridSearch)
# --------------------------
print("Tuning Model...")

# Define the model
xgb = XGBClassifier(eval_metric='logloss')

# Define the grid of settings to test
# We want to know: Is a deeper tree better? Is a slower learning rate better?
param_grid = {
    'n_estimators': [10, 25, 50, 100],     # Number of trees
    'max_depth': [2, 3, 5, 7],             # Depth of the forest
    'learning_rate': [0.01, 0.1, 0.2, 0.25] # Step size
}

# GridSearch tries EVERY combination (3x3x3 = 27 models)
grid_search = GridSearchCV(param_grid=param_grid, estimator=xgb, cv=3, verbose=1)
grid_search.fit(X, y)

# Get the winner
best_model = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

# --------------------------
# 3. Evaluation & Saving
# --------------------------
prediction = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

# SAVE THE MODEL (Serialization)
# This creates a file on your hard drive
model_filename = 'titanic_xgb_v1.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")
