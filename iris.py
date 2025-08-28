# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()

# 'data' contains the feature measurements 
X = iris.data

# 'target' contains the species labels (0, 1, or 2, corresponding to the three species)
y = iris.target

df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y
print("First 5 rows of the dataset:")
print(df.head())
print("\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

model = DecisionTreeClassifier(random_state=50)

# Train the model

print("Training the model...")
model.fit(X_train, y_train)

# Make predictions on the test data
print("Making predictions...")
predictions = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)

# Print the results
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("\nExample of predictions vs. actual values:")
print(f"Predicted: {predictions[:5]}") # Print the first 5 predictions
print(f"Actual:    {y_test[:5]}")     # Print the first 5 actual values

