# Import dataset loader (Iris dataset)
from sklearn.datasets import load_iris  

# Import function to split dataset into training and testing sets
from sklearn.model_selection import train_test_split  

# Import Decision Tree model and visualization function
from sklearn.tree import DecisionTreeClassifier, plot_tree  

# Import AdaBoost (Boosting algorithm)
from sklearn.ensemble import AdaBoostClassifier  

# Import evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix  

# Import plotting library
import matplotlib.pyplot as plt  


# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
data = load_iris()  
# Loads the Iris dataset (features + labels)

X = data.data  
# Feature matrix (sepal length, sepal width, etc.)

y = data.target  
# Target labels (flower classes: setosa, versicolor, virginica)


# -----------------------------
# STEP 2: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Splits data:
# 80% for training, 20% for testing
# random_state ensures reproducibility


# -----------------------------
# STEP 3: Train Decision Tree
# -----------------------------
dt_model = DecisionTreeClassifier(random_state=42)  
# Create Decision Tree model

dt_model.fit(X_train, y_train)  
# Train the model using training data


# -----------------------------
# STEP 4: Make Predictions
# -----------------------------
dt_pred = dt_model.predict(X_test)  
# Predict labels for test data


# -----------------------------
# STEP 5: Evaluate Model
# -----------------------------
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))  
# Accuracy = correct predictions / total predictions

print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))  
# Confusion matrix shows correct vs incorrect predictions per class


# -----------------------------
# STEP 6: Visualize Decision Tree
# -----------------------------
plt.figure(figsize=(12, 8))  
# Set figure size

plot_tree(
    dt_model,
    feature_names=data.feature_names,  
    # Names of features (for readability)

    class_names=data.target_names,  
    # Names of classes

    filled=True  
    # Color nodes based on class
)

plt.title("Decision Tree Visualization")  
# Add title

plt.show()  
# Display the tree


# -----------------------------
# STEP 7: Create Weak Learner
# -----------------------------
weak_model = DecisionTreeClassifier(max_depth=1)  
# Decision stump (very simple tree)
# Used as base learner in boosting


# -----------------------------
# STEP 8: Apply AdaBoost
# -----------------------------
boost_model = AdaBoostClassifier(
    estimator=weak_model,   # Base model (weak learner)
    n_estimators=50,        # Number of trees
    random_state=42
)

boost_model.fit(X_train, y_train)  
# Train boosting model


# -----------------------------
# STEP 9: Boosting Predictions
# -----------------------------
boost_pred = boost_model.predict(X_test)  
# Predict using boosted model


# -----------------------------
# STEP 10: Evaluate Boosting
# -----------------------------
print("\nAdaBoost Accuracy:", accuracy_score(y_test, boost_pred))  
# Accuracy of boosted model

print("AdaBoost Confusion Matrix:\n", confusion_matrix(y_test, boost_pred))  
# Compare with Decision Tree
