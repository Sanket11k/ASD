
# Import required libraries
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create student placement dataset
data = {
    "CGPA": [7.5, 6.2, 8.5, 5.5, 9.0, 6.8, 7.8, 5.8, 8.2, 6.0,
             7.2, 5.0, 8.8, 6.5, 7.9, 5.6, 8.0, 6.3, 9.2, 5.9],

    "Aptitude_Score": [75, 60, 85, 50, 90, 65, 80, 55, 88, 58,
                       72, 45, 92, 62, 78, 48, 82, 59, 95, 52],

    "Technical_Skills_Score": [80, 55, 90, 45, 95, 60, 85, 50, 82, 52,
                              75, 40, 90, 58, 88, 50, 85, 57, 98, 48],

    "Placement": ["Placed", "Not Placed", "Placed", "Not Placed",
                  "Placed", "Not Placed", "Placed", "Not Placed",
                  "Placed", "Not Placed", "Placed", "Not Placed",
                  "Placed", "Not Placed", "Placed", "Not Placed",
                  "Placed", "Not Placed", "Placed", "Not Placed"]
}

# 2. Convert data into DataFrame
df = pd.DataFrame(data)

# 3. Create CSV file
df.to_csv("student_placement.csv", index=False)

print("CSV dataset created successfully!")
print(df)

# 4. Load the dataset
df = pd.read_csv("student_placement.csv")

# 5. Separate input features and target
X = df[["CGPA", "Aptitude_Score", "Technical_Skills_Score"]]
y = df["Placement"]

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining and testing data split successfully!")

# 7. Set MLflow experiment
mlflow.set_experiment("Student Placement Prediction")

# 8. Compare two experiments
for depth in [2, 4]:

    with mlflow.start_run(run_name=f"Decision_Tree_Depth_{depth}"):

        # Create Decision Tree model
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict placement
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nMax Depth: {depth}")
        print("Accuracy:", accuracy)

        # Log parameters
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("algorithm", "Decision Tree")
        mlflow.log_param("test_size", 0.25)

        # Log accuracy
        mlflow.log_metric("accuracy", accuracy)

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="decision_tree_model"
        )

        print("Model and experiment logged successfully!")

# 9. Completion message
print("\nBoth experiments completed successfully!")
print("Run 'mlflow ui' to compare experiments.")



python -m pip install pandas scikit-learn
python -m mlflow ui
