import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Load Dataset
df = pd.read_csv("phishing_data1.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

if y.min() == -1:
  y = y.map({-1: 0, 1: 1})

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling (Needed for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the 3 distinct algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

print("\n--- Training Models ---")
for name, model in models.items():
  if name == "KNN":
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
  else:
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

  print(f"{name} Accuracy: {acc * 100:.2f}%")
  joblib.dump(model, f"model_{name.lower().replace(' ', '_')}.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")
print("\n[+] 3 Models trained and saved successfully!")