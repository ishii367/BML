import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
url = "/content/drive/MyDrive/BML_practical/winequality-red.csv"
data = pd.read_csv(url, sep=';')

# Step 2: Explore data
print(data.head())
print(data.info())

# Step 3: Split features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Optional: Convert quality scores into classes (e.g., good vs bad)
# For simplicity, let's classify wines into "good" (quality >=7) and "not good" (quality <7)
y = (y >= 7).astype(int)

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train SVM classifier
svm_clf = SVC(kernel='rbf', random_state=42)
svm_clf.fit(X_train, y_train)

# Step 7: Predict and evaluate accuracy
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM classifier accuracy: {accuracy:.4f}")
