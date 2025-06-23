import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
ds = load_dataset("ucirvine/sms_spam", split="train")
df = pd.DataFrame(ds)

# 2. Show original structure
print("📄 Original Columns:", df.columns.tolist())
print("🔍 Raw label values:", df["label"].unique())

# 3. Label mapping
if df["label"].dtype == object:
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 4. Clean data: drop missing/invalid rows
df = df.dropna(subset=["sms", "label"])
df["label"] = df["label"].astype(int)

# 5. Confirm cleaned structure
print("✅ Final unique labels:", df["label"].unique())
print("🧾 Label distribution:\n", df["label"].value_counts())

# 6. Vectorize SMS text
vectorizer = CountVectorizer(max_features=1000, stop_words="english", binary=True)
X = vectorizer.fit_transform(df["sms"]).toarray()
y = df["label"].values

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
