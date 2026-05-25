import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import tkinter as tk
from tkinter import messagebox

# LOAD DATA
df = pd.read_csv("student.csv")

# ONLY FEATURES USED IN GUI
selected_features = [
    "Age at enrollment",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Scholarship holder",
    "Debtor",
    "Tuition fees up to date"
]

X = df[selected_features]
y = df["Target"]

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

print("Model trained successfully!")


# GUI
root = tk.Tk()
root.title("Student Dropout Prediction System")
root.geometry("520x600")

title = tk.Label(root, text="Student Dropout Prediction",
                 font=("Arial", 16, "bold"))
title.pack(pady=10)

entries = {}

for f in selected_features:
    frame = tk.Frame(root)
    frame.pack(pady=6)

    label = tk.Label(frame, text=f, width=32, anchor="w")
    label.pack(side="left")

    entry = tk.Entry(frame)
    entry.pack(side="right")

    entries[f] = entry


def predict():
    try:
        data = []
        for f in selected_features:
            data.append(float(entries[f].get()))

        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)

        pred = model.predict(data)
        result = target_encoder.inverse_transform(pred)[0]

        messagebox.showinfo(
            "Prediction Result",
            f"Predicted Student Outcome: {result}"
        )

    except Exception as e:
        messagebox.showerror(
            "Input Error",
            "Please enter valid numeric values"
        )
        print(e)


btn = tk.Button(root, text="Predict Outcome",
                font=("Arial", 12, "bold"),
                command=predict)
btn.pack(pady=20)

root.mainloop()
