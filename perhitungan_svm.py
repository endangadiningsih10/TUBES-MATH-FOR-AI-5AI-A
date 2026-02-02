import pandas as pd
import os

# ==============================
# LOAD DATASET
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "dataset_svm.csv")

data = pd.read_csv(csv_path, sep=";")

# ==============================
# FEATURES & TARGET
# ==============================
X = data[["nilai_kehadiran", "nilai_ujian"]].values
y = data["target"].values

# ==============================
# PARAMETER AWAL
# ==============================
w1, w2, b = 0.0, 0.0, 0.0
learning_rate = 0.0001
lambda_reg = 0.01
n_iter = 5

# ==============================
# TRAINING SVM (MANUAL)
# ==============================
for iterasi in range(1, n_iter + 1):

    total_loss = 0.0
    grad_w1 = 0.0
    grad_w2 = 0.0
    grad_b  = 0.0

    for i in range(len(X)):
        x1, x2 = X[i]
        yi = y[i]

        fx = w1 * x1 + w2 * x2 + b
        y_fx = yi * fx

        loss = max(0, 1 - y_fx)
        total_loss += loss

        if y_fx < 1:
            grad_w1 += -yi * x1
            grad_w2 += -yi * x2
            grad_b  += -yi

    grad_w1 += lambda_reg * w1
    grad_w2 += lambda_reg * w2

    print(f"\nITERASI {iterasi}")
    print(f"Total Loss = {total_loss:.6f}")

    if total_loss > 0:
        print("Masih terdapat error (loss > 0) -> dilakukan update bobot")
    else:
        print("Tidak ada error -> bobot tidak diperbarui")

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    b  -= learning_rate * grad_b

    print("Update bobot:")
    print(f"w1 = {w1:.6f}")
    print(f"w2 = {w2:.6f}")
    print(f"b  = {b:.6f}")
    print("-" * 40)