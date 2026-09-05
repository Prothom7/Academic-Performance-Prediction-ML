"""
Create a reproducible stratified 70/15/15 split for the Year1CGPA project.

Input:
    E1_Year1CGPA_FinalClean.csv

Outputs:
    train.csv
    validation.csv
    test.csv

The split is stratified on Year1CGPA so each split approximately preserves
the five-class target distribution.
"""

from pathlib import Path
import csv
from sklearn.model_selection import train_test_split

INPUT = Path("E1_Year1CGPA_FinalClean.csv")
TARGET = "Year1CGPA"
RANDOM_STATE = 1

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

labels = [row[TARGET] for row in rows]

# 70% training set.
train_rows, temp_rows = train_test_split(
    rows,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=labels,
)

# Remaining 30% is divided equally into validation and test sets.
temp_labels = [row[TARGET] for row in temp_rows]
validation_rows, test_rows = train_test_split(
    temp_rows,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_labels,
)

def save_csv(filename, data):
    with Path(filename).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

save_csv("train.csv", train_rows)
save_csv("validation.csv", validation_rows)
save_csv("test.csv", test_rows)

print(f"Train:      {len(train_rows)}")
print(f"Validation: {len(validation_rows)}")
print(f"Test:       {len(test_rows)}")
print(f"Total:      {len(train_rows) + len(validation_rows) + len(test_rows)}")
