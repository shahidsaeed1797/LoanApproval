import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# =========================================
# 1. Load Dataset
# =========================================

df = pd.read_csv("loan.csv")


# =========================================
# 2. Handle Missing Values
# =========================================

num_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


cat_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# =========================================
# 3. Convert Categorical Values
# =========================================

df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

df["Married"] = df["Married"].map({
    "Yes": 1,
    "No": 0
})

df["Education"] = df["Education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["Self_Employed"] = df["Self_Employed"].map({
    "Yes": 1,
    "No": 0
})

df["Dependents"] = df["Dependents"].replace("3+", 3)
df["Dependents"] = pd.to_numeric(df["Dependents"])

df["Property_Area"] = df["Property_Area"].map({
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
})

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})


# =========================================
# 4. Remove Loan_ID
# =========================================

df = df.drop(columns=["Loan_ID"])


# =========================================
# 5. X and Y
# =========================================

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]


# =========================================
# 6. Train Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================
# 7. Decision Tree Model
# =========================================

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# =========================================
# 8. Accuracy
# =========================================

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))


# =========================================
# 9. USER INPUT
# =========================================

print("\n========================================")
print("         LOAN APPROVAL SYSTEM")
print("========================================")


# Gender
gender = input("\nGender (Male/Female): ").strip().title()

while gender not in ["Male", "Female"]:
    print("Invalid input!")
    gender = input("Gender (Male/Female): ").strip().title()

gender = 1 if gender == "Male" else 0


# Married
married = input("Married (Yes/No): ").strip().title()

while married not in ["Yes", "No"]:
    print("Invalid input!")
    married = input("Married (Yes/No): ").strip().title()

married = 1 if married == "Yes" else 0


# Dependents
dependents = input("Dependents (0/1/2/3+): ").strip()

while dependents not in ["0", "1", "2", "3", "3+"]:
    print("Invalid input!")
    dependents = input("Dependents (0/1/2/3+): ").strip()

if dependents == "3+":
    dependents = 3
else:
    dependents = int(dependents)


# Education
education = input(
    "Education (Graduate/Not Graduate): "
).strip().title()

while education not in ["Graduate", "Not Graduate"]:
    print("Invalid input!")
    education = input(
        "Education (Graduate/Not Graduate): "
    ).strip().title()

education = 1 if education == "Graduate" else 0


# Self Employed
self_employed = input(
    "Self Employed (Yes/No): "
).strip().title()

while self_employed not in ["Yes", "No"]:
    print("Invalid input!")
    self_employed = input(
        "Self Employed (Yes/No): "
    ).strip().title()

self_employed = 1 if self_employed == "Yes" else 0


# =========================================
# APPLICANT INCOME
# Maximum = 10,000,000
# =========================================

while True:

    try:
        applicant_income = float(
            input("Applicant Income (Max 10,000,000): ")
        )

        if 0 <= applicant_income <= 10000000:
            break
        else:
            print("Income must be between 0 and 10,000,000.")

    except ValueError:
        print("Please enter a valid number.")


# =========================================
# CO-APPLICANT INCOME
# Maximum = 5,000,000
# =========================================

while True:

    try:
        coapplicant_income = float(
            input("Coapplicant Income (Max 5,000,000): ")
        )

        if 0 <= coapplicant_income <= 5000000:
            break
        else:
            print("Income must be between 0 and 5,000,000.")

    except ValueError:
        print("Please enter a valid number.")


# =========================================
# LOAN AMOUNT
# Maximum = 5,000,000
# =========================================

while True:

    try:
        loan_amount = float(
            input("Loan Amount (Max 5,000,000): ")
        )

        if 0 < loan_amount <= 5000000:
            break
        else:
            print("Loan amount must be between 1 and 5,000,000.")

    except ValueError:
        print("Please enter a valid number.")


# =========================================
# LOAN TERM
# =========================================

while True:

    try:
        loan_term = float(
            input("Loan Amount Term (months, Max 480): ")
        )

        if 1 <= loan_term <= 480:
            break
        else:
            print("Loan term must be between 1 and 480 months.")

    except ValueError:
        print("Please enter a valid number.")


# =========================================
# CREDIT HISTORY
# =========================================

while True:

    try:
        credit_history = int(
            input("Credit History (1 = Good, 0 = Bad): ")
        )

        if credit_history in [0, 1]:
            break
        else:
            print("Please enter 1 or 0.")

    except ValueError:
        print("Please enter 1 or 0.")


# =========================================
# PROPERTY AREA
# =========================================

property_area = input(
    "Property Area (Urban/Semiurban/Rural): "
).strip().title()

while property_area not in ["Urban", "Semiurban", "Rural"]:

    print("Invalid property area!")

    property_area = input(
        "Property Area (Urban/Semiurban/Rural): "
    ).strip().title()


property_area = {
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
}[property_area]


# =========================================
# 10. Create User Data
# =========================================

user_data = pd.DataFrame({

    "Gender": [gender],

    "Married": [married],

    "Dependents": [dependents],

    "Education": [education],

    "Self_Employed": [self_employed],

    "ApplicantIncome": [applicant_income],

    "CoapplicantIncome": [coapplicant_income],

    "LoanAmount": [loan_amount],

    "Loan_Amount_Term": [loan_term],

    "Credit_History": [credit_history],

    "Property_Area": [property_area]

})


# =========================================
# 11. Prediction
# =========================================

prediction = model.predict(user_data)


# =========================================
# 12. Result
# =========================================

print("\n========================================")

if prediction[0] == 1:
    print("          LOAN APPROVED")
else:
    print("       LOAN NOT APPROVED")

print("========================================")