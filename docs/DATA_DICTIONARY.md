# Data Dictionary

Canonical final modeling file: `data/processed/final/E1_Year1CGPA_FinalClean.csv`

- Rows: **3,516**
- Columns: **24**
- Predictors: **23**
- Target: **Year1CGPA**

| Attribute | Role | Type | Distinct non-missing values | Missing |
|---|---|---|---:|---:|
| Faculty | Predictor | Categorical / nominal | 6 | 0 |
| CurrentInstitution | Predictor | Categorical / nominal | 15 | 0 |
| HSCGraduationYear | Predictor | Numeric | 4 | 0 |
| Gender | Predictor | Categorical / nominal | 2 | 0 |
| CollegeCategory | Predictor | Categorical / nominal | 2 | 0 |
| CollegeLocation | Predictor | Categorical / nominal | 4 | 0 |
| HSCLearningSource | Predictor | Categorical / nominal | 5 | 0 |
| CollegeAttendance | Predictor | Categorical / nominal | 5 | 0 |
| HSCMath | Predictor | Categorical / nominal | 5 | 0 |
| HSCPhysics | Predictor | Categorical / nominal | 5 | 0 |
| HSCChemistry | Predictor | Categorical / nominal | 5 | 0 |
| FamilyMembers | Predictor | Categorical / nominal | 5 | 0 |
| BirthOrder | Predictor | Categorical / nominal | 4 | 0 |
| HouseholdIncomeYearly | Predictor | Categorical / nominal | 4 | 0 |
| HSCMonthlyExpenditure | Predictor | Categorical / nominal | 5 | 0 |
| InternetAvailability | Predictor | Categorical / nominal | 4 | 0 |
| ComputerAvailability | Predictor | Categorical / nominal | 2 | 0 |
| Year1CGPA | Target | Categorical / nominal | 5 | 0 |
| UniversityPreferenceOrder | Predictor | Categorical / nominal | 5 | 0 |
| DepartmentPreferenceOrder | Predictor | Categorical / nominal | 5 | 0 |
| FatherEducation | Predictor | Categorical / nominal | 6 | 0 |
| MotherEducation | Predictor | Categorical / nominal | 6 | 0 |
| FatherEmploymentSector | Predictor | Categorical / nominal | 5 | 0 |
| MotherEmploymentSector | Predictor | Categorical / nominal | 6 | 1 |

## Target class distribution

| Class | Count | Percent |
|---|---:|---:|
| 3.25 - 3.50 (৩.২৫ - ৩.৫০) | 985 | 28.01% |
| 3.50 - 3.75 (৩.৫০ - ৩.৭৫) | 820 | 23.32% |
| 3.00 - 3.25 (৩.০০ - ৩.২৫) | 655 | 18.63% |
| Above 3.75 (৩.৭৫ এর উপরে) | 593 | 16.87% |
| Less than 3.00 (৩.০০ এর কম) | 463 | 13.17% |

## Missing values

- `MotherEmploymentSector`: 1 missing value(s)
