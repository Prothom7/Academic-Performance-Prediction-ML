"""Shared paths, serialization, ordered survey encodings and audit helpers."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processed/intermediate/academic_performance_clean_v3.csv'
SEED = 1
BANDS = ['<3.00', '3.00–3.25', '3.25–3.50', '3.50–3.75', '>3.75']
TABLES = ROOT / 'results/tables'
FIGURES = ROOT / 'results/figures'
METRICS = ROOT / 'results/metrics'
LOGS = ROOT / 'results/logs'
DOCS = ROOT / 'docs'
BASE_FEATURES = [
    'Faculty', 'CurrentInstitution', 'HSCGraduationYear', 'Gender',
    'CollegeCategory', 'CollegeLocation', 'HSCLearningSource', 'CollegeAttendance',
    'HSCMath', 'HSCPhysics', 'HSCChemistry', 'FamilyMembers', 'BirthOrder',
    'HouseholdIncomeYearly', 'HSCMonthlyExpenditure', 'InternetAvailability',
    'ComputerAvailability', 'UniversityPreferenceOrder', 'DepartmentPreferenceOrder',
    'FatherEducation', 'MotherEducation', 'FatherEmploymentSector', 'MotherEmploymentSector',
]
REQUIRED_COLUMNS = ['Timestamp', *BASE_FEATURES, 'Year1CGPA', 'Year2CGPA',
    'LivingArrangement', 'PartTimeEmployment', 'UniversityMonthlyExpenditure',
    'FinancialSupportSource', 'StudentClubMember', 'ResearchParticipation']

def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def git(*args):
    result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return relative(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=convert, allow_nan=False) + '\n', encoding='utf-8')

def save_table(name, frame):
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f'{name}.csv', index=False, float_format='%.12g', lineterminator='\n')
    # Machine-readable CSVs also have automatically rendered Markdown companions.
    (TABLES / f'{name}.md').write_text(frame.to_markdown(index=False, floatfmt='.5g') + '\n', encoding='utf-8')
    return frame

def human_name(name):
    return re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name).replace('CGPA', ' CGPA').strip()

def english(value):
    if pd.isna(value):
        return np.nan
    # Keep useful English parentheses; remove the bilingual suffix only.
    return re.split(r'\s*\([^)]*[\u0980-\u09ff]', str(value), maxsplit=1)[0].strip()

def cgpa_band(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    for prefix, band in [('Less than 3.00', 0), ('3.00', 1), ('3.25', 2), ('3.50', 3), ('Above 3.75', 4)]:
        if text.startswith(prefix):
            return band
    if text in BANDS:
        return BANDS.index(text)
    raise ValueError(f'Unrecognized CGPA category: {value!r}')

def research_binary(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text.startswith('yes') or text == 'participated':
        return 1
    if text.startswith('no') or text == 'notparticipated':
        return 0
    raise ValueError(f'Unrecognized research category: {value!r}')

def trajectory(delta):
    return np.where(np.asarray(delta) > 0, 'Improved', np.where(np.asarray(delta) < 0, 'Declined', 'Stable'))

def ordered_code(value, prefixes):
    if pd.isna(value):
        return np.nan
    for i, prefix in enumerate(prefixes):
        if str(value).startswith(prefix):
            return i
    raise ValueError(f'Unrecognized ordered category: {value!r}; expected {prefixes}')

ORDER_PREFIXES = {
    **{k: ['< 120', '120', '140', '160', '190'] for k in ['HSCMath', 'HSCPhysics', 'HSCChemistry']},
    'StudentClubMember': ['No', 'Yes, one', 'Yes, two', 'Yes, three'],
    'CollegeAttendance': ['Very rarely', 'Rarely', 'Occasionally', 'Regularly', 'Very regularly'],
    'FamilyMembers': ['2', '3', '4', '6', '8'],
    'HouseholdIncomeYearly': ['Less than', '5 -', '10-', 'Above'],
    'HSCMonthlyExpenditure': ['Less than', '5,000', '10,000', '20,000', 'Above'],
    'UniversityMonthlyExpenditure': ['Less than', '5,000', '10,000', '20,000', 'Above'],
    'UniversityPreferenceOrder': ['1st', '2nd', '3rd', '4th', '5th'],
    'DepartmentPreferenceOrder': ['1st', '2nd', '3rd', '4th', '5th'],
    'FatherEducation': ['No formal', 'Primary', 'Secondary', 'Higher Secondary', 'Undergraduate', 'Postgraduate'],
    'MotherEducation': ['No formal', 'Primary', 'Secondary', 'Higher Secondary', 'Undergraduate', 'Postgraduate'],
    'PartTimeEmployment': ['No', 'Yes, less', 'Yes, 10', 'Yes, 21', 'Yes, more'],
}

def load_population():
    if not DATA.is_file():
        raise FileNotFoundError(f'Required authoritative dataset missing: {DATA}')
    raw = pd.read_csv(DATA, encoding='utf-8-sig')
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing or raw.shape != (3516, 32):
        raise ValueError(f'Dataset contract failed: shape={raw.shape}, missing={missing}')
    df = raw.copy()
    for col in df.select_dtypes(exclude='number'):
        df[col] = df[col].map(english)
    df['Year1Band'] = raw.Year1CGPA.map(cgpa_band).astype(int)
    df['Year2Band'] = raw.Year2CGPA.map(cgpa_band).astype(int)
    df['trajectory_delta'] = df.Year2Band - df.Year1Band
    df['Trajectory'] = trajectory(df.trajectory_delta)
    df['high_year1'] = (df.Year1Band >= 3).astype(int)
    df['high_year2'] = (df.Year2Band >= 3).astype(int)
    df['low_year2'] = (df.Year2Band == 0).astype(int)
    df['improved'] = (df.Trajectory == 'Improved').astype(int)
    df['declined'] = (df.Trajectory == 'Declined').astype(int)
    df['research'] = raw.ResearchParticipation.map(research_binary)
    for col, prefixes in ORDER_PREFIXES.items():
        df[f'{col}Ordinal'] = raw[col].map(lambda x: ordered_code(x, prefixes))
    df['Male'] = (df.Gender == 'Male').astype(int)
    return raw, df

def feature_list(study):
    if study == 'year2_without_year1' or study == 'year1':
        return BASE_FEATURES.copy()
    if study in ['year2_with_year1', 'research', 'research_historical_python']:
        # Preserve historical placement of the Year-1 feature for the RP encoder.
        return BASE_FEATURES[:17] + ['Year1CGPA'] + BASE_FEATURES[17:]
    raise ValueError(study)
