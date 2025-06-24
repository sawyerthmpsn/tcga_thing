import pandas as pd
import matplotlib.pyplot as plt

# Load the downloaded BRCA clinical data (standard CSV format)
df = pd.read_csv('tcga_brca_data/brca_clinical_data.csv')

print("=== TCGA BRCA Dataset Overview ===")
print(f"Dataset shape: {df.shape}")
print(f"Total patients: {len(df):,}")

# Show some column names to understand the data structure
print(f"\nSample columns (first 10):")
print(list(df.columns[:10]))

# Examine key clinical features
print("\n=== Patient Demographics ===")
print("Vital Status:")
print(df['demo_vital_status'].value_counts())

print("\nAge Distribution:")
plt.figure(figsize=(8, 6))
plt.hist(df['demo_age_at_index'].dropna(), bins=30, edgecolor='black')
plt.title('Age Distribution at Diagnosis')
plt.xlabel('Age (years)')
plt.ylabel('Number of Patients')
plt.show()

print("\n=== Tumor Characteristics ===")
print("Stage Distribution:")
print(df['stage_group'].value_counts())

print("Grade Distribution:")
print(df['tumor_grade_clean'].value_counts())

# Survival data quality check
print("\n=== Survival Data Completeness ===")
survival_cols = ['overall_survival_months', 'progression_free_interval_months']
for col in survival_cols:
    if col in df.columns:
        completeness = df[col].notna().sum() / len(df) * 100
        print(f"{col}: {completeness:.1f}% complete")

print("\n✅ Data exploration complete! Ready for imaging analysis.")
print("💾 The CSV file can also be opened directly in Excel for manual review.")
