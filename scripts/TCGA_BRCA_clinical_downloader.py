#!/usr/bin/env python3
"""
TCGA BRCA Clinical Data Downloader
Educational script for downloading and processing breast cancer clinical data from TCGA
Compatible with the 6-14 TCGA executable guide
"""

import requests
import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import time
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings if needed
urllib3.disable_warnings(InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TCGABRCADownloader:
    """
    Simplified TCGA downloader focused on BRCA (breast cancer) data
    Designed for educational use following the 6-14 executable guide
    """
    
    def __init__(self):
        self.gdc_api_base = "https://api.gdc.cancer.gov"
        self.cases_endpoint = f"{self.gdc_api_base}/cases"
        self.session = requests.Session()
        
        # Session configuration for reliable downloads
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({
            'User-Agent': 'TCGA-BRCA-Downloader/1.0',
            'Content-Type': 'application/json'
        })
        
        # Setup output directory
        self.output_dir = Path("tcga_brca_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def test_connection(self):
        """Test connection to GDC API"""
        print("🔌 Testing connection to GDC API...")
        
        try:
            response = self.session.get(f"{self.gdc_api_base}/status", timeout=30)
            response.raise_for_status()
            
            status_data = response.json()
            print("✅ Connected to GDC API successfully!")
            print(f"   Data Release: {status_data.get('data_release', 'Unknown')}")
            print(f"   Version: {status_data.get('tag', 'Unknown')}")
            return True
            
        except requests.exceptions.SSLError:
            print("⚠️  SSL verification failed, trying without verification...")
            self.session.verify = False
            try:
                response = self.session.get(f"{self.gdc_api_base}/status", timeout=30, verify=False)
                response.raise_for_status()
                print("✅ Connected (SSL verification disabled)")
                return True
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def download_brca_clinical_data(self):
        """
        Download comprehensive BRCA clinical data from TCGA
        This mirrors the approach shown in the executable guide Task 2
        """
        print("📊 Downloading BRCA clinical data from TCGA...")
        print("   This will collect comprehensive patient information including:")
        print("   • Demographics (age, race, gender, vital status)")
        print("   • Diagnosis details (tumor grade, stage, histology)")
        print("   • Follow-up information (survival, recurrence)")
        print("   • Treatment exposure data")
        
        # Query parameters - requesting comprehensive clinical data for BRCA
        payload = {
            "filters": {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": ["TCGA-BRCA"]  # Focus only on breast cancer
                }
            },
            # Expand to get all related clinical information
            "expand": "demographic,diagnoses,exposures,follow_ups,annotations",
            "format": "JSON",
            "size": "3000"  # Ensure we get all BRCA cases
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries}: Querying GDC API...")
                
                response = self.session.post(
                    self.cases_endpoint,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                
                data = response.json()
                cases = data.get('data', {}).get('hits', [])
                
                print(f"✅ Successfully retrieved {len(cases)} BRCA cases!")
                return cases
                
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"   Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception("Max retries exceeded - API may be unavailable")
                    
            except Exception as e:
                print(f"❌ Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    raise
                    
        raise Exception("Failed to download data after all retries")
    
    def flatten_case_data(self, case):
        """
        Convert nested TCGA case data into flat structure for analysis
        This follows the data processing approach from the comprehensive pipeline
        """
        flat_case = {
            # Basic case information
            'case_id': case.get('id'),
            'submitter_id': case.get('submitter_id'),
            'primary_site': case.get('primary_site'),
            'disease_type': case.get('disease_type'),
            'project_id': 'TCGA-BRCA'
        }
        
        # Demographics - patient characteristics
        demo = case.get('demographic', {})
        if demo:
            flat_case.update({
                'demo_race': demo.get('race'),
                'demo_gender': demo.get('gender'),
                'demo_ethnicity': demo.get('ethnicity'),
                'demo_vital_status': demo.get('vital_status'),
                'demo_age_at_index': demo.get('age_at_index'),
                'demo_days_to_birth': demo.get('days_to_birth'),
                'demo_days_to_death': demo.get('days_to_death'),
                'demo_year_of_birth': demo.get('year_of_birth'),
                'demo_year_of_death': demo.get('year_of_death')
            })
        
        # Primary diagnosis information
        diagnoses = case.get('diagnoses', [])
        if diagnoses:
            # Find primary diagnosis or use first one
            primary_dx = None
            for dx in diagnoses:
                if dx.get('diagnosis_is_primary_disease', False):
                    primary_dx = dx
                    break
            if not primary_dx:
                primary_dx = diagnoses[0]
            
            flat_case.update({
                'dx_primary_diagnosis': primary_dx.get('primary_diagnosis'),
                'dx_age_at_diagnosis': primary_dx.get('age_at_diagnosis'),
                'dx_days_to_diagnosis': primary_dx.get('days_to_diagnosis'),
                'dx_year_of_diagnosis': primary_dx.get('year_of_diagnosis'),
                'dx_tumor_grade': primary_dx.get('tumor_grade'),
                'dx_ajcc_pathologic_stage': primary_dx.get('ajcc_pathologic_stage'),
                'dx_ajcc_pathologic_t': primary_dx.get('ajcc_pathologic_t'),
                'dx_ajcc_pathologic_n': primary_dx.get('ajcc_pathologic_n'),
                'dx_ajcc_pathologic_m': primary_dx.get('ajcc_pathologic_m'),
                'dx_tissue_or_organ_of_origin': primary_dx.get('tissue_or_organ_of_origin'),
                'dx_morphology': primary_dx.get('morphology'),
                'dx_prior_malignancy': primary_dx.get('prior_malignancy'),
                'dx_residual_disease': primary_dx.get('residual_disease'),
                'dx_classification_of_tumor': primary_dx.get('classification_of_tumor')
            })
        
        # Follow-up information - critical for survival analysis
        follow_ups = case.get('follow_ups', [])
        if follow_ups:
            # Get most recent follow-up
            recent_fu = max(follow_ups, 
                          key=lambda x: x.get('days_to_follow_up', 0) if x.get('days_to_follow_up') is not None else 0)
            
            flat_case.update({
                'fu_days_to_follow_up': recent_fu.get('days_to_follow_up'),
                'fu_disease_response': recent_fu.get('disease_response'),
                'fu_progression_or_recurrence': recent_fu.get('progression_or_recurrence'),
                'fu_days_to_progression': recent_fu.get('days_to_progression'),
                'fu_days_to_recurrence': recent_fu.get('days_to_recurrence'),
                'fu_ecog_performance_status': recent_fu.get('ecog_performance_status'),
                'fu_karnofsky_performance_status': recent_fu.get('karnofsky_performance_status')
            })
        
        # Exposure information
        exposures = case.get('exposures', [])
        if exposures:
            exposure = exposures[0]
            flat_case.update({
                'exp_alcohol_history': exposure.get('alcohol_history'),
                'exp_tobacco_smoking_history': exposure.get('tobacco_smoking_history'),
                'exp_years_smoked': exposure.get('years_smoked'),
                'exp_cigarettes_per_day': exposure.get('cigarettes_per_day'),
                'exp_alcohol_drinks_per_day': exposure.get('alcohol_drinks_per_day')
            })
        
        return flat_case
    
    def calculate_survival_endpoints(self, df):
        """
        Calculate standardized survival endpoints for research
        This implements the enhanced survival analysis from main_clinical.py
        """
        print("🔬 Calculating standardized survival endpoints...")
        print("   Computing: Overall Survival, Progression-Free Interval, Disease-Specific Survival")
        
        # Initialize survival columns
        survival_cols = {
            'overall_survival_months': np.nan,
            'overall_survival_status': 0,
            'progression_free_interval_months': np.nan,
            'progression_free_interval_status': 0,
            'disease_specific_survival_months': np.nan,
            'disease_specific_survival_status': 0,
            'disease_free_interval_months': np.nan,
            'disease_free_interval_status': 0
        }
        
        for col, default_val in survival_cols.items():
            df[col] = default_val
        
        # Overall Survival (OS) calculation
        print("   📊 Computing Overall Survival...")
        for idx, row in df.iterrows():
            vital_status = str(row.get('demo_vital_status', '')).lower()
            days_to_death = row.get('demo_days_to_death')
            days_to_followup = row.get('fu_days_to_follow_up')
            
            if vital_status in ['dead', 'deceased']:
                df.at[idx, 'overall_survival_status'] = 1
                if pd.notna(days_to_death):
                    df.at[idx, 'overall_survival_months'] = days_to_death / 30.44
            elif vital_status in ['alive']:
                df.at[idx, 'overall_survival_status'] = 0
                if pd.notna(days_to_followup):
                    df.at[idx, 'overall_survival_months'] = days_to_followup / 30.44
        
        # Progression-Free Interval (PFI) calculation
        print("   📊 Computing Progression-Free Interval...")
        for idx, row in df.iterrows():
            progression = str(row.get('fu_progression_or_recurrence', '')).lower()
            days_to_progression = row.get('fu_days_to_progression')
            days_to_recurrence = row.get('fu_days_to_recurrence')
            days_to_followup = row.get('fu_days_to_follow_up')
            
            if progression in ['yes', 'true', 'progression', 'recurrence']:
                df.at[idx, 'progression_free_interval_status'] = 1
                
                # Use earliest progression/recurrence time
                prog_time = None
                if pd.notna(days_to_progression):
                    prog_time = days_to_progression
                if pd.notna(days_to_recurrence):
                    if prog_time is None or days_to_recurrence < prog_time:
                        prog_time = days_to_recurrence
                
                if prog_time is not None:
                    df.at[idx, 'progression_free_interval_months'] = prog_time / 30.44
            else:
                df.at[idx, 'progression_free_interval_status'] = 0
                if pd.notna(days_to_followup):
                    df.at[idx, 'progression_free_interval_months'] = days_to_followup / 30.44
        
        # Disease-Specific Survival (DSS) - assume cancer-related if died within follow-up
        print("   📊 Computing Disease-Specific Survival...")
        for idx, row in df.iterrows():
            vital_status = str(row.get('demo_vital_status', '')).lower()
            days_to_death = row.get('demo_days_to_death')
            days_to_followup = row.get('fu_days_to_follow_up')
            
            if vital_status in ['dead', 'deceased']:
                df.at[idx, 'disease_specific_survival_status'] = 1
                if pd.notna(days_to_death):
                    df.at[idx, 'disease_specific_survival_months'] = days_to_death / 30.44
            else:
                df.at[idx, 'disease_specific_survival_status'] = 0
                if pd.notna(days_to_followup):
                    df.at[idx, 'disease_specific_survival_months'] = days_to_followup / 30.44
        
        # Print survival summary
        print("\n📈 Survival Endpoint Summary:")
        for endpoint in ['overall_survival', 'progression_free_interval', 'disease_specific_survival']:
            months_col = f"{endpoint}_months"
            status_col = f"{endpoint}_status"
            
            valid_months = df[months_col].notna().sum()
            events = df[status_col].sum()
            
            print(f"   {endpoint.upper()}:")
            print(f"     Valid time data: {valid_months:,} cases ({valid_months/len(df)*100:.1f}%)")
            print(f"     Events: {events:,} cases ({events/len(df)*100:.1f}%)")
            
            if valid_months > 0:
                median_time = df[months_col].median()
                print(f"     Median follow-up: {median_time:.1f} months")
        
        return df
    
    def add_clinical_covariates(self, df):
        """Add standardized clinical covariates for analysis"""
        print("🔬 Adding clinical covariates...")
        
        # Age groups for survival analysis
        df['age_group'] = pd.cut(df['demo_age_at_index'],
                                bins=[0, 50, 65, 100],
                                labels=['≤50', '51-65', '>65'])
        
        # Standardized tumor grade
        grade_map = {
            'G1': 'Grade 1', 'Grade 1': 'Grade 1', 'Well differentiated': 'Grade 1',
            'G2': 'Grade 2', 'Grade 2': 'Grade 2', 'Moderately differentiated': 'Grade 2',
            'G3': 'Grade 3', 'Grade 3': 'Grade 3', 'Poorly differentiated': 'Grade 3',
            'G4': 'Grade 4', 'Grade 4': 'Grade 4', 'Undifferentiated': 'Grade 4',
            'GX': 'Unknown', 'Unknown': 'Unknown'
        }
        df['tumor_grade_clean'] = df['dx_tumor_grade'].map(grade_map).fillna('Unknown')
        
        # Simplified stage groups
        def simplify_stage(stage):
            if pd.isna(stage):
                return 'Unknown'
            stage_str = str(stage).upper()
            if any(x in stage_str for x in ['I', '1']):
                if any(x in stage_str for x in ['II', '2']):
                    return 'Stage II'
                return 'Stage I'
            elif any(x in stage_str for x in ['II', '2']):
                return 'Stage II'
            elif any(x in stage_str for x in ['III', '3']):
                return 'Stage III'
            elif any(x in stage_str for x in ['IV', '4']):
                return 'Stage IV'
            else:
                return 'Unknown'
        
        df['stage_group'] = df['dx_ajcc_pathologic_stage'].apply(simplify_stage)
        
        # Nodal status
        def simplify_nodes(n_stage):
            if pd.isna(n_stage):
                return 'Unknown'
            n_str = str(n_stage).upper()
            if 'N0' in n_str:
                return 'N0'
            elif any(x in n_str for x in ['N1', 'N2', 'N3']):
                return 'N+'
            else:
                return 'Unknown'
        
        df['nodal_status'] = df['dx_ajcc_pathologic_n'].apply(simplify_nodes)
        
        return df
    
    def save_data(self, df):
        """Save data in multiple formats for different analysis tools"""
        print("💾 Saving data files...")
        
        # Save main files
        csv_file = self.output_dir / "brca_clinical_data.csv"
        parquet_file = self.output_dir / "brca_clinical_data.parquet"
        
        df.to_csv(csv_file, index=False)
        df.to_parquet(parquet_file, index=False)
        
        print(f"✅ Saved CSV: {csv_file}")
        print(f"✅ Saved Parquet: {parquet_file}")
        
        # Save metadata
        metadata = {
            "dataset": "TCGA-BRCA Clinical Data with Survival Endpoints",
            "description": "Comprehensive breast cancer clinical data from TCGA with calculated survival metrics",
            "source": "GDC Cases Endpoint",
            "project": "TCGA-BRCA",
            "total_cases": len(df),
            "total_columns": len(df.columns),
            "created_date": datetime.now().isoformat(),
            "survival_endpoints": [
                "overall_survival_months", "overall_survival_status",
                "progression_free_interval_months", "progression_free_interval_status",
                "disease_specific_survival_months", "disease_specific_survival_status"
            ],
            "usage_examples": {
                "load_data": "df = pd.read_csv('brca_clinical_data.csv')",
                "survival_analysis": "df[['overall_survival_months', 'overall_survival_status']]",
                "filter_by_stage": "df[df['stage_group'] == 'Stage II']",
                "age_groups": "df['age_group'].value_counts()"
            },
            "column_descriptions": {
                "case_id": "Unique TCGA case identifier",
                "demo_vital_status": "Patient vital status (Alive/Dead)",
                "demo_age_at_index": "Age at initial pathologic diagnosis",
                "dx_tumor_grade": "Tumor histologic grade",
                "dx_ajcc_pathologic_stage": "AJCC pathologic stage",
                "overall_survival_months": "Overall survival time in months",
                "overall_survival_status": "Overall survival event (0=censored, 1=death)"
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved metadata: {metadata_file}")
        
        return csv_file, parquet_file, metadata_file
    
    def print_summary(self, df):
        """Print comprehensive dataset summary"""
        print("\n" + "="*60)
        print("📊 TCGA BRCA CLINICAL DATA SUMMARY")
        print("="*60)
        
        print(f"📋 Dataset Shape: {df.shape[0]:,} patients × {df.shape[1]:,} features")
        print(f"🏥 Project: TCGA-BRCA (Breast Invasive Carcinoma)")
        
        # Demographics summary
        print(f"\n👥 Demographics:")
        if 'demo_vital_status' in df.columns:
            vital_counts = df['demo_vital_status'].value_counts()
            for status, count in vital_counts.items():
                print(f"   {status}: {count:,} patients")
        
        if 'demo_age_at_index' in df.columns:
            age_median = df['demo_age_at_index'].median()
            age_range = (df['demo_age_at_index'].min(), df['demo_age_at_index'].max())
            print(f"   Age: median {age_median:.0f} years (range: {age_range[0]:.0f}-{age_range[1]:.0f})")
        
        # Clinical characteristics
        print(f"\n🔬 Clinical Characteristics:")
        if 'stage_group' in df.columns:
            print("   Tumor Stages:")
            stage_counts = df['stage_group'].value_counts()
            for stage, count in stage_counts.items():
                print(f"     {stage}: {count:,} patients")
        
        if 'tumor_grade_clean' in df.columns:
            print("   Tumor Grades:")
            grade_counts = df['tumor_grade_clean'].value_counts()
            for grade, count in grade_counts.items():
                print(f"     {grade}: {count:,} patients")
        
        # Survival data availability
        print(f"\n📈 Survival Data Completeness:")
        survival_endpoints = [
            ('Overall Survival', 'overall_survival_months', 'overall_survival_status'),
            ('Progression-Free Interval', 'progression_free_interval_months', 'progression_free_interval_status'),
            ('Disease-Specific Survival', 'disease_specific_survival_months', 'disease_specific_survival_status')
        ]
        
        for name, months_col, status_col in survival_endpoints:
            if months_col in df.columns and status_col in df.columns:
                valid_time = df[months_col].notna().sum()
                events = df[status_col].sum()
                print(f"   {name}:")
                print(f"     Time data: {valid_time:,}/{len(df):,} ({valid_time/len(df)*100:.1f}%)")
                print(f"     Events: {events:,} ({events/len(df)*100:.1f}%)")
        
        # Data quality
        print(f"\n✅ Data Quality:")
        key_fields = ['demo_age_at_index', 'dx_primary_diagnosis', 'dx_tumor_grade', 'dx_ajcc_pathologic_stage']
        for field in key_fields:
            if field in df.columns:
                completeness = df[field].notna().sum() / len(df) * 100
                print(f"   {field}: {completeness:.1f}% complete")
        
        print("\n" + "="*60)
        print("✅ BRCA clinical data download and processing completed!")
        print(f"📁 Output directory: {self.output_dir}")
        print("🔗 Ready for analysis and integration with imaging data!")
        print("="*60)


def main():
    """
    Main function - implements the BRCA download workflow from the executable guide
    """
    print("🏥 ===============================================")
    print("🏥 TCGA BRCA CLINICAL DATA DOWNLOADER")
    print("🏥 Compatible with 6-14 Executable Guide")
    print("🏥 ===============================================")
    print("📊 This script will:")
    print("   1. Connect to TCGA's GDC API")
    print("   2. Download comprehensive BRCA clinical data")
    print("   3. Process and flatten nested JSON structure")
    print("   4. Calculate standardized survival endpoints")
    print("   5. Add clinical covariates for analysis")
    print("   6. Save analysis-ready dataset")
    print("===============================================")
    
    try:
        # Initialize downloader
        downloader = TCGABRCADownloader()
        
        # Test connection
        if not downloader.test_connection():
            raise ConnectionError("Cannot connect to GDC API")
        
        # Download BRCA clinical data
        cases = downloader.download_brca_clinical_data()
        
        if not cases:
            raise ValueError("No BRCA cases retrieved from API")
        
        # Process cases into flat structure
        print(f"🔄 Processing {len(cases)} BRCA cases...")
        
        processed_cases = []
        for case in cases:
            flat_case = downloader.flatten_case_data(case)
            processed_cases.append(flat_case)
        
        # Create DataFrame
        df = pd.DataFrame(processed_cases)
        
        # Calculate survival endpoints
        df = downloader.calculate_survival_endpoints(df)
        
        # Add clinical covariates
        df = downloader.add_clinical_covariates(df)
        
        # Save data
        csv_file, parquet_file, metadata_file = downloader.save_data(df)
        
        # Print summary
        downloader.print_summary(df)
        
        # Success message with next steps
        print(f"\n🎉 SUCCESS! BRCA clinical data ready for analysis.")
        print(f"\n📋 Next steps from the executable guide:")
        print(f"   1. Load data in Python: df = pd.read_csv('{csv_file.name}')")
        print(f"   2. Explore survival data: df[['overall_survival_months', 'overall_survival_status']]")
        print(f"   3. Filter by characteristics: df[df['stage_group'] == 'Stage II']")
        print(f"   4. Proceed to Task 3: Download whole slide images")
        print(f"   5. Continue with QuPath and MONAI analysis")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Execute the download when script is run directly
    main()