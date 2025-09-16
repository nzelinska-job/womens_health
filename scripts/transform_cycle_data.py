#!/usr/bin/env python3
"""
Script to transform cycle-based data to daily records
Each row represents one day of a menstrual cycle with calculated phase
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class CycleToDailyTransformer:
    """Transforms cycle-level data to daily records with phase calculation"""
    
    def __init__(self, input_file: str, output_file: str = None):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else self.input_file.parent / "daily_cycle_data.csv"
        
        # Columns to transfer from original data
        self.columns_to_transfer = [
            'Age', 'Maristatus', 'Schoolyears', 'BMI', 
            'Medvits', 'Medvitexplain'
        ]
        
        # Columns needed for phase calculation
        self.phase_columns = [
            'LengthofCycle', 'LengthofMenses', 'TotalDaysofFertility', 
            'LengthofLutealPhase', 'EstimatedDayofOvulation'
        ]
        
        print(f"🔄 Initializing transformer")
        print(f"📁 Input file: {self.input_file}")
        print(f"📁 Output file: {self.output_file}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate input data"""
        try:
            # Load with proper handling of whitespace and empty values
            df = pd.read_csv(self.input_file, na_values=['', ' ', '  ', 'NA', 'N/A', 'null'], 
                           keep_default_na=True, skipinitialspace=True)
            print(f"✅ Loaded {len(df)} cycle records")
            
            # Additional cleaning for numeric columns
            numeric_columns = self.phase_columns + ['CycleNumber']
            for col in numeric_columns:
                if col in df.columns:
                    # Clean whitespace and convert to numeric
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace(['', ' ', '  '], np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for required columns
            required_cols = self.phase_columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ Missing critical columns: {missing_cols}")
                # Only fail if we're missing absolutely critical columns
                critical_missing = [col for col in ['LengthofCycle'] if col in missing_cols]
                if critical_missing:
                    print(f"❌ Cannot proceed without: {critical_missing}")
                    sys.exit(1)
            
            # Check for user characteristic columns
            missing_user_cols = [col for col in self.columns_to_transfer if col not in df.columns]
            if missing_user_cols:
                print(f"⚠️ Missing user characteristic columns: {missing_user_cols}")
                for col in missing_user_cols:
                    df[col] = np.nan
                    print(f"   Added missing column '{col}' with NaN values")
            
            # Clean user characteristic columns as well
            for col in self.columns_to_transfer:
                if col in df.columns:
                    # For string columns, just clean whitespace
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
                        df[col] = df[col].replace(['', ' ', '  ', 'nan'], np.nan)
            
            print(f"📊 Data shape after cleaning: {df.shape}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    
    def clean_numeric_value(self, value):
        """Clean and validate numeric values, handling spaces and other edge cases"""
        if pd.isna(value):
            return np.nan
        
        # Convert to string and strip whitespace
        str_value = str(value).strip()
        
        # Check for empty string or just whitespace
        if str_value == '' or str_value.isspace():
            return np.nan
        
        # Try to convert to numeric
        try:
            return float(str_value)
        except (ValueError, TypeError):
            return np.nan
    
    def calculate_cycle_phase(self, day: int, cycle_length: int, menses_length: int, 
                             fertile_days: int, luteal_length: int, ovulation_day: int = None) -> str:
        """
        Calculate menstrual cycle phase for a given day
        If key data is missing, return NaN to preserve missing values
        
        Phases:
        - Menstrual: Days 1 to menses_length
        - Follicular: After menses until fertile window
        - Fertile: Fertile window (includes ovulation)
        - Luteal: After fertile window until end of cycle
        """
        
        # Clean all input values
        cycle_length = self.clean_numeric_value(cycle_length)
        menses_length = self.clean_numeric_value(menses_length)
        fertile_days = self.clean_numeric_value(fertile_days)
        luteal_length = self.clean_numeric_value(luteal_length)
        ovulation_day = self.clean_numeric_value(ovulation_day)
        
        # Handle missing values - if critical data is missing, return NaN
        if pd.isna(cycle_length):
            return np.nan
        
        # If menses length is missing, we can't reliably calculate phases
        if pd.isna(menses_length):
            return np.nan
        
        day = int(day)
        cycle_length = int(cycle_length)
        menses_length = int(menses_length)
        
        # Ensure day is within cycle
        if day < 1 or day > cycle_length:
            return np.nan
        
        # Menstrual phase (beginning of cycle)
        if day <= menses_length:
            return 'menstrual'
        
        # Calculate fertile window
        if not pd.isna(ovulation_day):
            ovulation_day = int(ovulation_day)
            # Fertile window: typically 5 days before ovulation + ovulation day + 1 day after
            fertile_start = max(1, ovulation_day - 5)
            fertile_end = min(cycle_length, ovulation_day + 1)
        else:
            # If we have fertile days info, use it
            if not pd.isna(fertile_days):
                fertile_days = int(fertile_days)
                # Estimate fertile window in the middle of cycle
                cycle_mid = cycle_length // 2
                fertile_start = max(menses_length + 1, cycle_mid - fertile_days // 2)
                fertile_end = min(cycle_length - (luteal_length if not pd.isna(luteal_length) else 14), 
                                 fertile_start + fertile_days - 1)
            else:
                # If no fertile window data, we can only determine menstrual and possibly luteal
                # Return NaN for follicular/fertile distinction
                if not pd.isna(luteal_length):
                    luteal_start = cycle_length - int(luteal_length) + 1
                    if day >= luteal_start:
                        return 'luteal'
                    else:
                        return 'follicular'  # Best guess
                else:
                    # Not enough data to determine phase beyond menstrual
                    return 'follicular'  # Conservative estimate
        
        # Luteal phase (end of cycle)
        if not pd.isna(luteal_length):
            luteal_start = cycle_length - int(luteal_length) + 1
        else:
            luteal_start = fertile_end + 1
        
        # Assign phase based on day
        if day <= menses_length:
            return 'menstrual'
        elif day >= fertile_start and day <= fertile_end:
            return 'fertile'
        elif day >= luteal_start:
            return 'luteal'
        else:
            return 'follicular'
    
    def create_user_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unique user IDs and forward-fill user characteristics"""
        
        # First, we need to identify which records belong to the same user
        # This is tricky since we don't have explicit user IDs
        # We'll use a combination of demographic data and sequential ordering
        
        print("🔄 Identifying users and propagating characteristics...")
        
        # Sort by potential user grouping indicators
        # Assuming data is generally grouped by user in the original file
        df = df.reset_index(drop=True)
        
        # Create user groups based on when demographic data appears
        user_chars = ['Age', 'Maristatus', 'Schoolyears', 'BMI', 'Medvits', 'Medvitexplain']
        available_chars = [col for col in user_chars if col in df.columns]
        
        # Initialize user_id column
        df['user_id'] = None
        current_user_id = 1
        current_user_data = {}
        
        for idx, row in df.iterrows():
            # Check if this row has new user demographic data
            has_new_demographics = False
            new_user_data = {}
            
            for col in available_chars:
                value = row[col]
                if not pd.isna(value):
                    new_user_data[col] = value
                    # If this is different from current user data, it might be a new user
                    if col in current_user_data and current_user_data[col] != value:
                        has_new_demographics = True
            
            # If we have new demographic data and it's different, start new user
            if new_user_data and (not current_user_data or has_new_demographics):
                current_user_data = new_user_data.copy()
                current_user_id += 1
            
            # Assign user ID
            df.at[idx, 'user_id'] = f"user_{current_user_id:03d}"
            
            # Forward fill the user characteristics for this user
            for col in available_chars:
                if col in current_user_data:
                    df.at[idx, col] = current_user_data[col]
        
        # Alternative approach: group by demographic similarity within sequential blocks
        df = self._refine_user_grouping(df, available_chars)
        
        return df
    
    def _refine_user_grouping(self, df: pd.DataFrame, user_chars: List[str]) -> pd.DataFrame:
        """Refine user grouping and ensure characteristics are properly propagated"""
        
        # Group consecutive rows with similar demographic patterns
        user_groups = []
        current_group = []
        current_demographics = {}
        
        for idx, row in df.iterrows():
            # Extract non-null demographics from this row
            row_demographics = {}
            for col in user_chars:
                if not pd.isna(row[col]):
                    row_demographics[col] = row[col]
            
            # If this row has demographics data
            if row_demographics:
                # Check if it matches current group
                if current_demographics and self._demographics_match(current_demographics, row_demographics):
                    # Same user, add to current group
                    current_group.append(idx)
                else:
                    # New user detected
                    if current_group:
                        user_groups.append((current_group.copy(), current_demographics.copy()))
                    current_group = [idx]
                    current_demographics = row_demographics.copy()
            else:
                # No demographics, assume continuation of current user
                if current_group:
                    current_group.append(idx)
                else:
                    # Orphaned record - create single-record group
                    user_groups.append(([idx], {}))
        
        # Add the last group
        if current_group:
            user_groups.append((current_group, current_demographics))
        
        # Assign user IDs and propagate characteristics
        for user_num, (indices, demographics) in enumerate(user_groups, 1):
            user_id = f"user_{user_num:03d}"
            for idx in indices:
                df.at[idx, 'user_id'] = user_id
                # Propagate all characteristics to all records of this user
                for col, value in demographics.items():
                    df.at[idx, col] = value
        
        print(f"✅ Identified {len(user_groups)} unique users")
        return df
    
    def _demographics_match(self, demo1: Dict, demo2: Dict) -> bool:
        """Check if two demographic profiles match (same user)"""
        # Get common keys
        common_keys = set(demo1.keys()) & set(demo2.keys())
        
        if not common_keys:
            return True  # No common data to compare
        
        # Check if all common values match
        for key in common_keys:
            if demo1[key] != demo2[key]:
                return False
        
        return True
    
    def transform_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform cycle data to daily records"""
        print("🔄 Transforming cycles to daily records...")
        
        # Add user IDs
        df = self.create_user_id(df)
        
        daily_records = []
        
        for idx, row in df.iterrows():
            cycle_length = self.clean_numeric_value(row.get('LengthofCycle'))
            
            # Skip cycles where we don't have basic length information
            if pd.isna(cycle_length) or cycle_length <= 0:
                print(f"⚠️ Skipping cycle {row.get('CycleNumber', 'unknown')} - missing/invalid cycle length")
                continue
                
            cycle_length = int(cycle_length)
            
            # Create daily records for this cycle
            for day in range(1, cycle_length + 1):
                # Calculate phase, preserving NaN when data is insufficient
                phase = self.calculate_cycle_phase(
                    day=day,
                    cycle_length=cycle_length,
                    menses_length=row.get('LengthofMenses'),
                    fertile_days=row.get('TotalDaysofFertility'),
                    luteal_length=row.get('LengthofLutealPhase'),
                    ovulation_day=row.get('EstimatedDayofOvulation')
                )
                
                daily_record = {
                    'user_id': row['user_id'],
                    'cycle_number': self.clean_numeric_value(row.get('CycleNumber')),
                    'cycle_day': day,
                    'cycle_phase': phase,
                    'cycle_length': cycle_length
                }
                
                # Add transferred columns - preserving NaN/empty values
                for col in self.columns_to_transfer:
                    value = row.get(col)
                    # Clean and preserve NaN values properly
                    if col in ['Age', 'BMI', 'Schoolyears']:  # Numeric columns
                        daily_record[col] = self.clean_numeric_value(value)
                    else:  # String columns
                        if pd.isna(value) or str(value).strip() in ['', ' ', 'nan']:
                            daily_record[col] = np.nan
                        else:
                            daily_record[col] = str(value).strip()
                
                # Add some additional useful features
                daily_record['days_from_cycle_start'] = day - 1
                daily_record['days_to_cycle_end'] = cycle_length - day
                
                # Phase encoding for ML - handle NaN phases
                phase_encoding = {
                    'menstrual': 1, 'follicular': 2, 
                    'fertile': 3, 'luteal': 4
                }
                if pd.isna(phase):
                    daily_record['phase_encoded'] = np.nan
                else:
                    daily_record['phase_encoded'] = phase_encoding.get(phase, np.nan)
                
                daily_records.append(daily_record)
        
        daily_df = pd.DataFrame(daily_records)
        print(f"✅ Created {len(daily_df)} daily records from {len(df)} cycles")
        
        return daily_df
    
    def add_statistical_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features for each user"""
        print("🔄 Adding statistical features...")
        
        # Calculate user-level statistics - handle cases where users might have different data availability
        user_stats_list = []
        
        for user_id in daily_df['user_id'].unique():
            user_data = daily_df[daily_df['user_id'] == user_id]
            
            # Calculate cycle statistics
            cycle_lengths = user_data['cycle_length'].unique()
            cycle_count = user_data['cycle_number'].nunique() if 'cycle_number' in user_data.columns else len(cycle_lengths)
            
            user_stat = {
                'user_id': user_id,
                'avg_cycle_length': np.mean(cycle_lengths) if len(cycle_lengths) > 0 else np.nan,
                'cycle_length_variation': np.std(cycle_lengths) if len(cycle_lengths) > 1 else 0.0,
                'min_cycle_length': np.min(cycle_lengths) if len(cycle_lengths) > 0 else np.nan,
                'max_cycle_length': np.max(cycle_lengths) if len(cycle_lengths) > 0 else np.nan,
                'total_cycles': cycle_count
            }
            user_stats_list.append(user_stat)
        
        user_stats_df = pd.DataFrame(user_stats_list)
        
        # Merge back to daily data
        daily_df = daily_df.merge(user_stats_df, on='user_id', how='left')
        
        print("✅ Added statistical features")
        return daily_df
    
    def save_data(self, daily_df: pd.DataFrame):
        """Save transformed data and create summary"""
        
        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset - preserve NaN values properly
        daily_df.to_csv(self.output_file, index=False, na_rep='')
        print(f"✅ Saved daily data to: {self.output_file}")
        
        # Create and save summary
        summary_file = self.output_file.parent / "daily_data_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("DAILY CYCLE DATA SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total daily records: {len(daily_df):,}\n")
            f.write(f"Unique users: {daily_df['user_id'].nunique()}\n")
            
            # Handle potential NaN in cycle_number
            cycles_per_user = daily_df.groupby('user_id')['cycle_number'].max()
            valid_cycles = cycles_per_user.dropna()
            if len(valid_cycles) > 0:
                f.write(f"Average cycles per user: {valid_cycles.mean():.1f}\n\n")
            else:
                f.write("Average cycles per user: N/A (cycle numbers missing)\n\n")
            
            f.write("PHASE DISTRIBUTION:\n")
            phase_dist = daily_df['cycle_phase'].value_counts(dropna=False)
            for phase, count in phase_dist.items():
                percentage = (count / len(daily_df)) * 100
                phase_name = 'Missing/Unknown' if pd.isna(phase) else phase
                f.write(f"  {phase_name}: {count:,} ({percentage:.1f}%)\n")
            
            f.write(f"\nCYCLE LENGTH STATISTICS:\n")
            f.write(f"  Mean: {daily_df['cycle_length'].mean():.1f} days\n")
            f.write(f"  Std: {daily_df['cycle_length'].std():.1f} days\n")
            f.write(f"  Range: {daily_df['cycle_length'].min()}-{daily_df['cycle_length'].max()} days\n")
            
            f.write(f"\nMISSING DATA SUMMARY:\n")
            for col in self.columns_to_transfer + ['cycle_phase']:
                if col in daily_df.columns:
                    missing_count = daily_df[col].isna().sum()
                    missing_pct = (missing_count / len(daily_df)) * 100
                    f.write(f"  {col}: {missing_count:,} missing ({missing_pct:.1f}%)\n")
            
            f.write(f"\nCOLUMNS IN OUTPUT:\n")
            for col in daily_df.columns:
                f.write(f"  - {col}\n")
        
        print(f"✅ Saved summary to: {summary_file}")
        
        # Show sample of data
        print("\n📊 Sample of transformed data:")
        print(daily_df.head(10).to_string())
        
        print(f"\n📈 Data shape: {daily_df.shape}")
        print(f"📈 Phase distribution:")
        print(daily_df['cycle_phase'].value_counts(dropna=False))
    
    def run_transformation(self):
        """Run complete transformation pipeline"""
        print("🚀 Starting cycle to daily transformation...\n")
        
        # Load data
        df = self.load_data()
        
        # Transform to daily records
        daily_df = self.transform_to_daily(df)
        
        # Add statistical features
        daily_df = self.add_statistical_features(daily_df)
        
        # Save results
        self.save_data(daily_df)
        
        print(f"\n🎉 Transformation completed successfully!")
        print(f"📁 Output saved to: {self.output_file}")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform cycle data to daily records')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', '-o', help='Path to output CSV file')
    parser.add_argument('--preview', '-p', action='store_true', 
                       help='Preview transformation without saving')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    transformer = CycleToDailyTransformer(args.input_file, args.output)
    
    if args.preview:
        print("👁️ Preview mode - no files will be saved")
        df = transformer.load_data()
        sample_df = transformer.transform_to_daily(df.head(5))  # Just preview first 5 cycles
        print("\n📊 Preview of transformed data:")
        print(sample_df.head(20).to_string())
    else:
        transformer.run_transformation()

if __name__ == "__main__":
    main()