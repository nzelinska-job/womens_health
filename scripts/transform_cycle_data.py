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

# Set random seed for reproducible results
np.random.seed(42)
warnings.filterwarnings('ignore')


class CycleToDailyTransformer:
    """Transforms cycle-level data to daily records with phase calculation"""

    def __init__(self, input_file: str, output_file: str = None):
        self.input_file = Path(input_file)
        self.output_file = Path(
            output_file) if output_file else self.input_file.parent / "daily_cycle_data.csv"

        # Columns to transfer from original data
        self.columns_to_transfer = [
            'ClientID', 'Age', 'Maristatus', 'Schoolyears', 'BMI',
            'Breastfeeding'  # , 'Medvitexplain'
        ]

        # Columns needed for phase calculation
        self.phase_columns = [
            'LengthofCycle', 'LengthofMenses', 'TotalDaysofFertility',
            'LengthofLutealPhase', 'EstimatedDayofOvulation'
        ]

        # For tracking processed users
        self.processed_users = set()

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
            missing_cols = [
                col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"⚠️ Missing critical columns: {missing_cols}")
                # Only fail if we're missing absolutely critical columns
                critical_missing = [col for col in [
                    'LengthofCycle'] if col in missing_cols]
                if critical_missing:
                    print(f"❌ Cannot proceed without: {critical_missing}")
                    sys.exit(1)

            # Check for user characteristic columns
            missing_user_cols = [
                col for col in self.columns_to_transfer if col not in df.columns]
            if missing_user_cols:
                print(
                    f"⚠️ Missing user characteristic columns: {missing_user_cols}")
                for col in missing_user_cols:
                    df[col] = np.nan
                    print(f"   Added missing column '{col}' with NaN values")

            # Clean user characteristic columns as well
            for col in self.columns_to_transfer:
                if col in df.columns:
                    # For string columns, just clean whitespace
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
                        df[col] = df[col].replace(
                            ['', ' ', '  ', 'nan'], np.nan)

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

        Phases:
        - m: Menstrual (Days 1 to menses_length)
        - f: Follicular (After menses until fertile window)
        - o: Ovulation (Around ovulation day)
        - l: Luteal (After ovulation until end of cycle)
        """

        # Clean all input values
        cycle_length = self.clean_numeric_value(cycle_length)
        menses_length = self.clean_numeric_value(menses_length)
        fertile_days = self.clean_numeric_value(fertile_days)
        luteal_length = self.clean_numeric_value(luteal_length)
        ovulation_day = self.clean_numeric_value(ovulation_day)

        # Handle missing values
        if pd.isna(cycle_length):
            return np.nan
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
            return 'm'

        # Determine ovulation day and window
        if not pd.isna(ovulation_day):
            ovulation_day = int(ovulation_day)
        else:
            # Estimate ovulation day if missing
            if not pd.isna(luteal_length):
                ovulation_day = cycle_length - int(luteal_length) + 1
            else:
                # Default estimation: around day 14 or mid-cycle
                ovulation_day = min(14, cycle_length // 2 + 3)

        # Ovulation phase (1-2 days around ovulation)
        ovulation_start = max(menses_length + 1, ovulation_day - 1)
        ovulation_end = min(cycle_length - 1, ovulation_day + 1)

        if day >= ovulation_start and day <= ovulation_end:
            return 'o'

        # Luteal phase (after ovulation until end)
        if not pd.isna(luteal_length):
            luteal_start = cycle_length - int(luteal_length) + 1
        else:
            luteal_start = ovulation_end + 1

        if day >= luteal_start:
            return 'l'

        # Follicular phase (after menstruation, before ovulation)
        return 'f'

    def generate_mood(self, phase: str, cycle_day: int, cycle_length: int, age: float) -> int:
        """Generate mood based on cycle phase and age"""
        if pd.isna(phase) or pd.isna(age):
            return np.random.choice([-2, -1, 0, 1, 2])

        # Age factor for PMS (increases with age)
        if age <= 20:
            pms_intensity = 0.036  # 3.6%
        elif age <= 25:
            pms_intensity = 0.042
        elif age <= 30:
            pms_intensity = 0.048
        else:
            pms_intensity = 0.054  # 5.4%

        # Determine if in pre-menstrual period (last 7 days of luteal phase)
        is_premenstrual = False
        if phase == 'l':
            # Estimate when luteal phase started
            luteal_days_remaining = cycle_length - cycle_day
            if luteal_days_remaining <= 7:
                is_premenstrual = True

        if is_premenstrual:
            # Pre-menstrual: mostly negative (75%) with some positive (25%)
            if np.random.random() < 0.75:
                return np.random.choice([-2, -1, 0], p=[0.3, 0.4, 0.3])
            else:
                return np.random.choice([1, 2], p=[0.7, 0.3])
        elif phase in ['m', 'f']:
            # Menstrual and follicular: mostly positive (75%)
            if np.random.random() < 0.75:
                return np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            else:
                return np.random.choice([-2, -1], p=[0.3, 0.7])
        else:
            # Ovulation and other luteal days: balanced
            return np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])

    def generate_energy(self, phase: str) -> int:
        """Generate energy level (-2 to 2)"""
        if pd.isna(phase):
            return np.random.choice([-2, -1, 0, 1, 2])

        if phase == 'm':
            # Menstrual: lower energy
            return np.random.choice([-2, -1, 0, 1, 2], p=[0.3, 0.3, 0.3, 0.1, 0.0])
        elif phase == 'f':
            # Follicular: increasing energy
            return np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        elif phase == 'o':
            # Ovulation: high energy
            return np.random.choice([-2, -1, 0, 1, 2], p=[0.0, 0.1, 0.2, 0.4, 0.3])
        else:  # luteal
            # Luteal: decreasing energy
            return np.random.choice([-2, -1, 0, 1, 2], p=[0.2, 0.3, 0.3, 0.2, 0.0])

    def generate_symptoms(self, phase: str, cycle_day: int, cycle_length: int) -> str:
        """Generate symptoms based on cycle phase"""
        symptoms_list = ['cramps', 'headache', 'breast_tenderness', 'acne',
                         'food_cravings', 'sleep_problems', 'not_defined']

        if pd.isna(phase):
            return np.random.choice(symptoms_list)

        # Determine if in pre-menstrual or menstrual period
        is_premenstrual = False
        if phase == 'l':
            luteal_days_remaining = cycle_length - cycle_day
            if luteal_days_remaining <= 7:
                is_premenstrual = True

        is_high_symptom_period = (phase == 'm') or is_premenstrual

        if is_high_symptom_period:
            # 75% chance of symptoms during menstrual/pre-menstrual
            if np.random.random() < 0.75:
                # 50% chance of multiple symptoms
                if np.random.random() < 0.5:
                    # Multiple symptoms
                    if phase == 'm':
                        possible_symptoms = [
                            'cramps', 'headache', 'sleep_problems']
                        num_symptoms = np.random.choice([2, 3], p=[0.7, 0.3])
                        selected_symptoms = np.random.choice(possible_symptoms,
                                                             size=min(num_symptoms, len(
                                                                 possible_symptoms)),
                                                             replace=False)
                        return ', '.join(selected_symptoms)
                    else:  # pre-menstrual
                        possible_symptoms = [
                            'breast_tenderness', 'food_cravings', 'acne', 'headache']
                        num_symptoms = np.random.choice([2, 3], p=[0.7, 0.3])
                        selected_symptoms = np.random.choice(possible_symptoms,
                                                             size=min(num_symptoms, len(
                                                                 possible_symptoms)),
                                                             replace=False)
                        return ', '.join(selected_symptoms)
                else:
                    # Single symptom
                    if phase == 'm':
                        return np.random.choice(['cramps', 'headache', 'sleep_problems'],
                                                p=[0.5, 0.3, 0.2])
                    else:  # pre-menstrual
                        return np.random.choice(['breast_tenderness', 'food_cravings', 'acne'],
                                                p=[0.4, 0.4, 0.2])
            else:
                return 'not_defined'
        else:
            # 25% chance during other phases
            if np.random.random() < 0.25:
                # 50% chance of multiple symptoms even in other phases
                if np.random.random() < 0.5:
                    possible_symptoms = [
                        s for s in symptoms_list if s != 'not_defined']
                    num_symptoms = np.random.choice([2, 3], p=[0.8, 0.2])
                    selected_symptoms = np.random.choice(possible_symptoms,
                                                         size=min(num_symptoms, len(
                                                             possible_symptoms)),
                                                         replace=False)
                    return ', '.join(selected_symptoms)
                else:
                    # Exclude 'not_defined'
                    return np.random.choice(symptoms_list[:-1])
            else:
                return 'not_defined'

    def generate_stress_level(self, mood: int) -> int:
        """Generate stress level correlated with mood (75% correlation)"""
        if pd.isna(mood):
            return np.random.choice([-3, -2, -1, 0])

        # 75% correlation with mood
        if np.random.random() < 0.75:
            # Correlated: bad mood = high stress
            if mood <= -1:
                return np.random.choice([-3, -2], p=[0.6, 0.4])
            elif mood == 0:
                return np.random.choice([-2, -1, 0], p=[0.2, 0.3, 0.5])
            else:  # mood >= 1
                return np.random.choice([-1, 0], p=[0.3, 0.7])
        else:
            # 25% random
            return np.random.choice([-3, -2, -1, 0], p=[0.2, 0.3, 0.3, 0.2])

    def create_user_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create unique user IDs and forward-fill user characteristics"""

        # First, we need to identify which records belong to the same user
        # This is tricky since we don't have explicit user IDs
        # We'll use a combination of demographic data and sequential ordering

        print("🔄 Identifying users and propagating characteristics...")

        # Check if ClientID is the index, if so, make it a column
        if df.index.name == 'ClientID' or 'ClientID' in str(df.index.name):
            df = df.reset_index()  # This will make the index a column named 'ClientID'
            # Store the current row position as original_index
            df['original_index'] = df.index
        else:
            # Sort by potential user grouping indicators
            # Assuming data is generally grouped by user in the original file
            df = df.reset_index(drop=False)  # Keep original index as a column
            # Rename to avoid confusion
            df = df.rename(columns={'index': 'original_index'})

        # Create user groups based on when demographic data appears
        user_chars = ['ClientID', 'Age', 'Maristatus', 'Schoolyears',
                      'BMI', 'Breastfeeding']  # , 'Medvitexplain']
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
                        user_groups.append(
                            (current_group.copy(), current_demographics.copy()))
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

        # Store original dataframe info for user tracking
        self.original_df_size = len(df)
        self.processed_indices = []  # Track which ClientIDs we actually process

        # Add user IDs and preserve original index
        df = self.create_user_id(df)

        daily_records = []

        for idx, row in df.iterrows():
            cycle_length = self.clean_numeric_value(row.get('LengthofCycle'))

            # Skip cycles where we don't have basic length information
            if pd.isna(cycle_length) or cycle_length <= 0:
                print(
                    f"⚠️ Skipping cycle {row.get('CycleNumber', 'unknown')} for ClientID {row.get('ClientID', 'unknown')} - missing/invalid cycle length")
                continue

            # Track that we processed this ClientID
            client_id = row.get('ClientID')
            if client_id is not None and client_id not in self.processed_indices:
                self.processed_indices.append(client_id)

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
                    'user_id': f"{row['user_id']}_{self.clean_numeric_value(row.get('CycleNumber', 1)) or 1:.0f}",
                    # Add original index from input file
                    'original_index': row.get('original_index'),
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
                    elif col == 'ClientID':  # ClientID - handle as index value
                        # ClientID is now a regular column (converted from index)
                        if pd.isna(value):
                            daily_record[col] = np.nan
                        else:
                            # Keep ClientID as its original type (likely int or string)
                            daily_record[col] = value
                    else:  # String columns
                        if pd.isna(value) or str(value).strip() in ['', ' ', 'nan']:
                            daily_record[col] = np.nan
                        else:
                            daily_record[col] = str(value).strip()

                # Generate psychological and physical data
                age = daily_record.get('Age', 25)  # Default age if missing

                mood = self.generate_mood(phase, day, cycle_length, age)
                energy = self.generate_energy(phase)
                symptoms = self.generate_symptoms(phase, day, cycle_length)
                stress = self.generate_stress_level(mood)

                daily_record['mood'] = mood
                daily_record['energy'] = energy
                daily_record['symptoms'] = symptoms
                daily_record['stress_level'] = stress

                # Phase encoding for ML
                phase_encoding = {
                    'm': 1, 'f': 2, 'o': 3, 'l': 4
                }
                if pd.isna(phase):
                    daily_record['phase_encoded'] = np.nan
                else:
                    daily_record['phase_encoded'] = phase_encoding.get(
                        phase, np.nan)

                daily_records.append(daily_record)

        # Track processed users
        processed_user_ids = set(df['user_id'].unique())
        self.processed_users.update(processed_user_ids)

        daily_df = pd.DataFrame(daily_records)
        print(f"✅ Created {len(daily_df)} daily records from {len(df)} cycles")

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

            # Use ClientID tracking instead of indices
            total_original_records = getattr(self, 'original_df_size', 0)
            processed_client_ids = getattr(
                self, 'processed_indices', [])  # Now contains ClientIDs

            f.write(f"Total records in input file: {total_original_records}\n")
            f.write(
                f"Successfully processed records: {len(processed_client_ids)}\n")
            f.write(
                f"Skipped records: {total_original_records - len(processed_client_ids)}\n")

            f.write(f"\nPROCESSED CLIENT IDs FROM INPUT FILE:\n")
            f.write("-" * 50 + "\n")

            # Write the actual processed ClientIDs from the original file
            if processed_client_ids:
                # Sort ClientIDs for better readability (handle both numeric and string IDs)
                try:
                    # Try to sort as numbers if all are numeric
                    sorted_client_ids = sorted([cid for cid in processed_client_ids if pd.notna(cid)],
                                               key=lambda x: float(x) if str(x).replace('.', '').replace('-', '').isdigit() else float('inf'))
                except (ValueError, TypeError):
                    # Sort as strings if not all numeric
                    sorted_client_ids = sorted([cid for cid in processed_client_ids if pd.notna(cid)],
                                               key=lambda x: str(x))

                # Write in chunks of 20 for readability
                for i in range(0, len(sorted_client_ids), 20):
                    chunk = sorted_client_ids[i:i+20]
                    f.write(f"{chunk}\n")

                f.write(
                    f"\nTotal processed ClientIDs: {len(sorted_client_ids)}\n")
            else:
                f.write("No ClientIDs were successfully processed.\n")
            f.write(f"\nPHASE DISTRIBUTION:\n")
            phase_dist = daily_df['cycle_phase'].value_counts(dropna=False)
            phase_names = {'m': 'Menstrual', 'f': 'Follicular',
                           'o': 'Ovulation', 'l': 'Luteal'}
            for phase, count in phase_dist.items():
                percentage = (count / len(daily_df)) * 100
                phase_name = phase_names.get(
                    phase, 'Missing/Unknown') if pd.notna(phase) else 'Missing/Unknown'
                f.write(f"  {phase_name}: {count:,} ({percentage:.1f}%)\n")

            f.write(f"\nCYCLE LENGTH STATISTICS:\n")
            f.write(f"  Mean: {daily_df['cycle_length'].mean():.1f} days\n")
            f.write(f"  Std: {daily_df['cycle_length'].std():.1f} days\n")
            f.write(
                f"  Range: {daily_df['cycle_length'].min()}-{daily_df['cycle_length'].max()} days\n")

            f.write(f"\nPSYCHOLOGICAL & PHYSICAL DATA:\n")
            f.write(
                f"  Mood distribution: {dict(daily_df['mood'].value_counts().sort_index())}\n")
            f.write(
                f"  Energy distribution: {dict(daily_df['energy'].value_counts().sort_index())}\n")
            f.write(
                f"  Stress distribution: {dict(daily_df['stress_level'].value_counts().sort_index())}\n")

            symptoms_dist = daily_df['symptoms'].value_counts()
            f.write(f"  Top symptoms: {dict(symptoms_dist.head())}\n")

            f.write(f"\nMISSING DATA SUMMARY:\n")
            for col in self.columns_to_transfer + ['cycle_phase', 'mood', 'energy', 'symptoms', 'stress_level']:
                if col in daily_df.columns:
                    missing_count = daily_df[col].isna().sum()
                    missing_pct = (missing_count / len(daily_df)) * 100
                    f.write(
                        f"  {col}: {missing_count:,} missing ({missing_pct:.1f}%)\n")

            f.write(f"\nCOLUMNS IN OUTPUT:\n")
            for col in daily_df.columns:
                f.write(f"  - {col}\n")

        print(f"✅ Saved summary to: {summary_file}")

        # Show sample of data
        print("\n📊 Sample of transformed data:")
        print(daily_df.head(10).to_string())

        print(f"\n📈 Data shape: {daily_df.shape}")
        print(f"📈 Phase distribution:")
        phase_names = {'m': 'Menstrual', 'f': 'Follicular',
                       'o': 'Ovulation', 'l': 'Luteal'}
        phase_counts = daily_df['cycle_phase'].value_counts(dropna=False)
        for phase, count in phase_counts.items():
            phase_name = phase_names.get(
                phase, 'Unknown') if pd.notna(phase) else 'Missing'
            print(f"  {phase_name}: {count}")

        print(f"\n🧠 Psychological data sample:")
        print(
            f"  Mood range: {daily_df['mood'].min()} to {daily_df['mood'].max()}")
        print(
            f"  Energy range: {daily_df['energy'].min()} to {daily_df['energy'].max()}")
        print(
            f"  Stress range: {daily_df['stress_level'].min()} to {daily_df['stress_level'].max()}")
        print(
            f"  Most common symptom: {daily_df['symptoms'].value_counts().index[0]}")

    def run_transformation(self):
        """Run complete transformation pipeline"""
        print("🚀 Starting cycle to daily transformation...\n")

        # Load data
        df = self.load_data()

        # Transform to daily records
        daily_df = self.transform_to_daily(df)

        # Save results
        self.save_data(daily_df)

        print(f"\n🎉 Transformation completed successfully!")
        print(f"📁 Output saved to: {self.output_file}")


def main():
    """Main function with command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Transform cycle data to daily records')
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
        sample_df = transformer.transform_to_daily(
            df.head(5))  # Just preview first 5 cycles
        print("\n📊 Preview of transformed data:")
        print(sample_df.head(20).to_string())
    else:
        transformer.run_transformation()


if __name__ == "__main__":
    main()
