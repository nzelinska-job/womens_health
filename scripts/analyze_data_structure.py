#!/usr/bin/env python3
"""
Analyzer for raw cycle data structure
Helps understand user grouping patterns and data distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_data_structure(file_path: str):
    """Analyze the structure of raw cycle data"""
    
    print(f"🔍 Analyzing data structure: {file_path}\n")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} records")
        print(f"📊 Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Key user characteristics to analyze
    user_chars = ['Age', 'Maristatus', 'Schoolyears', 'BMI', 'Medvits', 'Medvitexplain']
    available_chars = [col for col in user_chars if col in df.columns]
    
    print(f"\n📋 Available user characteristics: {available_chars}")
    
    # Analyze missing patterns for user characteristics
    print(f"\n🔍 MISSING DATA PATTERN ANALYSIS:")
    print("=" * 50)
    
    for col in available_chars:
        total_count = len(df)
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        
        print(f"{col}:")
        print(f"  Non-null values: {non_null_count} ({non_null_count/total_count*100:.1f}%)")
        print(f"  Null values: {null_count} ({null_count/total_count*100:.1f}%)")
        
        # Analyze distribution of non-null values
        if non_null_count > 0:
            non_null_indices = df[df[col].notna()].index.tolist()
            gaps = []
            if len(non_null_indices) > 1:
                gaps = [non_null_indices[i+1] - non_null_indices[i] for i in range(len(non_null_indices)-1)]
                avg_gap = np.mean(gaps)
                print(f"  Average gap between non-null values: {avg_gap:.1f} rows")
                print(f"  Max gap: {max(gaps)} rows")
        print()
    
    # Analyze potential user groupings
    print(f"🧩 USER GROUPING ANALYSIS:")
    print("=" * 50)
    
    # Look for rows where multiple user characteristics appear together
    rows_with_demographics = df[available_chars].notna().any(axis=1)
    demo_rows = df[rows_with_demographics]
    
    print(f"Rows with demographic data: {len(demo_rows)} ({len(demo_rows)/len(df)*100:.1f}%)")
    
    if len(demo_rows) > 0:
        print(f"\nFirst few rows with demographics:")
        print(demo_rows[['CycleNumber'] + available_chars].head(10).to_string())
        
        # Analyze cycle numbers for demographic rows
        if 'CycleNumber' in df.columns:
            demo_cycle_numbers = demo_rows['CycleNumber'].dropna()
            print(f"\nCycle numbers in demographic rows:")
            print(f"  Unique cycle numbers: {sorted(demo_cycle_numbers.unique())}")
            print(f"  Cycle 1 rows: {(demo_cycle_numbers == 1).sum()}")
    
    # Analyze cycle patterns
    print(f"\n📊 CYCLE ANALYSIS:")
    print("=" * 50)
    
    if 'CycleNumber' in df.columns:
        cycle_counts = df['CycleNumber'].value_counts().sort_index()
        print(f"Cycle distribution:")
        for cycle_num in sorted(cycle_counts.index)[:10]:  # Show first 10 cycles
            print(f"  Cycle {cycle_num}: {cycle_counts[cycle_num]} records")
        
        if len(cycle_counts) > 10:
            print(f"  ... and {len(cycle_counts) - 10} more cycles")
        
        print(f"\nCycle statistics:")
        print(f"  Max cycle number: {df['CycleNumber'].max()}")
        print(f"  Users with >10 cycles: {(cycle_counts > 10).sum()}")
    
    # Estimate number of users based on demographic patterns
    print(f"\n👥 USER ESTIMATION:")
    print("=" * 50)
    
    # Method 1: Count unique demographic combinations
    if available_chars:
        # Create signature for each user based on available characteristics
        demo_subset = df[available_chars].dropna(how='all')
        if len(demo_subset) > 0:
            unique_combinations = demo_subset.drop_duplicates()
            print(f"Unique demographic combinations: {len(unique_combinations)}")
    
    # Method 2: Count demographic transitions
    transitions = 0
    prev_demo = None
    
    for idx, row in df.iterrows():
        current_demo = {}
        for col in available_chars:
            if not pd.isna(row[col]):
                current_demo[col] = row[col]
        
        if current_demo:
            if prev_demo and current_demo != prev_demo:
                transitions += 1
            prev_demo = current_demo
    
    estimated_users = transitions + 1 if transitions > 0 else 1
    print(f"Estimated users (based on transitions): {estimated_users}")
    
    # Show potential user grouping
    print(f"\n🔗 POTENTIAL USER GROUPS (first 50 rows):")
    print("=" * 50)
    
    display_cols = ['CycleNumber'] + available_chars[:3]  # Limit columns for readability
    available_display_cols = [col for col in display_cols if col in df.columns]
    
    print(df[available_display_cols].head(50).to_string())
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    print("1. Demographics appear to be filled only for first occurrence of each user")
    print("2. Use forward-fill strategy to propagate user characteristics")
    print("3. Group consecutive cycles with similar demographic patterns")
    if 'CycleNumber' in df.columns:
        cycle1_with_demo = len(df[(df['CycleNumber'] == 1) & rows_with_demographics])
        print(f"4. Found {cycle1_with_demo} cycle 1 records with demographics")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze raw cycle data structure')
    parser.add_argument('file_path', help='Path to CSV file to analyze')
    
    args = parser.parse_args()
    
    if not Path(args.file_path).exists():
        print(f"❌ File not found: {args.file_path}")
        sys.exit(1)
    
    analyze_data_structure(args.file_path)

if __name__ == "__main__":
    main()