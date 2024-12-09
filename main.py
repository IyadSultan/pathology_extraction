#!/usr/bin/env python3
# main_extractor.py

"""
This script processes pathology reports to extract structured information:
- Loads reference data for cancer types, ICD-O codes, and stains
- Initializes dependencies for the extraction process
- Reads pathology reports from a CSV file
- Processes reports in batches using async functions
- Extracts key information like diagnosis, staging, margins etc.
- Matches findings to standardized ICD-O codes and stain names
- Saves extracted data to CSV with error handling
- Provides summary statistics of the extraction results

The extracted data includes:
- Patient identifiers (MRN, Document Number)
- Dates
- Histologic diagnoses and ICD-O codes
- Staging information
- Microscopic findings
- IHC stain results
"""

import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
from extractor import (
    process_pathology_report, 
    ExtractionDependencies,
    DiagnosisMapper,
    StainMapper
)
from typing import Dict, Any, List, Optional

warnings.filterwarnings('ignore')

async def process_batch(reports_df: pd.DataFrame,
                       deps: ExtractionDependencies,
                       start_idx: int = 0,
                       batch_size: int = 200) -> List[Dict[str, Any]]:
    """
    Process a batch of pathology reports and return results
    """
    results = []
    end_idx = min(start_idx + batch_size, len(reports_df))
    
    for idx in tqdm(range(start_idx, end_idx), desc="Processing reports"):
        row = reports_df.iloc[idx]
        try:
            # Process the report
            result = await process_pathology_report(row['Note'], deps)
            
            # Combine with original metadata
            extracted_data = {
                'MRN': row['MRN'],
                'Document_Number': row['Document_Number'],
                'Entry_Date': row['Entry_Date'],
                **result.dict()
            }
            results.append(extracted_data)
            
        except Exception as e:
            print(f"Error processing report {idx} (MRN: {row['MRN']}): {str(e)}")
            # Add error record
            error_data = {
                'MRN': row['MRN'],
                'Document_Number': row['Document_Number'],
                'Entry_Date': row['Entry_Date'],
                'error': str(e)
            }
            results.append(error_data)
    
    return results

async def main():
    print("Loading reference data...")
    try:
        # Load ICDO3 reference data
        icdo_df = pd.read_csv('resources/ICDO3Terminology.csv')
        print(f"Loaded {len(icdo_df)} terminology codes")
        print("Sample terminology entries:")
        print(icdo_df.head())
        
        topography_df = pd.read_csv('resources/ICDO3Topography.csv')
        print(f"\nLoaded {len(topography_df)} topography codes")
        print("Sample topography entries:")
        print(topography_df.head())
        
        # Load stains reference data
        stains_df = pd.read_csv('resources/stains.csv')
        print(f"\nLoaded {len(stains_df)} stains")
        print("Sample stains entries:")
        print(stains_df.head())
        
        # Cancer types (could also be loaded from a file if needed)
        cancer_special_items = pd.DataFrame({'Cancer': ['Breast Cancer', 'Lung Cancer']})

        deps = ExtractionDependencies(
            cancer_types=list(cancer_special_items['Cancer'].unique()),
            special_items_df=cancer_special_items,
            diagnosis_mapper=DiagnosisMapper(icdo_df, topography_df),
            stain_mapper=StainMapper(stains_df)
        )

        # Load pathology reports
        print("\nLoading pathology reports...")
        reports_df = pd.read_csv(r"E:\Dropbox\AI\Data\pathology_1000_reports.csv")
        
        # Process reports in batch
        results = await process_batch(reports_df, deps, batch_size=200)
        
        # Convert results to DataFrame
        print("Converting results to DataFrame...")
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = "path_extraction_results_200.csv"
        print(f"Saving results to {output_path}...")
        results_df.to_csv(output_path, index=False)
        print("Extraction complete!")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total reports processed: {len(results_df)}")
        print(f"Reports with errors: {results_df['error'].notna().sum() if 'error' in results_df.columns else 0}")
        if 'malignancy' in results_df.columns:
            print("\nMalignancy distribution:")
            print(results_df['malignancy'].value_counts())

    except FileNotFoundError as e:
        print(f"Error: Could not find reference file - {e}")
        print("Please ensure the following files exist in the 'resources' directory:")
        print("- ICDO3Terminology.csv")
        print("- ICDO3Topography.csv")
        print("- stains.csv")
        raise
    except Exception as e:
        print(f"Error loading reference data: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())