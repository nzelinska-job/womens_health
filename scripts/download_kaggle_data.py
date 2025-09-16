#!/usr/bin/env python3
"""
Script for downloading datasets from Kaggle
Automatically downloads and organizes datasets in the project structure
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import kagglehub
except ImportError:
    print("❌ kagglehub not installed. Run: pip install kagglehub")
    sys.exit(1)

class KaggleDatasetDownloader:
    """Handles Kaggle dataset downloads with proper organization"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = self._find_project_root()
        self.config_path = config_path or self.project_root / "config" / "datasets.yaml"
        self.raw_data_path = self.project_root / "data" / "raw"
        self.config = self._load_config()
    
    def _find_project_root(self) -> Path:
        """Find project root directory by looking for config.yaml"""
        current = Path.cwd()
        
        # Look for project indicators
        indicators = ["config.yaml", "setup.py", "requirements.txt"]
        
        for parent in [current] + list(current.parents):
            if any((parent / indicator).exists() for indicator in indicators):
                return parent
        
        # If not found, assume current directory
        return current
    
    def _load_config(self) -> Dict[str, Any]:
        """Load dataset configuration"""
        if not self.config_path.exists():
            # Create default config if it doesn't exist
            self._create_default_config()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self):
        """Create default dataset configuration"""
        default_config = {
            'datasets': {
                'menstrual_cycle': {
                    'kaggle_path': 'nikitabisht/menstrual-cycle-data',
                    'description': 'Menstrual cycle tracking data',
                    'local_folder': 'menstrual_cycle',
                    'files': [
                        'menstrual_cycle_data.csv'
                    ]
                },
                'fertility_data': {
                    'kaggle_path': 'example/fertility-dataset',
                    'description': 'Fertility prediction dataset',
                    'local_folder': 'fertility',
                    'files': []
                }
            },
            'download_settings': {
                'overwrite_existing': False,
                'create_metadata': True,
                'organize_by_date': False
            }
        }
        
        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ Created default config at: {self.config_path}")
    
    def download_dataset(self, dataset_name: str) -> bool:
        """Download specific dataset by name from config"""
        if dataset_name not in self.config['datasets']:
            print(f"❌ Dataset '{dataset_name}' not found in config")
            self._list_available_datasets()
            return False
        
        dataset_config = self.config['datasets'][dataset_name]
        kaggle_path = dataset_config['kaggle_path']
        local_folder = dataset_config.get('local_folder', dataset_name)
        
        return self._download_from_kaggle(kaggle_path, local_folder, dataset_config)
    
    def _download_from_kaggle(self, kaggle_path: str, local_folder: str, 
                             dataset_config: Dict[str, Any]) -> bool:
        """Download dataset from Kaggle"""
        try:
            print(f"🔄 Downloading dataset: {kaggle_path}")
            
            # Create target directory
            target_dir = self.raw_data_path / local_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if already exists and overwrite setting
            if target_dir.exists() and any(target_dir.iterdir()):
                if not self.config['download_settings'].get('overwrite_existing', False):
                    print(f"⚠️  Dataset already exists in {target_dir}")
                    response = input("Overwrite? (y/N): ").lower().strip()
                    if response != 'y':
                        print("❌ Download cancelled")
                        return False
            
            # Download using kagglehub
            download_path = kagglehub.dataset_download(kaggle_path)
            print(f"✓ Downloaded to temporary location: {download_path}")
            
            # Move files to our organized structure
            self._organize_downloaded_files(download_path, target_dir)
            
            # Create metadata if enabled
            if self.config['download_settings'].get('create_metadata', True):
                self._create_metadata_file(target_dir, kaggle_path, dataset_config)
            
            print(f"✅ Dataset '{kaggle_path}' successfully downloaded to: {target_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    
    def _organize_downloaded_files(self, source_path: str, target_dir: Path):
        """Organize downloaded files in target directory"""
        source = Path(source_path)
        
        if source.is_file():
            # Single file download
            shutil.copy2(source, target_dir / source.name)
        else:
            # Directory download - copy all contents
            for item in source.rglob('*'):
                if item.is_file():
                    relative_path = item.relative_to(source)
                    target_file = target_dir / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_file)
    
    def _create_metadata_file(self, target_dir: Path, kaggle_path: str, 
                             dataset_config: Dict[str, Any]):
        """Create metadata file for the dataset"""
        from datetime import datetime
        
        metadata = {
            'kaggle_path': kaggle_path,
            'description': dataset_config.get('description', ''),
            'download_date': datetime.now().isoformat(),
            'local_path': str(target_dir.relative_to(self.project_root)),
            'files': [f.name for f in target_dir.rglob('*') if f.is_file()]
        }
        
        metadata_file = target_dir / 'dataset_metadata.yaml'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"✓ Created metadata file: {metadata_file}")
    
    def _list_available_datasets(self):
        """List all available datasets in config"""
        print("\n📋 Available datasets in config:")
        for name, config in self.config['datasets'].items():
            print(f"  • {name}: {config['kaggle_path']}")
            if config.get('description'):
                print(f"    Description: {config['description']}")
        print()
    
    def download_all(self) -> bool:
        """Download all datasets from config"""
        print("🚀 Downloading all datasets...")
        success_count = 0
        
        for dataset_name in self.config['datasets'].keys():
            if self.download_dataset(dataset_name):
                success_count += 1
        
        total = len(self.config['datasets'])
        print(f"\n✅ Downloaded {success_count}/{total} datasets successfully")
        return success_count == total
    
    def add_dataset_to_config(self, name: str, kaggle_path: str, 
                             description: str = "", local_folder: str = None):
        """Add new dataset to configuration"""
        if local_folder is None:
            local_folder = name
        
        new_dataset = {
            'kaggle_path': kaggle_path,
            'description': description,
            'local_folder': local_folder,
            'files': []
        }
        
        self.config['datasets'][name] = new_dataset
        
        # Save updated config
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✓ Added dataset '{name}' to config")

def main():
    """Main function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets from Kaggle')
    parser.add_argument('action', choices=['download', 'download-all', 'list', 'add'], 
                       help='Action to perform')
    parser.add_argument('--dataset', '-d', help='Dataset name to download')
    parser.add_argument('--kaggle-path', '-k', help='Kaggle dataset path (for add action)')
    parser.add_argument('--description', help='Dataset description (for add action)')
    parser.add_argument('--config', '-c', help='Path to config file')
    
    args = parser.parse_args()
    
    downloader = KaggleDatasetDownloader(args.config)
    
    if args.action == 'download':
        if not args.dataset:
            print("❌ Please specify dataset name with --dataset")
            downloader._list_available_datasets()
            return
        downloader.download_dataset(args.dataset)
    
    elif args.action == 'download-all':
        downloader.download_all()
    
    elif args.action == 'list':
        downloader._list_available_datasets()
    
    elif args.action == 'add':
        if not args.dataset or not args.kaggle_path:
            print("❌ Please specify both --dataset and --kaggle-path")
            return
        downloader.add_dataset_to_config(
            args.dataset, 
            args.kaggle_path, 
            args.description or ""
        )

if __name__ == "__main__":
    main()