#!/usr/bin/env python3
import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional
import time


class BirdAudioDownloader:
    def __init__(self, api_key: str, taxonomy_file: str = "birds_only.csv", output_dir: str = "more_birds"):
       
        self.api_key = api_key
        self.api_base_url = "https://xeno-canto.org/api/3/recordings"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if os.path.exists(taxonomy_file):
            self.taxonomy_df = pd.read_csv(taxonomy_file)
        else:
            print(f"Warning: Taxonomy file '{taxonomy_file}' not found. Species lookup may be limited.")
            self.taxonomy_df = None
    
    def _get_scientific_name(self, bird_identifier: str) -> Optional[str]:
        
        if self.taxonomy_df is None:
            return bird_identifier

        match = self.taxonomy_df[self.taxonomy_df['primary_label'] == bird_identifier]
        if not match.empty:
            return match.iloc[0]['scientific_name']
        
        match = self.taxonomy_df[self.taxonomy_df['scientific_name'] == bird_identifier]
        if not match.empty:
            return match.iloc[0]['scientific_name']
        
        match = self.taxonomy_df[self.taxonomy_df['common_name'].str.contains(bird_identifier, case=False, na=False)]
        if not match.empty:
            return match.iloc[0]['scientific_name']
        
        return None
    
    def _get_primary_label(self, bird_identifier: str) -> str:
       
        if self.taxonomy_df is None:
            return bird_identifier.replace(" ", "_")
        
        match = self.taxonomy_df[self.taxonomy_df['primary_label'] == bird_identifier]
        if not match.empty:
            return match.iloc[0]['primary_label']
        
        match = self.taxonomy_df[self.taxonomy_df['scientific_name'] == bird_identifier]
        if not match.empty:
            return match.iloc[0]['primary_label']
        
        match = self.taxonomy_df[self.taxonomy_df['common_name'].str.contains(bird_identifier, case=False, na=False)]
        if not match.empty:
            return match.iloc[0]['primary_label']
        
        return bird_identifier.replace(" ", "_")
    
    def _search_recordings(self, scientific_name: str, per_page: int = 500) -> List[dict]:
        
        all_recordings = []
        page = 1
        
        while True:
            query = f'sp:"{scientific_name}"'
            params = {
                'query': query,
                'key': self.api_key,
                'per_page': min(per_page, 500),
                'page': page
            }
            
            try:
                response = requests.get(self.api_base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'error' in data:
                    print(f"  API Error: {data['error']['message']}")
                    break
                
                recordings = data.get('recordings', [])
                if not recordings:
                    break
                
                all_recordings.extend(recordings)
                
                num_pages = int(data.get('numPages', 1))
                if page >= num_pages:
                    break
                
                page += 1
                
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Error getting recordings: {e}")
                break
        
        return all_recordings
    
    def _download_audio_file(self, recording: dict, output_path: Path) -> bool:
        file_url = recording.get('file', '')
        if not file_url:
            return False
        
        if file_url.startswith('//'):
            file_url = 'https:' + file_url
        elif not file_url.startswith('http'):
            file_url = 'https://xeno-canto.org' + file_url
        
        try:
            response = requests.get(file_url, timeout=60, stream=True)
            response.raise_for_status()
            
            filename = recording.get('file-name', f"XC{recording['id']}.mp3")
            filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
            if not filename.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                filename += '.mp3'
            
            file_path = output_path / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_url}: {e}")
            return False
    
    def download_bird_audio(self, bird_identifier: str, max_files: int = 10) -> dict:
        scientific_name = self._get_scientific_name(bird_identifier)
        if not scientific_name:
            print(f"Error: Could not find scientific name for '{bird_identifier}'")
            return {'success': False, 'downloaded': 0, 'available': 0}
        
        primary_label = self._get_primary_label(bird_identifier)
        
        print(f"Bird: {scientific_name} ({primary_label})")
        print(f"Maximum: {max_files} files")
        
        print("Searching Xeno-canto...")
        recordings = self._search_recordings(scientific_name)
        
        if not recordings:
            print(f"  No recordings found for {scientific_name}")
            return {'success': False, 'downloaded': 0, 'available': 0}
        
        print(f"  Found {len(recordings)} recording(s)")
        
        recordings_to_download = recordings[:max_files]
        
        bird_dir = self.output_dir / primary_label
        bird_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {len(recordings_to_download)} file(s) to {bird_dir}...")
        downloaded = 0
        failed = 0
        
        for i, recording in enumerate(recordings_to_download, 1):
            recording_id = recording.get('id', 'unknown')
            print(f"  [{i}/{len(recordings_to_download)}] XC{recording_id}...", end=' ')
            
            if self._download_audio_file(recording, bird_dir):
                print("yes")
                downloaded += 1
            else:
                print("no")
                failed += 1
            
            time.sleep(0.3)
        
        print(f"\nDownloaded: {downloaded} files, Failed: {failed} files")
        
        return {
            'success': True,
            'downloaded': downloaded,
            'available': len(recordings),
            'requested': max_files,
            'directory': str(bird_dir)
        }
    
    def download_multiple_birds(self, bird_identifiers: List[str], max_files: int = 10) -> dict:
        results = {}
        total_downloaded = 0
        total_available = 0
        
        for bird in bird_identifiers:
            result = self.download_bird_audio(bird, max_files)
            results[bird] = result
            if result.get('success'):
                total_downloaded += result['downloaded']
                total_available += result['available']
        
        print(f"Summary")
        print(f"Birds processed: {len(bird_identifiers)}")
        print(f"Total files downloaded: {total_downloaded}")
        print(f"Total recordings available: {total_available}")
        
        return {
            'results': results,
            'total_downloaded': total_downloaded,
            'total_available': total_available
        }


def main():
    api_key = "5964a2ece9bc1032f1ffbd8df26295241ae8c070"
    
    taxonomy_file = "birds_only.csv"
    df = pd.read_csv(taxonomy_file)
    all_birds = df['primary_label'].tolist()
    
    bird = all_birds
    max_files = 500
    
    downloader = BirdAudioDownloader(
        api_key=api_key,
        taxonomy_file=taxonomy_file,
        output_dir="more_birds"
    )
    
    if isinstance(bird, list):
        downloader.download_multiple_birds(bird, max_files)
    else:
        downloader.download_bird_audio(bird, max_files)


if __name__ == "__main__":
    main()

