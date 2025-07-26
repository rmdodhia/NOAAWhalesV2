#!/usr/bin/env python3
"""
Script to process all Beluga annotation files and create a unified dataset.

This script:
1. Reads all CSV files from DataInput/Beluga/Annotations/
2. Matches annotations to audio files based on Local_Time
3. Calculates duration in seconds using audio file sample rates
4. Outputs a processed CSV with all required columns

Audio file matching logic:
- Converts Local_Time to yymmddHHMMSS format
- Finds the audio file with the largest timestamp <= annotation time
- Audio files are in DataInput/Beluga/{location}/{subfolder}/{prefix}.{yymmddHHMMSS}.wav
"""

import pandas as pd
import numpy as np
import os
import glob
import librosa
import logging
from datetime import datetime
import bisect

# Set up logging
os.makedirs("Logs", exist_ok=True)
log_file = f'Logs/process_beluga_annotations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def parse_local_time_to_timestamp(local_time_str):
    """
    Parse local time string to a timestamp for matching with audio filenames.
    
    Args:
        local_time_str: String like "2017-09-30 15:47:14.053000-07:00"
    
    Returns:
        String in yymmddHHMMSS format for matching
    """
    try:
        # Remove timezone information for parsing
        if '+' in local_time_str:
            dt_part = local_time_str.split('+')[0]
        elif local_time_str.count('-') > 2:  # has timezone
            # Find the last '-' which should be timezone
            parts = local_time_str.split('-')
            dt_part = '-'.join(parts[:-1])
        else:
            dt_part = local_time_str
        
        # Parse the datetime part
        dt = pd.to_datetime(dt_part)
        
        # Format as yymmddHHMMSS (matches audio filename format)
        timestamp = dt.strftime('%y%m%d%H%M%S')
        return timestamp
        
    except Exception as e:
        logging.warning(f"Could not parse time {local_time_str}: {e}")
        return None

def load_all_audio_files():
    """
    Load metadata for all audio files in the Beluga directory structure.
    
    Returns:
        DataFrame with columns: location, subfolder, prefix, timestamp, audiofile_path, sample_rate
    """
    audio_files = []
    base_path = "DataInput/Beluga"
    
    # Find all location directories (201D, 206D, etc.)
    location_dirs = [d for d in os.listdir(base_path) 
                    if d.startswith('2') and d.endswith('D') and os.path.isdir(os.path.join(base_path, d))]
    
    logging.info(f"Found location directories: {location_dirs}")
    
    for location in location_dirs:
        location_path = os.path.join(base_path, location)
        
        # Find subdirectories in each location
        subdirs = [d for d in os.listdir(location_path) 
                  if os.path.isdir(os.path.join(location_path, d)) and d.isdigit()]
        
        for subdir in subdirs:
            subdir_path = os.path.join(location_path, subdir)
            
            # Find all wav files in this subdirectory
            wav_pattern = os.path.join(subdir_path, "*.wav")
            wav_files = glob.glob(wav_pattern)
            
            for wav_file in wav_files:
                try:
                    filename = os.path.basename(wav_file)
                    # Parse filename: prefix.yymmddHHMMSS.wav
                    parts = filename.split('.')
                    if len(parts) >= 3 and parts[-1] == 'wav':
                        prefix = parts[0]
                        timestamp = parts[1]
                        
                        # Get sample rate
                        sample_rate = librosa.get_samplerate(wav_file)
                        
                        audio_files.append({
                            'location': location,
                            'subfolder': subdir,
                            'prefix': prefix,
                            'timestamp': timestamp,
                            'audiofile_path': wav_file,
                            'audiofile_name': filename,
                            'sample_rate': sample_rate
                        })
                        
                except Exception as e:
                    logging.warning(f"Error processing audio file {wav_file}: {e}")
    
    df = pd.DataFrame(audio_files)
    logging.info(f"Loaded {len(df)} audio files")
    return df

def find_matching_audiofile(annotation_timestamp, deployment_id, audio_files_df):
    """
    Find the audio file that corresponds to an annotation.
    
    Args:
        annotation_timestamp: yymmddHHMMSS string from annotation
        deployment_id: Deployment ID (e.g., "201")
        audio_files_df: DataFrame of all audio files
    
    Returns:
        Dictionary with audio file info or None if no match found
    """
    if not annotation_timestamp or pd.isna(deployment_id):
        return None
    
    # Convert deployment_id to location format (201 -> 201D)
    if isinstance(deployment_id, (int, float)):
        location = f"{int(deployment_id)}D"
    else:
        location = f"{str(deployment_id)}D" if not str(deployment_id).endswith('D') else str(deployment_id)
    
    # Filter audio files for this location
    location_files = audio_files_df[audio_files_df['location'] == location].copy()
    
    if location_files.empty:
        logging.debug(f"No audio files found for location {location}")
        return None
    
    # Sort timestamps and find the largest one <= annotation timestamp
    location_files = location_files.sort_values('timestamp')
    timestamps = location_files['timestamp'].tolist()
    
    # Find insertion point
    pos = bisect.bisect_right(timestamps, annotation_timestamp)
    
    if pos == 0:
        # No timestamp <= annotation timestamp
        logging.debug(f"No audio file timestamp <= {annotation_timestamp} for location {location}")
        return None
    
    # Get the largest timestamp <= annotation timestamp
    matching_timestamp = timestamps[pos - 1]
    matching_row = location_files[location_files['timestamp'] == matching_timestamp].iloc[0]
    
    return matching_row.to_dict()

def process_annotation_file(csv_path, audio_files_df):
    """
    Process a single annotation CSV file.
    
    Args:
        csv_path: Path to the annotation CSV file
        audio_files_df: DataFrame of all audio files
    
    Returns:
        DataFrame with processed annotations
    """
    try:
        # Read the annotation file
        df = pd.read_csv(csv_path, low_memory=False)
        logging.info(f"Processing {csv_path}: {len(df)} rows")
        
        # Filter for Beluga species only (Species='B')
        if 'Species' in df.columns:
            original_count = len(df)
            df = df[df['Species'] == 'B'].copy()
            logging.info(f"Filtered to Beluga species only: {len(df)}/{original_count} rows")
        else:
            logging.warning(f"No 'Species' column found in {csv_path}")
        
        if df.empty:
            logging.info(f"No Beluga annotations found in {csv_path}")
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ['startSeconds', 'duration', 'lowFreq', 'highFreq', 'Species', 'Local_Time', 'UTC_Time', 'Deployment_ID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing columns in {csv_path}: {missing_cols}")
            return pd.DataFrame()
        
        # Add timestamp column for matching
        df['annotation_timestamp'] = df['Local_Time'].apply(parse_local_time_to_timestamp)
        
        # Initialize new columns
        df['audiofile'] = None
        df['audiofile_path'] = None
        df['sampleRate'] = None
        df['durationSeconds'] = None
        
        # Group by Deployment_ID for efficiency
        matched_count = 0
        for deployment_id, group in df.groupby('Deployment_ID'):
            # Convert deployment_id to location format
            if isinstance(deployment_id, (int, float)):
                location = f"{int(deployment_id)}D"
            else:
                location = f"{str(deployment_id)}D" if not str(deployment_id).endswith('D') else str(deployment_id)
            
            logging.info(f"  Processing deployment {deployment_id} -> location {location} ({len(group)} annotations)")
            
            # Pre-filter audio files for this location
            location_audio = audio_files_df[audio_files_df['location'] == location].copy()
            if location_audio.empty:
                logging.warning(f"  No audio files found for location {location}")
                available_locations = sorted(audio_files_df['location'].unique())
                logging.warning(f"  Available locations: {available_locations}")
                continue
            
            # Sort timestamps once for this location (convert to int for faster comparison)
            location_audio['timestamp_int'] = location_audio['timestamp'].astype(int)
            location_audio = location_audio.sort_values('timestamp_int')
            timestamps = location_audio['timestamp_int'].tolist()
            
            # Process each annotation in this group
            processed = 0
            for idx in group.index:
                row = df.loc[idx]
                if pd.isna(row['annotation_timestamp']):
                    continue
                
                # Convert annotation timestamp to int for comparison
                try:
                    annotation_ts_int = int(row['annotation_timestamp'])
                except:
                    continue
                
                # Find matching audio file using binary search
                pos = bisect.bisect_right(timestamps, annotation_ts_int)
                
                if pos > 0:
                    # Get the largest timestamp <= annotation timestamp
                    matching_timestamp_int = timestamps[pos - 1]
                    audio_match = location_audio[location_audio['timestamp_int'] == matching_timestamp_int].iloc[0]
                    
                    df.at[idx, 'audiofile'] = audio_match['audiofile_name']
                    df.at[idx, 'audiofile_path'] = audio_match['audiofile_path']
                    df.at[idx, 'sampleRate'] = audio_match['sample_rate']
                    # Calculate duration in seconds (samples / sample_rate)
                    df.at[idx, 'durationSeconds'] = row['duration'] / audio_match['sample_rate']
                    matched_count += 1
                
                processed += 1
                if processed % 1000 == 0:
                    logging.info(f"    Processed {processed}/{len(group)} annotations for location {location}")
            
            logging.info(f"  Location {location}: matched {matched_count}/{len(group)} annotations")
        
        logging.info(f"Matched {matched_count}/{len(df)} annotations to audio files")
        
        # Remove temporary column
        df = df.drop(columns=['annotation_timestamp'])
        
        return df
        
    except Exception as e:
        logging.error(f"Error processing {csv_path}: {e}")
        return pd.DataFrame()

def main():
    """Main processing function."""
    logging.info("Starting Beluga annotation processing")
    
    # Load all audio file metadata
    logging.info("Loading audio file metadata...")
    audio_files_df = load_all_audio_files()
    
    if audio_files_df.empty:
        logging.error("No audio files found!")
        return
    
    # Find all annotation CSV files
    annotations_path = "DataInput/Beluga/Annotations"
    csv_files = glob.glob(os.path.join(annotations_path, "*.csv"))
    logging.info(f"Found {len(csv_files)} annotation CSV files")
    
    # Process each annotation file
    all_annotations = []
    for csv_file in csv_files:
        df = process_annotation_file(csv_file, audio_files_df)
        if not df.empty:
            df['source_file'] = os.path.basename(csv_file)
            all_annotations.append(df)
    
    if not all_annotations:
        logging.error("No annotations were processed successfully")
        return
    
    # Combine all annotations
    combined_df = pd.concat(all_annotations, ignore_index=True)
    logging.info(f"Combined {len(combined_df)} total annotations")
    
    # Rename Deployment_ID to location for clarity
    if 'Deployment_ID' in combined_df.columns:
        combined_df['location'] = combined_df['Deployment_ID']
        combined_df = combined_df.drop(columns=['Deployment_ID'])
    
    # Rename duration to durationSamples for clarity
    if 'duration' in combined_df.columns:
        combined_df = combined_df.rename(columns={'duration': 'durationSamples'})
    
    # Select and order final columns
    final_columns = [
        'Species',
        'location',
        'startSeconds',
        'durationSeconds',
        'audiofile',
        'Local_Time',
        'UTC_Time',
        'durationSamples', 
        'lowFreq',
        'highFreq',
        'sampleRate',
        'audiofile_path',
        'source_file'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in final_columns if col in combined_df.columns]
    final_df = combined_df[available_columns].copy()
    
    # Save the processed annotations
    output_path = "DataInput/Beluga/Beluga_annotations_processed.csv"
    final_df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(final_df)} processed annotations to {output_path}")
    
    # Print summary statistics
    logging.info("\nSUMMARY STATISTICS:")
    logging.info(f"Total annotations: {len(final_df)}")
    
    # Count by species
    species_counts = final_df['Species'].value_counts()
    logging.info("Species distribution:")
    for species, count in species_counts.items():
        logging.info(f"  {species}: {count}")
    
    # Count by location
    if 'location' in final_df.columns:
        location_counts = final_df['location'].value_counts()
        logging.info("Location distribution:")
        for location, count in location_counts.items():
            logging.info(f"  {location}: {count}")
    
    # Count matched vs unmatched
    matched = final_df['audiofile'].notna().sum()
    unmatched = len(final_df) - matched
    logging.info(f"Matched to audio files: {matched}")
    logging.info(f"Unmatched: {unmatched}")
    
    if matched > 0:
        avg_duration = final_df[final_df['durationSeconds'].notna()]['durationSeconds'].mean()
        logging.info(f"Average annotation duration: {avg_duration:.3f} seconds")
    
    logging.info("Processing completed successfully!")

if __name__ == "__main__":
    main()
