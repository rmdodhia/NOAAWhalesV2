#!/usr/bin/env python3
"""
Create spectrograms directly from WAV files based on species annotation data.
This script takes annotation data and generates spectrograms with configurable 
time padding before and after the annotated segments.

Usage:
  python create_spectrogram_from_wav.py annotations.csv --before 2.0 --after 2.0 --howmany 5
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')


def load_audio_segment(wav_path, start_time, end_time, before_pad, after_pad, sr=None):
    """
    Load a segment of audio from a WAV file with padding.
    
    Args:
        wav_path: Path to the WAV file
        start_time: Start time of annotation in seconds
        end_time: End time of annotation in seconds
        before_pad: Seconds to include before start_time
        after_pad: Seconds to include after end_time
        sr: Target sample rate (None to use original)
    
    Returns:
        audio_data: Audio samples
        sample_rate: Sample rate
        actual_start: Actual start time used
        actual_end: Actual end time used
    """
    if not os.path.exists(wav_path):
        print(f"Warning: WAV file not found: {wav_path}")
        return None, None, None, None
    
    try:
        # Load full audio to get duration
        y_full, sr_orig = librosa.load(wav_path, sr=sr)
        duration = len(y_full) / sr_orig if sr is None else len(y_full) / sr
        
        # Calculate padded start and end times
        padded_start = max(0, start_time - before_pad)
        padded_end = min(duration, end_time + after_pad)
        
        # Load the specific segment
        y_segment, sr_final = librosa.load(wav_path, sr=sr, 
                                         offset=padded_start, 
                                         duration=padded_end - padded_start)
        
        return y_segment, sr_final, padded_start, padded_end
        
    except Exception as e:
        print(f"Error loading audio from {wav_path}: {e}")
        return None, None, None, None


def create_spectrogram_from_audio(audio_data, sr, start_time, end_time, 
                                actual_start, actual_end, species, location, 
                                audiofile, annotation_id=None):
    """
    Create and save a spectrogram from audio data.
    
    Args:
        audio_data: Audio samples
        sr: Sample rate
        start_time: Original annotation start time
        end_time: Original annotation end time
        actual_start: Actual start time of loaded segment
        actual_end: Actual end time of loaded segment
        species: Species name
        location: Location name
        audiofile: Original audio filename
        annotation_id: Optional annotation identifier
    
    Returns:
        save_path: Path where spectrogram was saved, or None if failed
    """
    try:
        # Create spectrogram using librosa
        n_fft = 2048
        hop_length = 512
        S = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        # Create time axis in seconds
        time_frames = librosa.frames_to_time(np.arange(S.shape[1]), 
                                           sr=sr, hop_length=hop_length)
        time_frames += actual_start  # Offset by actual start time
        
        # Create frequency axis
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot spectrogram
        img = plt.imshow(S_db, aspect='auto', origin='lower', 
                        extent=[time_frames[0], time_frames[-1], 
                               freq_bins[0], freq_bins[-1]],
                        cmap='magma')
        
        # Add vertical lines to show original annotation boundaries
        plt.axvline(x=start_time, color='red', linestyle='--', linewidth=2, 
                   label=f'Annotation Start ({start_time:.2f}s)')
        plt.axvline(x=end_time, color='red', linestyle='--', linewidth=2, 
                   label=f'Annotation End ({end_time:.2f}s)')
        
        # Labels and title
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.colorbar(img, label='Power (dB)')
        
        title = f"{species} | {location} | {audiofile}"
        if annotation_id is not None:
            title += f" | ID: {annotation_id}"
        title += f"\nSegment: {actual_start:.2f}s - {actual_end:.2f}s | Annotation: {start_time:.2f}s - {end_time:.2f}s"
        
        plt.title(title, fontsize=10)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # Create output directory
        safe_species = species.replace(' ', '_').replace('/', '_')
        safe_location = location.replace(' ', '_').replace('/', '_')
        safe_audiofile = audiofile.replace(' ', '_').replace('/', '_').replace('.wav', '')
        
        folder_name = f"WavSpectrograms/{safe_species}_{safe_location}"
        base_dir = os.path.expanduser("~/ssdprivate/NOAAWhalesV2/Diagnostics")
        output_dir = os.path.join(base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        if annotation_id is not None:
            filename = f"{safe_audiofile}_id{annotation_id}_{start_time:.2f}s-{end_time:.2f}s.png"
        else:
            filename = f"{safe_audiofile}_{start_time:.2f}s-{end_time:.2f}s.png"
        
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        plt.close()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create spectrograms from WAV files based on species annotations."
    )
    parser.add_argument(
        "annotations_csv",
        help="Path to the species annotations CSV file."
    )
    parser.add_argument(
        "--before", type=float, default=1.0,
        help="Seconds to include before annotation start time (default: 1.0)."
    )
    parser.add_argument(
        "--after", type=float, default=1.0,
        help="Seconds to include after annotation end time (default: 1.0)."
    )
    parser.add_argument(
        "--howmany", type=int, default=None,
        help="Maximum number of annotations to process per species+location combination (default: all)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=None,
        help="Target sample rate for audio loading (default: use original)."
    )
    parser.add_argument(
        "--species", type=str, default=None,
        help="Filter to specific species only (default: all species)."
    )
    parser.add_argument(
        "--location", type=str, default=None,
        help="Filter to specific location only (default: all locations)."
    )
    
    args = parser.parse_args()
    
    print(f"Loading annotations from: {args.annotations_csv}")
    print(f"Time padding: {args.before}s before, {args.after}s after annotation")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load annotations
    df = pd.read_csv(args.annotations_csv)
    print(f"Loaded {len(df)} total annotations")
    
    # Check required columns
    required_cols = ['species', 'location', 'audiofile', 'start_time', 'end_time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Apply filters if specified
    if args.species:
        df = df[df['species'] == args.species]
        print(f"Filtered to species '{args.species}': {len(df)} annotations")
    
    if args.location:
        df = df[df['location'] == args.location]
        print(f"Filtered to location '{args.location}': {len(df)} annotations")
    
    if len(df) == 0:
        print("No annotations remaining after filtering!")
        return
    
    # Group by species and location
    grouped = df.groupby(['species', 'location'])
    print(f"Found {len(grouped)} unique species+location combinations")
    
    total_processed = 0
    total_successful = 0
    
    for (species, location), group in grouped:
        print(f"\nProcessing: {species} at {location} ({len(group)} available annotations)")
        
        # Sample if requested
        if args.howmany and len(group) > args.howmany:
            sampled = group.sample(n=args.howmany, random_state=args.seed)
            print(f"  Sampling {args.howmany} from {len(group)} annotations")
        else:
            sampled = group
        
        for idx, row in sampled.iterrows():
            total_processed += 1
            
            # Get annotation details
            audiofile = row['audiofile']
            start_time = float(row['start_time'])
            end_time = float(row['end_time'])
            annotation_id = row.get('id', idx)  # Use 'id' column if available, otherwise row index
            
            # Construct WAV file path - you may need to adjust this based on your file structure
            if 'wav_path' in row and pd.notna(row['wav_path']):
                wav_path = row['wav_path']
            elif 'dirpath' in row and pd.notna(row['dirpath']):
                wav_path = os.path.join(row['dirpath'], audiofile)
            else:
                # Try to construct path based on species and location
                # This is a guess - you may need to adjust based on your file structure
                base_audio_dir = "/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput"
                species_dir = species.capitalize()
                wav_path = os.path.join(base_audio_dir, species_dir, "AudioFiles", audiofile)
            
            print(f"  Processing annotation {annotation_id}: {audiofile} ({start_time:.2f}s - {end_time:.2f}s)")
            print(f"    WAV path: {wav_path}")
            
            # Load audio segment
            audio_data, sr, actual_start, actual_end = load_audio_segment(
                wav_path, start_time, end_time, args.before, args.after, args.sample_rate
            )
            
            if audio_data is not None:
                # Create spectrogram
                save_path = create_spectrogram_from_audio(
                    audio_data, sr, start_time, end_time, actual_start, actual_end,
                    species, location, audiofile, annotation_id
                )
                
                if save_path:
                    print(f"    Saved: {save_path}")
                    total_successful += 1
                else:
                    print(f"    Failed to create spectrogram")
            else:
                print(f"    Failed to load audio")
    
    print(f"\nSummary:")
    print(f"Total annotations processed: {total_processed}")
    print(f"Successfully created spectrograms: {total_successful}")
    print(f"Failed: {total_processed - total_successful}")
    print(f"Output directory: {os.path.expanduser('~/ssdprivate/NOAAWhalesV2/Diagnostics/WavSpectrograms')}")


if __name__ == '__main__':
    main()
