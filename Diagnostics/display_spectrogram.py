#!/usr/bin/env python3
"""
Utility script to compare pre-generated and newly generated spectrograms side by side.
Usage:
  python display_spectrogram.py --species Beluga --row 42
"""
import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import librosa


def load_pregenerated_spectrogram(pt_path, hop_length=256, sample_rate=None):
    """Load and prepare a pre-generated spectrogram from .pt file."""
    if not os.path.exists(pt_path):
        print(f"Warning: Spectrogram file not found: {pt_path}")
        return None, None, None
    
    # Load spectrogram
    spec = torch.load(pt_path)
    if hasattr(spec, 'cpu'):
        spec = spec.cpu().numpy()
    
    # Calculate time and frequency axes
    n_time_frames = spec.shape[1] if spec.ndim == 2 else spec.shape[-1]
    n_freq_bins = spec.shape[0] if spec.ndim == 2 else spec.shape[-2]
    
    if sample_rate is None:
        # Try to infer from common values
        sample_rate = 24000  # Common for marine audio
    
    # Time axis in seconds
    time_axis = np.arange(n_time_frames) * hop_length / sample_rate
    
    # Frequency axis in Hz (approximate for display)
    freq_axis = np.linspace(0, sample_rate/2, n_freq_bins)
    
    return spec, time_axis, freq_axis


def generate_spectrogram_from_audio(audio_path, start_ms, end_ms, n_fft=1024, hop_length=256):
    """Generate a spectrogram from raw audio file for the specified time segment."""
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return None, None, None, None
    
    try:
        # Load audio info first
        info = torchaudio.info(audio_path)
        sample_rate = info.sample_rate
        
        # Convert milliseconds to samples
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)
        num_samples = end_sample - start_sample
        
        if num_samples <= 0:
            print(f"Invalid time range: {start_ms}ms to {end_ms}ms")
            return None, None, None, None
        
        # Load the specific audio segment
        wave_tensor, sr = torchaudio.load(
            audio_path,
            frame_offset=start_sample,
            num_frames=num_samples
        )
        
        # Handle multi-channel audio
        if wave_tensor.shape[0] > 1:
            wave = wave_tensor.mean(dim=0)
        else:
            wave = wave_tensor[0]
        
        # Generate spectrogram using same parameters as original scripts
        spec = torch.stft(
            wave,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            return_complex=True
        )
        
        # Convert to magnitude and dB
        mag = spec.abs()
        db = 20 * torch.log10(mag + 1e-6)
        db = torch.clamp(db, min=-80, max=0)
        
        # Convert to numpy
        spec_np = db.numpy()
        
        # Calculate time and frequency axes
        n_time_frames = spec_np.shape[1]
        n_freq_bins = spec_np.shape[0]
        
        # Time axis in seconds (relative to segment start)
        time_axis = np.arange(n_time_frames) * hop_length / sample_rate
        
        # Frequency axis in Hz
        freq_axis = np.fft.fftfreq(n_fft, 1/sample_rate)[:n_freq_bins]
        
        return spec_np, time_axis, freq_axis, sample_rate
        
    except Exception as e:
        print(f"Error generating spectrogram from audio: {e}")
        return None, None, None, None


def extract_time_from_filename(filename):
    """Extract start and end milliseconds from filename like 'audio_1234_5678.pt'."""
    try:
        # Remove extension and split by underscore
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        
        # Get the last two parts as start and end times
        if len(parts) >= 2:
            start_ms = int(parts[-2])
            end_ms = int(parts[-1])
            return start_ms, end_ms
        else:
            raise ValueError("Filename doesn't contain timing information")
            
    except (ValueError, IndexError) as e:
        print(f"Error extracting time from filename '{filename}': {e}")
        return None, None


def find_audio_file(species, location, audiofile):
    """Find the audio file path based on species, location, and filename."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to NOAAWhalesV2
    
    # Extract the first part of the audiofile for subdirectory (e.g., "201875479" from "201875479.171002230002")
    audiofile_parts = audiofile.split('.')
    subdirectory = audiofile_parts[0] if len(audiofile_parts) > 1 else audiofile
    
    # Add .wav extension if not present
    if not audiofile.endswith('.wav'):
        audiofile_with_ext = audiofile + '.wav'
    else:
        audiofile_with_ext = audiofile
    
    # Try different possible paths including the nested structure
    possible_paths = [
        # Direct paths
        os.path.join(base_dir, f"DataInput/{species}/{location}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput/{species.lower()}/{location}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput/{species.capitalize()}/{location}/{audiofile_with_ext}"),
        # Nested paths with subdirectory
        os.path.join(base_dir, f"DataInput/{species}/{location}/{subdirectory}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput/{species.lower()}/{location}/{subdirectory}/{audiofile_with_ext}"),
        os.path.join(base_dir, f"DataInput/{species.capitalize()}/{location}/{subdirectory}/{audiofile_with_ext}"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If not found, also try to find the file recursively
    species_dir = os.path.join(base_dir, f"DataInput/{species}")
    if os.path.exists(species_dir):
        for root, dirs, files in os.walk(species_dir):
            if audiofile_with_ext in files:
                return os.path.join(root, audiofile_with_ext)
    
    return None


def check_annotation_overlap(species, location, audiofile, start_ms, end_ms, sample_rate):
    """
    Check if the spectrogram time range overlaps with any annotations.
    
    Args:
        species: Species name (Beluga, Humpback, Orca)
        location: Location name
        audiofile: Audio filename (without extension)
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        sample_rate: Sample rate for Beluga duration conversion
    
    Returns:
        List of overlapping annotations with details
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annotations_csv = os.path.join(base_dir, f"DataInput/{species}/{species}_annotations.csv")
    
    if not os.path.exists(annotations_csv):
        print(f"Warning: Annotations file not found: {annotations_csv}")
        return []
    
    try:
        # Load annotations
        df_ann = pd.read_csv(annotations_csv)
        
        # Convert spectrogram times to seconds
        spec_start_sec = start_ms / 1000.0
        spec_end_sec = end_ms / 1000.0
        
        # Filter annotations for the same audiofile and location
        if species.lower() == 'beluga':
            # For Beluga: match by exact audiofile name and location
            audiofile_with_ext = audiofile + '.wav' if not audiofile.endswith('.wav') else audiofile
            mask = (df_ann['audiofile'] == audiofile_with_ext) & (df_ann['location'] == location)
        else:
            # For Humpback and Orca: match by audiofile (Begin File column for Humpback)
            audiofile_with_ext = audiofile + '.wav' if not audiofile.endswith('.wav') else audiofile
            if 'Begin File' in df_ann.columns:
                # Humpback uses 'Begin File' column
                mask = df_ann['Begin File'] == audiofile_with_ext
            else:
                # Orca: extract filename from audiofile path
                mask = df_ann['audiofile'].str.contains(audiofile, na=False)
        
        filtered_annotations = df_ann[mask]
        
        if len(filtered_annotations) == 0:
            return []
        
        overlapping_annotations = []
        
        for _, ann_row in filtered_annotations.iterrows():
            if species.lower() == 'beluga':
                # For Beluga: calculate end time from startSeconds + duration/sampleRate
                ann_start = ann_row['startSeconds']
                ann_duration_sec = ann_row['duration'] / ann_row['sampleRate']
                ann_end = ann_start + ann_duration_sec
            else:
                # For Humpback and Orca: use Begin Time (s) and End Time (s)
                ann_start = ann_row['Begin Time (s)']
                ann_end = ann_row['End Time (s)']
            
            # Check for overlap: two intervals overlap if max(start1,start2) < min(end1,end2)
            overlap_start = max(spec_start_sec, ann_start)
            overlap_end = min(spec_end_sec, ann_end)
            
            if overlap_start < overlap_end:
                # There is an overlap
                overlap_duration = overlap_end - overlap_start
                overlap_info = {
                    'annotation_start': ann_start,
                    'annotation_end': ann_end,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration,
                    'annotation_row': ann_row
                }
                overlapping_annotations.append(overlap_info)
        
        return overlapping_annotations
        
    except Exception as e:
        print(f"Error checking annotation overlap: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Compare pre-generated and newly generated spectrograms side by side."
    )
    parser.add_argument(
        "--species", type=str, required=True,
        help="Species name (e.g., Beluga, Humpback, Orca)."
    )
    parser.add_argument(
        "--row", type=int, required=True,
        help="Row index (0-based) in the species' labels CSV file."
    )
    parser.add_argument(
        "--n-fft", type=int, default=1024,
        help="FFT size for new spectrogram generation (default: 1024)."
    )
    parser.add_argument(
        "--hop-length", type=int, default=256,
        help="Hop length for new spectrogram generation (default: 256)."
    )
    
    args = parser.parse_args()
    
    # Construct path to species labels CSV
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to NOAAWhalesV2
    labels_csv = os.path.join(base_dir, f"DataInput/{args.species}/LabelsOverlap400ms/{args.species}_labels.csv")
    
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    
    print(f"Loading labels from: {labels_csv}")
    print(f"Processing row index: {args.row}")
    
    # Load labels
    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} total samples")
    
    # Check if row index is valid
    if args.row < 0 or args.row >= len(df):
        raise ValueError(f"Row index {args.row} is out of range. Valid range: 0 to {len(df)-1}")
    
    # Get the specific row
    row = df.iloc[args.row]
    
    # Extract row information
    species = row['species']
    location = row['location']
    label = row['label']
    filename = row['filename']
    
    # Check if audiofile column exists and use it, otherwise extract from filename
    if 'audiofile' in row and pd.notna(row['audiofile']):
        audiofile = row['audiofile']
        # Remove .wav extension if present since we'll add it back in find_audio_file
        if audiofile.endswith('.wav'):
            audiofile = audiofile[:-4]
    else:
        # Extract audiofile from filename (before the first underscore and timestamp)
        # Pattern: {audiofile}_{start_ms}_{end_ms}.pt
        filename_base = filename.replace('.pt', '')
        parts = filename_base.split('_')
        if len(parts) >= 3:
            # Find where the timestamps start (should be all digits)
            for i in range(len(parts) - 2):
                if parts[i+1].isdigit() and parts[i+2].isdigit():
                    audiofile = '_'.join(parts[:i+1])
                    break
            else:
                print(f"Could not parse audiofile from filename: {filename}")
                return
        else:
            print(f"Unexpected filename format: {filename}")
            return
    
    print(f"\nProcessing row {args.row}:")
    print(f"  Species: {species}")
    print(f"  Location: {location}")
    print(f"  Label: {label}")
    print(f"  Audio file: {audiofile}")
    print(f"  Filename: {filename}")
    
    # Extract start and end times from filename
    start_ms, end_ms = extract_time_from_filename(filename)
    if start_ms is None or end_ms is None:
        print("Failed to extract timing information from filename")
        return
    
    print(f"  Time range: {start_ms}ms - {end_ms}ms")
    
    # Get paths
    if 'fullpath' in row and pd.notna(row['fullpath']):
        # Convert relative path to absolute path
        pt_path = row['fullpath']
        if pt_path.startswith('./'):
            pt_path = os.path.join(base_dir, pt_path[2:])  # Remove './' and prepend base_dir
        elif pt_path.startswith('/'):
            pt_path = base_dir + pt_path  # Prepend base_dir to absolute path starting with '/'
        else:
            pt_path = os.path.join(base_dir, pt_path)
    else:
        pt_path = os.path.join(base_dir, row.get('dirpath', '').lstrip('./'), filename)
    
    # Find audio file
    audio_path = find_audio_file(args.species, location, audiofile)
    if audio_path is None:
        print(f"Audio file not found: {audiofile}")
        return
    
    print(f"  Pre-generated spectrogram: {pt_path}")
    print(f"  Audio file path: {audio_path}")
    
    # Generate new spectrogram from audio
    print("Generating new spectrogram from audio...")
    spec2, time2, freq2, sample_rate = generate_spectrogram_from_audio(
        audio_path, start_ms, end_ms, n_fft=args.n_fft, hop_length=args.hop_length
    )
    if spec2 is None:
        print("Failed to generate new spectrogram")
        return
    
    # Check for annotation overlaps
    print("\nChecking for annotation overlaps...")
    overlaps = check_annotation_overlap(args.species, location, audiofile, start_ms, end_ms, sample_rate)
    
    if overlaps:
        print(f"Found {len(overlaps)} overlapping annotation(s):")
        for i, overlap in enumerate(overlaps, 1):
            print(f"  {i}. Annotation: {overlap['annotation_start']:.3f}s - {overlap['annotation_end']:.3f}s")
            print(f"     Overlap: {overlap['overlap_start']:.3f}s - {overlap['overlap_end']:.3f}s (duration: {overlap['overlap_duration']:.3f}s)")
            
            # Add frequency info if available
            ann_row = overlap['annotation_row']
            if 'lowFreq' in ann_row and pd.notna(ann_row['lowFreq']):
                print(f"     Frequency: {ann_row['lowFreq']:.1f}Hz - {ann_row['highFreq']:.1f}Hz")
            elif 'Low Freq (Hz)' in ann_row and pd.notna(ann_row['Low Freq (Hz)']):
                print(f"     Frequency: {ann_row['Low Freq (Hz)']:.1f}Hz - {ann_row['High Freq (Hz)']:.1f}Hz")
    else:
        print("No overlapping annotations found.")
    
    # Load pre-generated spectrogram using the actual sample rate from audio
    print("\nLoading pre-generated spectrogram...")
    spec1, time1, freq1 = load_pregenerated_spectrogram(pt_path, hop_length=args.hop_length, sample_rate=sample_rate)
    if spec1 is None:
        print("Failed to load pre-generated spectrogram")
        return
    
    # Create side-by-side comparison
    print("Creating comparison plot...")
    
    # Create vertical layout with annotation timeline if overlaps exist
    if overlaps:
        fig, (ax1, ax_timeline, ax2) = plt.subplots(3, 1, figsize=(14, 12), 
                                                   gridspec_kw={'height_ratios': [4, 1, 4]})
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax_timeline = None
    
    # Convert ms to seconds for plotting
    spec_start_sec = start_ms / 1000.0
    spec_end_sec = end_ms / 1000.0
    duration_sec = (end_ms - start_ms) / 1000.0
    
    # Plot pre-generated spectrogram
    im1 = ax1.imshow(spec1, aspect='auto', origin='lower', cmap='magma',
                     extent=[0, duration_sec, freq1[0], freq1[-1]])
    ax1.set_title(f"Pre-generated Spectrogram\n{filename}", fontsize=12)
    ax1.set_ylabel('Frequency (Hz)')
    if not overlaps:
        ax1.set_xlabel('Time (seconds)')
    # Removed colorbar for alignment
    
    # Add annotation timeline if overlaps exist
    if overlaps and ax_timeline:
        # Set up timeline axes
        ax_timeline.set_xlim(0, duration_sec)
        ax_timeline.set_ylim(-0.5, len(overlaps) - 0.5)
        ax_timeline.set_ylabel('Annotations', fontsize=10)
        ax_timeline.set_title('Annotation Timeline', fontsize=11)
        ax_timeline.set_xlabel('Time (seconds)')
        
        # Colors for annotations
        colors = ['red', 'cyan', 'yellow', 'lime', 'orange', 'magenta']
        
        for i, overlap in enumerate(overlaps):
            ann_start = overlap['annotation_start']
            ann_end = overlap['annotation_end']
            overlap_start = overlap['overlap_start']
            overlap_end = overlap['overlap_end']
            overlap_duration_ms = overlap['overlap_duration'] * 1000
            
            color = colors[i % len(colors)]
            
            # Convert absolute times to relative times (same as spectrogram axes)
            ann_start_rel = ann_start - spec_start_sec
            ann_end_rel = ann_end - spec_start_sec
            overlap_start_rel = overlap_start - spec_start_sec
            overlap_end_rel = overlap_end - spec_start_sec
            
            # Draw full annotation range as thick colored line
            ax_timeline.hlines(i, ann_start_rel, ann_end_rel, 
                             colors=color, linewidth=10, alpha=0.7, 
                             label=f'Ann{i+1}: {overlap_duration_ms:.0f}ms overlap')
            
            # Draw overlap region with black vertical lines at boundaries
            ax_timeline.axvline(overlap_start_rel, color='black', linewidth=4, alpha=0.9)
            ax_timeline.axvline(overlap_end_rel, color='black', linewidth=4, alpha=0.9)
            
            # Add text label for duration in the middle of overlap
            mid_time = (overlap_start_rel + overlap_end_rel) / 2
            ax_timeline.text(mid_time, i, f'{overlap_duration_ms:.0f}ms', 
                           ha='center', va='center', fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Style the timeline
        ax_timeline.set_yticks(range(len(overlaps)))
        ax_timeline.set_yticklabels([f'Ann{i+1}' for i in range(len(overlaps))])
        ax_timeline.grid(True, alpha=0.3, axis='x')
        ax_timeline.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Plot newly generated spectrogram
    im2 = ax2.imshow(spec2, aspect='auto', origin='lower', cmap='magma',
                     extent=[0, duration_sec, freq2[0], freq2[-1]])
    ax2.set_title(f"From Audio\n{audiofile} ({start_ms}-{end_ms}ms)", fontsize=12)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')
    # Removed colorbar for alignment
    
    # Overall title
    overlap_text = f" | {len(overlaps)} annotation(s)" if overlaps else " | No annotations"
    fig.suptitle(f"{species} | {location} | Label: {label} | Row: {args.row}{overlap_text}", fontsize=12)
    
    # Add explanation text if overlaps exist
    if overlaps:
        fig.text(0.5, 0.02, 
                'Colored horizontal lines: Full annotation range | Black vertical lines: Overlap boundaries',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the comparison
    output_dir = os.path.join(os.path.dirname(__file__), "Comparisons")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f"check_{species}_{location}_{start_ms}_{end_ms}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison saved: {output_path}")
    print(f"Spectrogram dimensions:")
    print(f"  Pre-generated: {spec1.shape}")
    print(f"  From audio: {spec2.shape}")
    print(f"  Sample rate: {sample_rate} Hz")


if __name__ == '__main__':
    main()
