#!/usr/bin/env python3
"""
Utility script to sample and display spectrograms by location and label combinations.
Usage:
  python display_spectrogram.py labels.csv --howmany 5
"""
import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import glob


def create_spectrogram_image(pt_path, label, location, species, audiofile=None, row_idx=None):
    """Create and save a spectrogram image from a .pt file."""
    if not os.path.exists(pt_path):
        print(f"Warning: Spectrogram file not found: {pt_path}")
        return False

    # Load spectrogram
    spec = torch.load(pt_path)
    if hasattr(spec, 'cpu'):
        spec = spec.cpu().numpy()

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    
    title = os.path.basename(pt_path)
    if audiofile is not None:
        title += f" | audio: {audiofile}"
    title += f" | species: {species} | location: {location} | label: {label}"
    if row_idx is not None:
        title += f" | row: {row_idx}"
    
    plt.title(title, fontsize=10)
    plt.colorbar(label='dB')
    plt.xlabel('Time frames')
    plt.ylabel('Frequency bins')
    plt.tight_layout()
    
    # Create folder structure: species_location_label
    folder_name = f"BulkDisplay/{species}_{location}_label{label}"
    base_dir = os.path.expanduser("~/ssdprivate/NOAAWhalesV2/Diagnostics")
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save file
    savefilename = os.path.splitext(os.path.basename(pt_path))[0]
    save_path = os.path.join(output_dir, f"{savefilename}.png")
    print(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sample and display spectrograms by location and label combinations."
    )
    parser.add_argument(
        "labels_csv",
        help="Path to the labels CSV file."
    )
    parser.add_argument(
        "--howmany", type=int, required=True,
        help="Number of random samples to take for each location+label combination."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)."
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="Create a single PDF with all generated spectrograms."
    )
    parser.add_argument(
        "--pdf-name", type=str, default=None,
        help="Name of the output PDF file (default: auto-generated with species name)."
    )
    
    args = parser.parse_args()
    
    print(f"Loading labels from: {args.labels_csv}")
    print(f"Sampling {args.howmany} examples per location+label combination")
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Load labels
    df = pd.read_csv(args.labels_csv)
    print(f"Loaded {len(df)} total samples")
    
    # Check required columns
    required_cols = ['location', 'label', 'species']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Group by location, label, and species
    grouped = df.groupby(['species', 'location', 'label'])
    print(f"Found {len(grouped)} unique species+location+label combinations")
    
    total_processed = 0
    total_successful = 0
    
    for (species, location, label), group in grouped:
        print(f"\nProcessing: {species} at {location} with label {label} ({len(group)} available samples)")
        
        # Sample random examples
        n_samples = min(args.howmany, len(group))
        sampled = group.sample(n=n_samples, random_state=args.seed)
        
        for idx, row in sampled.iterrows():
            # Determine file path
            if 'fullpath' in row and pd.notna('.'+row['fullpath']):
                pt_path = '.' + row['fullpath'] if row['fullpath'].startswith('/') else '.'+row['fullpath']
            else:
                pt_path = os.path.join(row.get('dirpath', ''), row['filename'])
            
            audiofile = row.get('audiofile', None)
            print(pt_path)
            # Create spectrogram image
            success = create_spectrogram_image(pt_path, label, location, species, audiofile, idx)
            total_processed += 1
            if success:
                total_successful += 1
        
        print(f"  Processed {n_samples} samples for this combination")
    
    print(f"\nSummary:")
    print(f"Total samples processed: {total_processed}")
    print(f"Successfully created: {total_successful}")
    print(f"Failed: {total_processed - total_successful}")
    print(f"Output directory: {os.path.expanduser('~/ssdprivate/NOAAWhalesV2/Diagnostics')}")
    
    # Create PDF if requested
    if args.pdf:
        # Generate species-based filename if not provided
        if args.pdf_name is None:
            species_list = sorted(df['species'].unique())
            species_str = "_".join(species_list)
            pdf_filename = f"{species_str}_spectrograms_summary.pdf"
        else:
            pdf_filename = args.pdf_name
        create_pdf_from_pngs(pdf_filename)


def create_pdf_from_pngs(pdf_name):
    """Create a PDF from all PNG files in the BulkDisplay folders."""
    base_dir = os.path.expanduser("~/ssdprivate/NOAAWhalesV2/Diagnostics")
    bulk_display_dir = os.path.join(base_dir, "BulkDisplay")
    
    if not os.path.exists(bulk_display_dir):
        print("No BulkDisplay directory found. No PDF created.")
        return
    
    # Find all PNG files in subdirectories
    png_pattern = os.path.join(bulk_display_dir, "**", "*.png")
    png_files = sorted(glob.glob(png_pattern, recursive=True))
    
    if not png_files:
        print("No PNG files found. No PDF created.")
        return
    
    pdf_path = os.path.join(base_dir, pdf_name)
    print(f"\nCreating PDF with {len(png_files)} images...")
    print(f"PDF will be saved as: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        for i, png_file in enumerate(png_files):
            # Extract metadata from path
            rel_path = os.path.relpath(png_file, bulk_display_dir)
            folder_name = os.path.dirname(rel_path)
            filename = os.path.basename(png_file)
            
            # Create a new figure for each image
            fig, ax = plt.subplots(figsize=(11, 8.5))  # Letter size
            
            # Load and display the PNG
            img = plt.imread(png_file)
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with metadata
            title = f"Page {i+1}/{len(png_files)} - {folder_name}\n{filename}"
            fig.suptitle(title, fontsize=10, y=0.95)
            
            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(png_files):
                print(f"  Processed {i+1}/{len(png_files)} images")
    
    print(f"PDF created successfully: {pdf_path}")


if __name__ == '__main__':
    main()
