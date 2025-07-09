#!/usr/bin/env python3
"""
Utility script to load a spectrogram tensor (saved as .pt) and display it using matplotlib.
Usage:
  python display_spectrogram.py labels.csv --index 5
  python display_spectrogram.py --file ./DataInput/Humpback/SpectrogramsOverlap400ms/AL16_BS4/AL16_BS4_1000_3000.pt
"""
import argparse
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Load and display a spectrogram tensor (.pt) from a labels CSV or directly from a file."
    )
    parser.add_argument(
        "labels_csv", nargs="?", default=None,
        help="Path to the labels CSV file (if providing --index)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--index", type=int,
        help="Row index in the CSV to load the spectrogram from."
    )
    group.add_argument(
        "--file", dest="pt_file",
        help="Direct path to a .pt spectrogram file."
    )
    args = parser.parse_args()

    if args.index is not None:
        if not args.labels_csv:
            parser.error("labels_csv must be provided when using --index")
        df = pd.read_csv(args.labels_csv)
        if args.index < 0 or args.index >= len(df):
            parser.error(f"Index {args.index} out of range (0..{len(df)-1})")
        row = df.iloc[args.index]
        # support either fullpath or reconstruct from dirpath+filename
        if 'fullpath' in row and os.path.exists(row['fullpath']):
            pt_path = row['fullpath']
        else:
            pt_path = os.path.join(row.get('dirpath', ''), row['filename'])
        label = row.get('label', None)
        audiofile = row.get('audiofile', None)
    else:
        pt_path = args.pt_file
        label = None
        audiofile = None

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Spectrogram file not found: {pt_path}")

    # Load and display
    spec = torch.load(pt_path)
    if hasattr(spec, 'cpu'):
        spec = spec.cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    title = os.path.basename(pt_path)
    if audiofile is not None:
        title += f" | audio: {audiofile}"
    if label is not None:
        title += f" | label: {label}"
    plt.title(title)
    plt.colorbar(label='dB')
    plt.xlabel('Time frames')
    plt.ylabel('Frequency bins')
    plt.tight_layout()
    plt.savefig(f"preview_{os.path.splitext(pt_path)[-1]}.png")


if __name__ == '__main__':
    main()
