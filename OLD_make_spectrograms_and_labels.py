import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging
import time
import datetime
import pytz
from collections import Counter
import glob
import pandas as pd

# Set up logging
os.makedirs('Logs', exist_ok=True)
log_file = f'Logs/make_spectrograms{datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")}.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

def spectrogram_to_image(args):
    """
    Extract a single spectrogram segment, save it as a PNG file, and generate a label.
    """
    segment, sr, filestem, destinationImageFolder, index, start, end, time_segments = args
    
    output_filename = f"{filestem}_{index}_{start}_{end}.png"
    output_path = os.path.join(destinationImageFolder, output_filename)
    
    # Compute the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)

    # Generate the plot and save it
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(librosa.power_to_db(spectrogram, ref=np.max), aspect='auto', origin='lower')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Check for overlap with labeled segments
    overlap_check = any(
        max(0, min(seg[1], end / 1000) - max(seg[0], start / 1000)) >= 0.2
        for seg in time_segments
    )
    label = 1 if overlap_check else 0
    
    return output_filename, label

def extract_spectrograms_from_file(filepath, destinationImageFolder, time_segments, segment_duration=2, overlap=0, offset=0, duration=None):
    """
    Extracts spectrograms from an audio file using efficient batch processing.
    """
    audio, sr = librosa.load(filepath, offset=offset, duration=duration, sr=None)
    segment_samples = segment_duration * sr
    overlap_samples = overlap * sr
    filestem = os.path.basename(filepath).split('.')[0]
    
    # Create jobs for each segment
    jobs = []
    start = 0
    while start + segment_samples <= len(audio):
        startms = round(start / sr * 1000)
        endms = round((start + segment_samples) / sr * 1000)
        segment = audio[int(start):int(start + segment_samples)]
        jobs.append((segment, sr, filestem, destinationImageFolder, len(jobs), startms, endms, time_segments))
        start += (segment_samples - overlap_samples)
    
    # Process jobs in parallel
    logging.info(f'{filepath} has {len(audio)/sr/60} minutes and {len(jobs)} images will be created')
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(spectrogram_to_image, jobs)
    
    return results

def make_spectrograms_with_labels(sourceAudioFolder, destinationImageFolder, allAnnotationsCsv, sdur=2, overlap=0):
    """
    Generate spectrograms and labels for each audio file found in the given folder path.
    """
    os.makedirs(destinationImageFolder, exist_ok=True)
    labels = pd.read_csv(allAnnotationsCsv)
    labels['time_segments'] = list(zip(labels['Begin Time (s)'], labels['End Time (s)']))
    
    all_results = []
    
    files = [i for i in os.listdir(sourceAudioFolder) if i.endswith('.wav')]
    logging.info(f'{len(files)} audio files to be processed')
    logging.info(f'spectrograms to be saved in {destinationImageFolder}')
    
    for f in files:
        filepath = os.path.join(sourceAudioFolder, f)
        time_segments = list(labels[labels['audiofile'] == f]['time_segments'])
        results = extract_spectrograms_from_file(filepath=filepath, destinationImageFolder=destinationImageFolder, time_segments=time_segments, segment_duration=sdur, overlap=overlap)
        all_results.extend(results)
        logging.info(f"{len([i for i in os.listdir(destinationImageFolder) if f.split('.')[0] in i])} images created from audio file {f}")
    
    df = pd.DataFrame(all_results, columns=['filename', 'label'])
    return df



###
HOW TO USE THIS FILE TO MAKE SPECTROGRAM IMAGES WITH labels

# make sure you have an annotation file like belugaAnnotations.csv or humpback_annotations.csv
## for beluga, use the file belugaAnnotations.py
## for humpback, and probably killer whale, use gather_annotations.py 
## Required columns are
### Begin Time (s)
### End Time (s)
### location
### audiofile
### labelfile


def generate_spectrograms_and_labels_for_species_locations(species_name, location_names, annotation_csv_path=None, sdur=2, overlap=0.4):
    """
    Generates spectrograms and label files for each location of a given species.

    Args:
        species_name (str): Name of the whale species (e.g., "Orca", "Beluga", "Humpback")
        location_names (list): List of location names as strings
        annotation_csv_path (str): Path to the annotation CSV file
        sdur (int): Segment duration in seconds
        overlap (float): Overlap in seconds
    """
    logging.info(f"Starting spectrogram generation for species: {species_name}")
    for name in location_names:
        logging.info(f"Processing location: {name}")
        source_audio_folder = f'./DataInput/{species_name}/{name}'
        destination_image_folder = f'./DataInput/{species_name}/SpectrogramsOverlap400ms/{name}'
        if annotation_csv_path is None:
            annotation_csv_path = f'/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/{species_name}/{species_name}_annotations.csv'

        df = make_spectrograms_with_labels(
            sourceAudioFolder=source_audio_folder,
            destinationImageFolder=destination_image_folder,
            allAnnotationsCsv=annotation_csv_path,
            sdur=sdur,
            overlap=overlap
        )

        output_csv_path = f'/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/{species_name}/{name.lower()}_{species_name.lower()}_overlap{int(overlap*1000)}ms_spectrogram_labels.csv'
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Finished processing location: {name}. Labels saved to {output_csv_path}")
    logging.info(f"Completed spectrogram generation for species: {species_name}")

def combine_label_csvs(species_name, overlap_ms=400):
    """
    Combines all per-location label CSVs into a master label file for a species.

    Args:
        species_name (str): Name of the whale species (e.g., "Orca")
        overlap_ms (int): Overlap duration in milliseconds, used in filename matching
    """
    logging.info(f"Starting CSV combination for species: {species_name}")
    base_path = f'/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/{species_name}'
    pattern = os.path.join(base_path, f'*_{species_name.lower()}_overlap{overlap_ms}ms_spectrogram_labels.csv')
    file_list = glob.glob(pattern)
    logging.info(f"Found {len(file_list)} CSV files to combine.")

    label_df = pd.concat([pd.read_csv(file) for file in file_list])

    label_df['location'] = label_df['filename'].str.split('_').str[0].str.replace('AU-', '')
    label_df['dirpath'] = label_df.apply(
        lambda x: os.path.join(base_path, 'SpectrogramsOverlap400ms', x['location']), axis=1
    )
    label_df['fullpath'] = label_df.apply(
        lambda x: os.path.join(base_path, 'SpectrogramsOverlap400ms', x['location'], x['filename']), axis=1
    )

    output_csv_path = os.path.join(base_path, f'{species_name.lower()}_spectrogram_labels_overlap{overlap_ms}ms.csv')
    label_df.to_csv(output_csv_path, index=False)
    logging.info(f"Combined CSV saved to {output_csv_path}")
    logging.info(f"Completed CSV combination for species: {species_name}")

generate_spectrograms_and_labels_for_species_locations('Orca',['Iniskin','PtGraham','Chinitna','SWCorner'])
combine_label_csvs(species_name="Orca", overlap_ms=400)