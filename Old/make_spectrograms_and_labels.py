import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging
import datetime
import pytz
import glob

#run this script from the root of the repository
os.chdir('/home/radodhia/ssdprivate/NOAAWhalesV2')

# ---- Config ----
SDUR = 2        # duration of each audio segment in seconds
OVERLAP = 0.4   # overlap between segments in seconds
N_RANDOM_NEGS = 1000  # number of random negative samples to draw for Beluga

# ---- Logging ----
os.makedirs('Logs', exist_ok=True)
log_file = f"Logs/make_spectrograms_{datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

# ---- Utilities ----
def resolve_audio_folder(base_path):
    """
    Look through the directory and subdirectories to find where the .wav files are located.
    Returns the first folder path containing .wav files.
    """    
    for root, _, files in os.walk(base_path):
        if any(f.endswith('.wav') for f in files):
            return root
    raise FileNotFoundError(f"No .wav files found under {base_path}")

def spectrogram_to_image(args):
    """
    Given a segment of audio, create and save a mel spectrogram image.
    Returns the filename and label (1 if overlapping a whale call, else 0).
    """

    segment, sr, filestem, destinationImageFolder, index, start, end, time_segments = args
    output_filename = f"{filestem}_{index}_{start}_{end}.png"
    output_path = os.path.join(destinationImageFolder, output_filename)
    spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.imshow(librosa.power_to_db(spectrogram, ref=np.max), aspect='auto', origin='lower')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    overlap_check = any(max(0, min(seg[1], end / 1000) - max(seg[0], start / 1000)) >= 0.2 for seg in time_segments)
    label = 1 if overlap_check else 0
    return output_filename, label

def extract_spectrograms_from_file(filepath, destinationImageFolder, time_segments, segment_duration, overlap):
    """
    Split a .wav file into overlapping segments, generate spectrogram images, and assign labels.
    Returns a list of (filename, label) pairs.
        
    Arguments:
    - filepath (str): Path to the .wav file
    - destinationImageFolder (str): Folder to save the spectrogram .png images
    - time_segments (list of tuples): List of (start_time, end_time) tuples in seconds representing known whale calls
    - segment_duration (float): Length of each segment in seconds (e.g., 2.0)
    - overlap (float): Overlap between consecutive segments in seconds (e.g., 0.4)

    Returns:
    - List of tuples: (output_filename, label), one per generated image
    """

    audio, sr = librosa.load(filepath, sr=None)
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    filestem = os.path.basename(filepath).split('.')[0]
    jobs, start = [], 0
    while start + segment_samples <= len(audio):
        startms = round(start / sr * 1000)
        endms = round((start + segment_samples) / sr * 1000)
        segment = audio[int(start):int(start + segment_samples)]
        jobs.append((segment, sr, filestem, destinationImageFolder, len(jobs), startms, endms, time_segments))
        start += (segment_samples - overlap_samples)
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(spectrogram_to_image, jobs)
    return results

def sample_beluga_negatives(df):
    """
    For Beluga only: retain all positives, include negatives near positives,
    and sample a limited number of other random negatives.
    """
 
    df['start_ms'] = df['filename'].apply(lambda x: int(x.split('_')[-2]))
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    window = int(SDUR * 1000)
    near = []
    for t in pos_df['start_ms']:
        mask = (neg_df['start_ms'] >= t - window) & (neg_df['start_ms'] <= t + window)
        nearby = neg_df[mask]
        if not nearby.empty:
            near.append(nearby.sample(n=min(2, len(nearby)), random_state=42))
    near_df = pd.concat(near) if near else pd.DataFrame(columns=df.columns)
    rest = neg_df.drop(near_df.index, errors='ignore')
    scatter_df = rest.sample(n=min(N_RANDOM_NEGS, len(rest)), random_state=42)
    return pd.concat([pos_df, near_df, scatter_df], ignore_index=True)[['filename', 'label']]

# ---- Main Generation ----
def make_spectrograms_with_labels(sourceAudioFolder, destinationImageFolder, allAnnotationsCsv, species_name):
    """
    Processes all .wav files in the source folder. For each:
    - Breaks into overlapping segments
    - Generates spectrogram PNGs
    - Assigns binary labels based on annotation overlap
    - Applies Beluga-specific negative sampling if needed
    
    Arguments:
    - sourceAudioFolder (str): path to the folder containing .wav files
    - destinationImageFolder (str): where to save the output spectrogram images
    - allAnnotationsCsv (str): path to the CSV with annotations (must include 'Begin Time (s)', 'End Time (s)', and 'audiofile')
    - species_name (str): used to trigger special logic (e.g., Beluga-specific downsampling)

    Returns:
    - DataFrame with two columns: 'filename' and 'label'
    """
    os.makedirs(destinationImageFolder, exist_ok=True)
    try:
        labels = pd.read_csv(allAnnotationsCsv)
    except Exception as e:
        logging.error(f"Failed to read {allAnnotationsCsv}: {e}")
        return pd.DataFrame()

    labels['time_segments'] = list(zip(labels['Begin Time (s)'], labels['End Time (s)']))
    files = [f for f in os.listdir(sourceAudioFolder) if f.endswith('.wav')]
    logging.info(f"Processing {len(files)} files in {sourceAudioFolder}")
    all_results = []
    for f in files:
        filepath = os.path.join(sourceAudioFolder, f)
        segs = list(labels[labels['audiofile'] == f]['time_segments'])
        results = extract_spectrograms_from_file(filepath, destinationImageFolder, segs, SDUR, OVERLAP)
        all_results.extend(results)
        logging.info(f"{f}: {len(results)} images created")

    df = pd.DataFrame(all_results, columns=['filename', 'label'])
    if species_name and species_name.lower() == 'beluga':
        df = sample_beluga_negatives(df)
        logging.info(f"Beluga class distribution after sampling: {df['label'].value_counts().to_dict()}")
    return df

def generate_spectrograms_and_labels_for_species_locations(species_name, location_names, annotation_csv_path=None):
    """
    For each location, generate labeled spectrograms from audio files using known annotations.
    Saves individual label CSVs for each location.
    
    Arguments:
    - species_name (str): The species to process (e.g., 'Beluga', 'Orca', 'Humpback').
    - location_names (list of str): Folder names under DataInput/{species_name}/ that contain audio files
                                    or subfolders with .wav files.
    - annotation_csv_path (str or None): Optional. Path to a CSV file with call annotations.
                                        If None, defaults to: ./DataInput/{species_name}/{species_name}_annotations.csv

    This function:
    - Resolves the correct folder containing .wav files for each location
    - Extracts 2s overlapping audio segments (default overlap: 0.4s)
    - Creates mel spectrogram images and labels them based on annotation overlap
    - For Beluga, it selectively samples negative segments to reduce imbalance
    - Saves one CSV per location with labels for each generated spectrogram
    """    
    logging.info(f"Starting for species: {species_name}")
    for name in location_names:
        base = f"./DataInput/{species_name}/{name}"
        try:
            src = resolve_audio_folder(base)
        except FileNotFoundError as e:
            logging.warning(str(e))
            continue
        dst = f"./DataInput/{species_name}/SpectrogramsOverlap{int(OVERLAP*1000)}ms/{name}"
        ann = annotation_csv_path or f"./DataInput/{species_name}/{species_name}_annotations.csv"
        df = make_spectrograms_with_labels(src, dst, ann, species_name)
        out_csv = f"./DataInput/{species_name}/{name.lower()}_{species_name.lower()}_overlap{int(OVERLAP*1000)}ms_spectrogram_labels.csv"
        df.to_csv(out_csv, index=False)
        logging.info(f"Saved labels to {out_csv}")
    logging.info(f"Completed for species: {species_name}")

def combine_label_csvs(species_name, overlap_ms=400):
    """
    Combine individual location label files into one master CSV with full paths and locations.
    """
    logging.info(f"Combining CSVs for {species_name}")
    base = f"./DataInput/{species_name}"
    pattern = os.path.join(base, f"*_{species_name.lower()}_overlap{overlap_ms}ms_spectrogram_labels.csv")
    files = glob.glob(pattern)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['location'] = df['filename'].str.split('_').str[0]
    df['dirpath'] = df['location'].apply(lambda x: os.path.join(base, f"SpectrogramsOverlap{overlap_ms}ms", x))
    df['fullpath'] = df.apply(lambda x: os.path.join(x['dirpath'], x['filename']), axis=1)
    out = os.path.join(base, f"{species_name.lower()}_spectrogram_labels_overlap{overlap_ms}ms.csv")
    df.to_csv(out, index=False)
    logging.info(f"Master CSV saved to {out}")

if __name__ == '__main__':
    # generate_spectrograms_and_labels_for_species_locations(
    #     species_name='Humpback',
    #     location_names=['AL16_BS4_humpback_data', 'LCI_Chinitna_humpback_data','LCI_Iniskin_humpback_data','LCI_Port_Graham_humpback_data'],
    #     annotation_csv_path=None
    # )

    # generate_spectrograms_and_labels_for_species_locations(
    #     species_name='Orca',
    #     location_names=['SWCorner', 'Chinitna','Iniskin','PtGraham'],
    #     annotation_csv_path=None
    # )

    #Ignoring 205D and 207D as per notes.txt
    generate_spectrograms_and_labels_for_species_locations(
        species_name='Beluga',
        location_names=['201D', '206D','213D','214D','215D'],
        annotation_csv_path=None
    )
