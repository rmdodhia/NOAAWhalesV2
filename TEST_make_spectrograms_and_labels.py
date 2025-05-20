import os
import torchaudio
import torch
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool
import logging
import datetime
import pytz
import glob
import multiprocessing as mp

# Set multiprocessing to use 'spawn' so CUDA works in subprocesses
mp.set_start_method('spawn', force=True)

# ---- Config ----
SDUR = 2        # duration of each audio segment in seconds
OVERLAP = 0.4   # overlap between segments in seconds
N_RANDOM_NEGS = 1000  # number of random negative samples to draw for Beluga
SAMPLE_RATE = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Logging ----
os.makedirs('Logs', exist_ok=True)

# Generate log filename with date and run number
current_date = datetime.datetime.now(pytz.timezone('UTC')).strftime('%Y-%m-%d')
log_files = glob.glob(f"Logs/make_spectrograms_{current_date}_*.log")
run_number = len(log_files) + 1
log_file = f"Logs/make_spectrograms_{current_date}_{run_number}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s', force=True)

# ---- Utilities ----
def resolve_audio_folder(base_path):
    for root, _, files in os.walk(base_path):
        if any(f.endswith('.wav') for f in files):
            return root
    raise FileNotFoundError(f"No .wav files found under {base_path}")

# ---- STFT Spectrogram Generation ----
def generate_stft_spectrogram_batch(segments):
    """
    Compute STFT spectrograms for a batch of audio segments.
    segments: list of 1D numpy arrays
    Returns a list of 2D numpy arrays (spectrograms)
    """
    batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in segments]).to(DEVICE)
    spec = torch.stft(
        batch,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.hann_window(1024).to(DEVICE),
        return_complex=True
    )
    spec = torch.abs(spec)
    spec = 20 * torch.log10(spec + 1e-6)
    spec = torch.clamp(spec, min=-80, max=0)
    return spec.cpu().numpy()


def batch_spectrogram_to_images(batch_args):
    """
    Process a batch of segments and save corresponding spectrogram images.
    """
    segments, info_list, time_segments = batch_args
    spectrograms = generate_stft_spectrogram_batch(segments)
    output = []
    for i, (img, info) in enumerate(zip(spectrograms, info_list)):
        filestem, destinationImageFolder, index, start, end = info
        output_filename = f"{filestem}_{index}_{start}_{end}.pt"
        output_path = os.path.join(destinationImageFolder, output_filename)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        torch.save(torch.tensor(img, dtype=torch.uint8), output_path.replace('.png', '.pt'))

        overlap_check = any(
            max(0, min(seg[1], end / 1000) - max(seg[0], start / 1000)) >= 0.2
            for seg in time_segments
        )
        label = 1 if overlap_check else 0
        output.append((output_filename, label))
    return output


total_gpu_time = 0.0
total_io_time = 0.0
def extract_spectrograms_from_file(filepath, destinationImageFolder, time_segments, segment_duration, overlap, batch_size=64):
    """
    Extract spectrograms from a .wav file using batched GPU processing.
    """
    waveform, sr = torchaudio.load(filepath)
    audio = waveform[0].numpy()
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    filestem = os.path.basename(filepath).split('.')[0]

    segments, info_list, batches = [], [], []
    start, index = 0, 0
    while start + segment_samples <= len(audio):
        startms = round(start / sr * 1000)
        endms = round((start + segment_samples) / sr * 1000)
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        info_list.append((filestem, destinationImageFolder, index, startms, endms))
        start += (segment_samples - overlap_samples)
        index += 1

        if len(segments) >= batch_size:
            batches.append((segments, info_list, time_segments))
            segments, info_list = [], []

    if segments:
        batches.append((segments, info_list, time_segments))

    results = []
    for batch in batches:
        batch_result = batch_spectrogram_to_images(batch)
        results.extend(batch_result)

    logging.info(f"Total segments processed from {filepath}: {len(results)}")
    logging.info(f"Cumulative GPU time: {total_gpu_time:.2f}s | Cumulative I/O time: {total_io_time:.2f}s")
    return results

# ---- Threaded Image Saving ----
from concurrent.futures import ThreadPoolExecutor
import time

def save_image_threaded(args):
    img, output_path = args
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)
    Image.fromarray(img).save(output_path)

# ---- STFT Spectrogram Generation ----
def batch_spectrogram_to_images(batch_args):
    """
    Process a batch of segments and save corresponding spectrogram images.
    """
    segments, info_list, time_segments = batch_args
    start_gpu = time.time()
    spectrograms = generate_stft_spectrogram_batch(segments)
    gpu_time = time.time() - start_gpu

    tasks, output = [], []
    label_start = time.time()

    for img, info in zip(spectrograms, info_list):
        filestem, destinationImageFolder, index, start, end = info
        output_filename = f"{filestem}_{index}_{start}_{end}.png"
        output_path = os.path.join(destinationImageFolder, output_filename)
        tasks.append((img, output_path))

        overlap_check = any(
            max(0, min(seg[1], end / 1000) - max(seg[0], start / 1000)) >= 0.2
            for seg in time_segments
        )
        label = 1 if overlap_check else 0
        output.append((output_filename, label))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_image_threaded, tasks))

    io_time = time.time() - label_start
    logging.info(f"Batch timing — GPU: {gpu_time:.2f}s | I/O + Label: {io_time:.2f}s")
    global total_gpu_time, total_io_time
    total_gpu_time += gpu_time
    total_io_time += io_time
    return output

# ---- Main Segment Extraction ----
def extract_spectrograms_from_file(filepath, destinationImageFolder, time_segments, segment_duration, overlap, batch_size=64):
    """
    Extract spectrograms from a .wav file using batched GPU processing and threaded image saving.
    """
    waveform, sr = torchaudio.load(filepath)
    audio = waveform[0].numpy()
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap * sr)
    filestem = os.path.basename(filepath).replace('.wav', '')


    segments, info_list, batches = [], [], []
    start, index = 0, 0
    while start + segment_samples <= len(audio):
        startms = round(start / sr * 1000)
        endms = round((start + segment_samples) / sr * 1000)
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        info_list.append((filestem, destinationImageFolder, index, startms, endms))
        start += (segment_samples - overlap_samples)
        index += 1

        if len(segments) >= batch_size:
            batches.append((segments, info_list, time_segments))
            segments, info_list = [], []

    if segments:
        batches.append((segments, info_list, time_segments))

    results = []
    for batch in batches:
        batch_result = batch_spectrogram_to_images(batch)
        results.extend(batch_result)

    logging.info(f"Total segments processed from {filepath}: {len(results)}")
    logging.info(f"Cumulative GPU time: {total_gpu_time:.2f}s | Cumulative I/O time: {total_io_time:.2f}s")
    return results

def sample_beluga_negatives(df):
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

def make_spectrograms_with_labels(sourceAudioFolder, destinationImageFolder, allAnnotationsCsv, species_name):
    os.makedirs(destinationImageFolder, exist_ok=True)
    try:
        labels = pd.read_csv(allAnnotationsCsv)
    except Exception as e:
        logging.error(f"Failed to read {allAnnotationsCsv}: {e}")
        return pd.DataFrame()

        nan_count = labels['Begin File'].isna().sum()
        logging.info(f"Number of NaN values in 'Begin File' column: {nan_count}")

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

    generate_spectrograms_and_labels_for_species_locations(
        species_name='Beluga',
        location_names=['201D', '206D','213D','214D','215D'],
        annotation_csv_path=None
    )
