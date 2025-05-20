import os
import pandas as pd
import numpy as np
import librosa
import pytz
import bisect
from datetime import datetime, timedelta
import logging

# Optional: enable logging to file instead of print
USE_LOGGING = True
if USE_LOGGING:
    os.makedirs("Logs", exist_ok=True)
    log_file = f'Logs/beluga_annotation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    log = logging.info
else:
    log = print

def parsedt(timestr, with_decimals=False):
    formats = ['%Y-%m-%d %H:%M:%S.%f%z', '%Y-%m-%d %H:%M:%S%z']
    for fmt in formats:
        try:
            dt = datetime.strptime(timestr, fmt)
            if with_decimals:
                return dt  # return full datetime with tzinfo
            else:
                return dt.strftime('%y%m%d%H%M%S')  # used for wav file matching
        except ValueError:
            continue
    return None

def find_closest_preceding(a_list, b_list):
    """
    For each value in b_list, find the closest value in a_list that is <= b.
    Returns a list of closest preceding values, or None if none found.
    """
    # Clean and sort the reference list
    a_sorted = sorted(filter(pd.notnull, a_list))

    results = []
    for i, b in enumerate(b_list):
        if pd.isnull(b):
            results.append(None)
            continue

        pos = bisect.bisect_right(a_sorted, b)
        if pos == 0:
            results.append(None)  # No preceding element
        else:
            results.append(a_sorted[pos - 1])
    
    return results

def load_wav_file_metadata(wav_path):
    wav_files = []
    for root, _, files in os.walk(wav_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    sample_info = []
    for wav in wav_files:
        try:
            rate = librosa.get_samplerate(wav)
            sample_info.append((os.path.basename(wav), rate))
        except Exception as e:
            log(f"Error reading sample rate for {wav}: {e}")

    df_sample = pd.DataFrame(sample_info, columns=['audiofile', 'sampleRate'])
    wav_stamps = [os.path.basename(wav).split('.')[1] for wav in wav_files]
    prefix = os.path.basename(wav_files[0]).split('.')[0] if wav_files else None
    return wav_files, df_sample, wav_stamps, prefix

def compute_coverage(wav_files, annotations):
    try:
        total_audio_duration = sum(librosa.get_duration(path=f) for f in wav_files)
    except Exception as e:
        log(f"Error computing duration for audio files: {e}")
        total_audio_duration = 0

    total_annotated_duration = annotations['durationSeconds'].sum()
    coverage_ratio = total_annotated_duration / total_audio_duration if total_audio_duration > 0 else 0
    log(f"Total audio: {total_audio_duration:.1f}s, Annotated: {total_annotated_duration:.1f}s, Coverage: {coverage_ratio:.2%}")

def process_annotation_csv(folder_name, base_path, annotations_path):
    wav_path = os.path.join(base_path, folder_name)
    wav_files, sample_rates_df, wav_stamps, wav_prefix = load_wav_file_metadata(wav_path)

    ann_files = [f for f in os.listdir(annotations_path) if f.startswith(folder_name) and f.endswith('.csv')]
    if not ann_files:
        log(f"No annotation file found for {folder_name}")
        return pd.DataFrame()

    ann_df = pd.read_csv(os.path.join(annotations_path, ann_files[0]), low_memory=False)
    ann_df = ann_df[ann_df.Species == 'B'].copy()
    ann_df['location'] = folder_name
    ann_df['annotationstamp'] = ann_df['Local_Time'].apply(parsedt)
    ann_df['labelfile'] = ann_files[0]

    closest_stamps = find_closest_preceding(wav_stamps, ann_df['annotationstamp'].tolist())
    unmatched = [i for i, val in enumerate(closest_stamps) if val is None]
    if unmatched:
        log(f"{len(unmatched)} unmatched annotations in {folder_name} (no corresponding .wav found)")

    ann_df['audiofile'] = [
        f"{wav_prefix}.{stamp}.wav" if stamp is not None else "unmatched.wav"
        for stamp in closest_stamps
    ]
    ann_df = ann_df[ann_df['audiofile'] != "unmatched.wav"]

    if ann_df.empty:
        log(f"All annotations in {folder_name} were unmatched.")
        return pd.DataFrame()

    ann_df = ann_df.merge(sample_rates_df, on='audiofile', how='left')
    ann_df['durationSeconds'] = ann_df[['duration', 'sampleRate']].apply(lambda x: x['duration'] / x['sampleRate'], axis=1)

    # Convert Local_Time to datetime with tz awareness
    ann_df['Begin Time (s)'] = pd.to_datetime(ann_df['Local_Time'], format='mixed', utc=True)
    ann_df['End Time (s)'] = ann_df['Begin Time (s)'] + pd.to_timedelta(ann_df['durationSeconds'], unit='s')


    # Ensure both datetime columns are timezone-aware
    ann_df['UTC_Time'] = pd.to_datetime(ann_df['UTC_Time'], format='mixed', utc=True)
    ann_df['Local_Time'] = pd.to_datetime(ann_df['Local_Time'], format='mixed')  # keep original time zone

    # Parse .wav start time using Local_Time's tzinfo per row
    def infer_filestarttime(row):
        try:
            ts_local_naive = datetime.strptime(row['audiofile'].split('.')[1], '%y%m%d%H%M%S')
            local_tz = row['Local_Time'].tzinfo  # e.g., UTC-08:00 or UTC-07:00 depending on DST
            ts_local = ts_local_naive.replace(tzinfo=local_tz)
            return ts_local.astimezone(pytz.UTC)
        except Exception as e:
            print(f"Error parsing filestarttime for row {row.name}: {e}")
            return pd.NaT

    # Apply and compute startseconds
    ann_df['filestarttime'] = ann_df.apply(infer_filestarttime, axis=1)
    ann_df['startseconds'] = (ann_df['UTC_Time'] - ann_df['filestarttime']).dt.total_seconds()

    # Convert Begin and End Time (s) to float seconds since epoch
    ann_df['Begin Time (s)'] = ann_df['Begin Time (s)'].astype('int64') / 1e9
    ann_df['End Time (s)'] = ann_df['End Time (s)'].astype('int64') / 1e9

    log(f"Processed {folder_name}: {len(ann_df)} valid annotations from {ann_files[0]}")
    compute_coverage(wav_files, ann_df)
    return ann_df

def main():
    base = 'DataInput/Beluga'
    annotations_folder_path = f'/home/radodhia/ssdprivate/NOAAWhalesV2/{base}'
    annotations_path = os.path.join(annotations_folder_path, 'Annotations')
    wav_folders = sorted([
        f for f in os.listdir(annotations_folder_path)
        if f.startswith('2') and os.path.isdir(os.path.join(annotations_folder_path, f)) and f not in ['205D', '207D']
    ])

    combined = pd.DataFrame()
    for folder in wav_folders:
        df = process_annotation_csv(folder, annotations_folder_path, annotations_path)
        if not df.empty:
            combined = pd.concat([combined, df], ignore_index=True)

    combined.drop(columns=[c for c in ['detections_not_on_datamap', 'detection_not_on_datamap'] if c in combined.columns], inplace=True, errors='ignore')
    output_path = os.path.join(annotations_folder_path, 'Beluga_annotations.csv')
    combined.to_csv(output_path, index=False)
    # Summary: Counts and percentages per location
    if not combined.empty and 'location' in combined.columns:
        summary = combined.groupby('location').size().reset_index(name='count')
        total = summary['count'].sum()
        summary['percentage'] = 100 * summary['count'] / total
        log("\nAnnotation counts by location:")
        log(summary.to_string(index=False, formatters={'percentage': '{:,.2f}%'.format}))

        # # Optionally: write to file
        # summary_path = os.path.join(annotations_path, 'beluga_annotation_summary.csv')
        # summary.to_csv(summary_path, index=False)
        # log(f"Saved summary to {summary_path}")

    log(f"Saved combined annotations to {output_path}")

if __name__ == '__main__':
    main()
