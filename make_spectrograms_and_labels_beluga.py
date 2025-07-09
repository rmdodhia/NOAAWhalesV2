# make_spectrograms_and_labels.py

import os
import glob
import math
import time
import torch
import torchaudio
import logging
import datetime
import pytz
import numpy as np
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
SDUR      = 2.0        # segment duration (s)
OVERLAP   = 0.4        # overlap    (s)
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
HOP_LEN   = 256        # must match STFT hop_length
N_FFT     = 1024       # must match STFT n_fft

# ─── LOGGING ───────────────────────────────────────────────────────────────
os.makedirs('Logs', exist_ok=True)
today = datetime.datetime.now().strftime('%Y-%m-%d')
local_time = datetime.datetime.now().strftime('%H%M%S')
logging.basicConfig(
    filename=f"Logs/make_spectrograms_and_labels_{today}_{local_time}.log",
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    force=True
)

# ─── UTILITIES ────────────────────────────────────────────────────────────
def resolve_audio_folder(base):
    for d,_,fs in os.walk(base):
        if any(f.endswith('.wav') for f in fs):
            return d
    raise FileNotFoundError(f"No .wav under {base}")

# ─── CORE WORK ─────────────────────────────────────────────────────────────

def process_wav_file(
    wav_path,
    dst_folder,
    ann_df,
    segment_duration=2.0,
    overlap=0.4,
    n_pad=5.0,
    k_null=3,
    min_overlap_for_positive=0.1,
    large_file_samples=15_000_000,  # ~600s at 24kHz
    chunk_duration_s=600           # 10 minutes per chunk
):
    import math
    wav_info = torchaudio.info(wav_path)
    sr = wav_info.sample_rate
    total_samples = wav_info.num_frames
    file_duration = total_samples / sr
    audiofile = os.path.basename(wav_path)
    os.makedirs(dst_folder, exist_ok=True)

    # --- Get all annotation intervals for this file and build padded intervals ---
    sub = ann_df[ann_df["audiofile"] == audiofile]
    annotation_intervals = [(row["startseconds"], row["startseconds"] + row["durationSeconds"]) for _, row in sub.iterrows()]
    padded_intervals = [
        (
            int(np.floor(max(row["startseconds"] - n_pad, 0))),
            int(np.ceil(min(row["startseconds"] + row["durationSeconds"] + n_pad, file_duration)))
        )
        for _, row in sub.iterrows()
    ]

    def merge_intervals(intervals):
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for current in sorted_intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    merged_padded = merge_intervals(padded_intervals)

    def overlaps_any_anno(t0, t1, intervals, min_overlap):
        return any(max(0, min(b, t1) - max(a, t0)) >= min_overlap for a, b in intervals)

    def process_padded_interval(start, end):
        try:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            # Clamp to file bounds
            start_sample = max(0, min(start_sample, total_samples))
            end_sample = max(0, min(end_sample, total_samples))
            num_samples = end_sample - start_sample
            if num_samples <= 0:
                logging.warning(f"Interval {start:.2f}-{end:.2f}s (samples {start_sample}:{end_sample}) is empty or out of bounds for {audiofile}")
                return []
            try:
                wave_tensor, _ = torchaudio.load(
                    wav_path,
                    frame_offset=start_sample,
                    num_frames=num_samples
                )
            except RuntimeError as e:
                logging.warning(
                    f"Partial load failed at {start_sample}:{end_sample} for {audiofile}, falling back to full load: {e}"
                )
                try:
                    full_wave, _ = torchaudio.load(wav_path)
                    wave_tensor = full_wave[:, start_sample:end_sample]
                except Exception as e2:
                    logging.error(f"Full load also failed for {audiofile}: {e2}")
                    return []
            # Handle multi-channel audio robustly
            if wave_tensor.ndim == 2 and wave_tensor.shape[0] > 1:
                wave = wave_tensor.mean(dim=0)
            else:
                wave = wave_tensor[0]
            chunk_start_s = start
            chunk_samples = wave.shape[-1]
            try:
                if DEVICE == 'cuda' and chunk_samples <= large_file_samples:
                    wave = wave.to(DEVICE).half()
                    device = DEVICE
                    dtype = torch.float16
                    logging.info(f"Processing padded interval {start:.2f}-{end:.2f}s on GPU (float16): {chunk_samples} samples")
                else:
                    wave = wave.cpu()
                    device = 'cpu'
                    dtype = torch.float32
                    logging.info(f"Processing padded interval {start:.2f}-{end:.2f}s on CPU: {chunk_samples} samples")
            except RuntimeError as oom:
                logging.warning(f"CUDA OOM for interval {start:.2f}-{end:.2f}s, retrying on CPU: {oom}")
                wave = wave.cpu().float()
                device = 'cpu'
                dtype = torch.float32
            try:
                spec = torch.stft(
                    wave,
                    n_fft=N_FFT,
                    hop_length=HOP_LEN,
                    win_length=N_FFT,
                    window=torch.hann_window(N_FFT, device=device, dtype=dtype),
                    return_complex=True
                )
                mag = spec.abs()
                db = 20 * torch.log10(mag + 1e-6)
                db = torch.clamp(db, min=-80, max=0).cpu()
            except Exception as e:
                logging.error(f"STFT failed for {audiofile} interval {start:.2f}-{end:.2f}s: {e}")
                return []
            seg_frames = math.ceil((segment_duration * sr) / HOP_LEN)
            step_frames = math.ceil(((segment_duration - overlap) * sr) / HOP_LEN)
            try:
                slices = db.unfold(dimension=1, size=seg_frames, step=step_frames)
                slices = slices.permute(1, 0, 2)
            except Exception as e:
                logging.error(f"Spectrogram slicing failed for {audiofile} interval {start:.2f}-{end:.2f}s: {e}")
                return []
            num_segments = slices.shape[0]
            stride = segment_duration - overlap
            results = []
            for i in range(num_segments):
                seg_start = i * stride
                seg_end = seg_start + segment_duration
                abs_seg_start = seg_start + chunk_start_s
                abs_seg_end = seg_end + chunk_start_s
                arr = slices[i].numpy()
                label = int(overlaps_any_anno(abs_seg_start, abs_seg_end, annotation_intervals, min_overlap_for_positive))
                fname = f"{audiofile.replace('.wav','')}_{int(abs_seg_start*1000)}_{int(abs_seg_end*1000)}.pt"
                fpath = os.path.join(dst_folder, fname)
                try:
                    torch.save(torch.from_numpy(arr), fpath)
                except Exception as e:
                    logging.error(f"Failed to save tensor for {audiofile} segment {abs_seg_start:.2f}-{abs_seg_end:.2f}s: {e}")
                    continue
                results.append((fname, label, audiofile))
            return results
        except Exception as e:
            logging.error(f"process_padded_interval failed for {audiofile} interval {start:.2f}-{end:.2f}s: {e}")
            return []

    all_results = []
    for (start, end) in merged_padded:
        all_results.extend(process_padded_interval(start, end))
    n_pos = sum(1 for _, label in all_results if label == 1)
    n_neg = sum(1 for _, label in all_results if label == 0)
    logging.info(f"Processed file {audiofile}: {n_pos} positive, {n_neg} negative images generated")
    return all_results


def generate_spectrograms_and_labels(
    species, locations, ann_csv=None, proportion=1.0,
    segment_duration=2.0, overlap=0.4, n_pad=5.0, k_null=3, min_overlap_for_positive=0.1
):
    ann_csv = ann_csv or f"./DataInput/{species}/{species}_annotations.csv"
    logging.info(f"Loading annotations from {ann_csv}")
    ann_df = pd.read_csv(ann_csv, low_memory=False)
    logging.info(f"Loaded {len(ann_df)} annotation rows")
    if proportion < 1.0:
        ann_df = ann_df.sample(frac=proportion, random_state=88)
        logging.info(f"Sampled {len(ann_df)} annotation rows (proportion={proportion})")

    for loc in locations:
        src = resolve_audio_folder(f"./DataInput/{species}/{loc}")
        dst = f"./DataInput/{species}/SpectrogramsOverlap{int(overlap*1000)}ms/{loc}"

        # Only process WAVs referenced in ann_df for this location
        wavs_in_ann = set(ann_df[ann_df['location'] == loc]['audiofile'].unique())
        wavs_to_process = [os.path.join(src, w) for w in wavs_in_ann if os.path.exists(os.path.join(src, w))]

        logging.info(f"\n\nProcessing location {loc}: src={src}, dst={dst}, {len(wavs_to_process)} wavs")
        all_out = []
        for wav in sorted(wavs_to_process):
            out = process_wav_file(
                wav_path=wav, dst_folder=dst, ann_df=ann_df,
                segment_duration=segment_duration,
                overlap=overlap,
                n_pad=n_pad,
                k_null=k_null,
                min_overlap_for_positive=min_overlap_for_positive
            )
            all_out.extend(out)

        # Write label CSV for this location
        df = pd.DataFrame(all_out, columns=['filename', 'label', 'audiofile'])
        df['location'] = loc
        df['dirpath'] = dst
        df['fullpath'] = df['filename'].apply(lambda fn: os.path.join(dst, fn))
        df['species'] = species.lower()
        csv_out = f"./DataInput/{species}/LabelsOverlap400ms/{species}_{loc}_overlap{int(overlap*1000)}ms_spectrogram_labels.csv"
        os.makedirs(os.path.dirname(csv_out), exist_ok=True)
        df.to_csv(csv_out, index=False)
        logging.info(f"Saved labels to {csv_out}")

    # After all locations processed, combine all label CSVs for this species into one file
    pattern = f"./DataInput/{species}/LabelsOverlap400ms/{species}_*_overlap{int(overlap*1000)}ms_spectrogram_labels.csv"
    all_label_files = glob.glob(pattern)
    if all_label_files:
        all_df = pd.concat([pd.read_csv(f) for f in all_label_files], ignore_index=True)
        combined_csv = f"./DataInput/{species}/LabelsOverlap400ms/{species}_labels.csv"
        all_df.to_csv(combined_csv, index=False)
        logging.info(f"Combined all label files into {combined_csv}")

# ─── RUNNER ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_spectrograms_and_labels(
        species   = "Beluga",
        # locations = ["201D","206D","213D","214D","215D"],
        locations = ["215D","223D"],
        ann_csv   = None,
        proportion = 0.25
    )

    # # adjust species & overlap_ms as needed
    # species    = "Beluga"
    # overlap_ms = 400
    # base       = f"./DataInput/{species}"
    # pattern    = f"{base}/*_{species.lower()}_overlap{overlap_ms}ms_spectrogram_labels.csv"

    # all_df = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)], ignore_index=True)
    # all_df.to_csv(f"{base}/{species.lower()}_spectrogram_labels_overlap{overlap_ms}ms.csv", index=False)

    # Check if any 'audiofile' entry begins with '604536840' and show row indices
