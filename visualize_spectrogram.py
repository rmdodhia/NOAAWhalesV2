import torchaudio
import librosa
import librosa.display
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse

def visualize_spectrogram_segment(wav_path, start_time_sec, duration):
    waveform_full, sr = torchaudio.load(wav_path)
    total_duration_sec = waveform_full.shape[1] / sr

    print(f"Sample rate: {sr}, audio length: {total_duration_sec:.2f} sec")
    print(f"Audio file duration: {total_duration_sec:.2f} seconds")
    print(f"Requested segment: start={start_time_sec}s, duration={duration}s")

    frame_offset = int(start_time_sec * sr)
    num_frames = int(duration * sr)

    if frame_offset >= waveform_full.shape[1]:
        raise ValueError(f"Start time {start_time_sec}s exceeds audio length of {total_duration_sec:.2f}s.")

    end_frame = min(frame_offset + num_frames, waveform_full.shape[1])
    waveform = waveform_full[:, frame_offset:end_frame]

    if waveform.shape[1] == 0:
        raise ValueError("Extracted waveform segment is empty. Check start time and duration.")

    audio_np = waveform[0].numpy()

    stft = torch.stft(
        waveform[0],
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.hann_window(1024),
        return_complex=True
    )
    stft_db = 20 * torch.log10(torch.abs(stft) + 1e-6)
    stft_db = torch.clamp(stft_db, min=-80, max=0).numpy()
    stft_db = (stft_db - stft_db.min()) / (stft_db.max() - stft_db.min()) * 255.0

    mel = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(stft_db, aspect='auto', origin='lower', cmap='magma')
    plt.title('STFT dB Spectrogram')
    plt.xlabel('Time frames')
    plt.ylabel('Frequency bins')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Mel Spectrogram (librosa)')
    plt.colorbar(format='%+2.0f dB')

    plt.suptitle(f"{os.path.basename(wav_path)} — {start_time_sec:.2f}s to {start_time_sec + duration:.2f}s")
    plt.tight_layout()
    plt.savefig("preview_spectrogram.png", dpi=150)
    print("Saved to preview_spectrogram.png")

    torchaudio.save("preview_audio_segment.wav", waveform, sample_rate=sr)
    print("Audio segment saved to preview_audio_segment.wav")

def process_csv_and_generate(csv_path, rownumber=0, duration=2.0):
    df = pd.read_csv(csv_path)

    if 'audiofile' not in df.columns or 'startSeconds' not in df.columns:
        raise ValueError("CSV must contain 'audiofile' and 'startSeconds' columns.")

    if rownumber >= len(df):
        raise IndexError(f"Row number {rownumber} exceeds number of rows in CSV.")

    row = df.iloc[rownumber]
    wav_path = "./DataInput/Beluga/201D/604536840/" + row['audiofile']
    start_seconds = row['startSeconds']

    print(f"Processing: {wav_path} at {start_seconds} seconds for {duration} seconds")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")

    visualize_spectrogram_segment(wav_path, start_seconds, duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file and generate spectrograms.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("--row", type=int, default=0, help="Row number to process (default: 0).")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of segment to extract in seconds (default: 2.0).")
    args = parser.parse_args()

    process_csv_and_generate(args.csv_path, rownumber=args.row, duration=args.duration)
