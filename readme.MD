Run gather_humpback_orca_annotations.py
* reads annotations file per location
* combines into one file with and stores iin DataInput/{species}/
* stores positive annotations only (i.e., "1" indicating a call) 

Run gather_beluga_annotations.py
* reads annotations file per location
* combines into one file with and stores iin DataInput/{species}/
* stores positive annotations only (i.e., "1" indicating a call) 
* needs separate logic because of highly unbalanced data

Required columns for annotations files are 
 Begin Time (s)
 End Time (s)
 location
 audiofile
 labelfile


 make_spectrograms_and_labels.py
 Configuration and Logging:

Sets up parameters for how audio will be split and processed.
Initializes logging to record progress and issues.
Finding Audio Files:

Searches for folders containing .wav audio files for each location.
Processing Each Audio File:

For each .wav file, reads the audio and its sample rate.
If the file is large, processes it in smaller chunks to save memory.
Moves data to the GPU if available and appropriate, otherwise uses the CPU.
Loads annotation data for the audio file and determines which time intervals are of interest (including padding).
Computes a spectrogram for the audio chunk.
Slices the spectrogram into overlapping segments.
For each segment in a padded region, checks if it overlaps with any annotation (e.g., whale call).
Saves each segment as a .pt file (PyTorch tensor) in a species/location-specific folder, and records whether it is a positive (contains an event) or negative (does not) example.
Generating Label Files:

For each location, creates a CSV file listing all generated spectrogram segments, their labels (positive/negative), file paths, and metadata.
Combining Label Files:

After processing all locations for a species, combines all per-location label CSVs into a single {species}_annotations.csv file for easier downstream use.
Parameters and Their Meanings
species: The species being processed (e.g., "Beluga"). Used to determine input/output folders and file naming.
locations: List of location codes (e.g., ["201D", "206D"]) to process for the given species.
ann_csv: Path to the annotation CSV file. If not provided, a default path is used.
proportion: Fraction of annotation rows to use (for subsampling, e.g., 0.01 for 1%).
segment_duration: Length (in seconds) of each spectrogram segment (default: 2.0).
overlap: Amount of overlap (in seconds) between consecutive segments (default: 0.4).
n_pad: Amount of padding (in seconds) to add before and after each annotation interval (default: 5.0).
k_null: Not used in the visible code, but typically would control the number of negative (null) segments to sample per positive.
min_overlap_for_positive: Minimum overlap (in seconds) between a segment and an annotation for the segment to be labeled positive (default: 0.1).
large_file_samples: If an audio file has more than this many samples, it is processed in chunks (default: 15,000,000).
chunk_duration_s: Duration (in seconds) of each chunk when processing large files (default: 600, i.e., 10 minutes).