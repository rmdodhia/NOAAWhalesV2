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
    spectrogram = librosa.feature.S
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


def singleAnnotationsFile():
    ''' 
    Combines selections.txt files (time segments of detected calls) into one file
    '''
    file_list = []
    for root, dirs, files in os.walk('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/Whales_Classification_2022/KillerWhale'):
        for file in files:
            if file.endswith('.txt'):
                file_list.append(os.path.join(root, file))

    # Combine the contents of all the text files into a single dataframe
    combined_df = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file,sep='\t')
        df['location'] = file.split('/')[-2]
        df['annotationfile'] = file
        df['audiofile'] = file.split('/')[-1].split('.')[0]+'.wav'
        combined_df = pd.concat([combined_df, df])

    # Save the combined dataframe to a CSV file
    combined_df.to_csv('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/killerwhale_annotations.csv', index=False)



###
HOW TO USE THIS FILE TO MAKE SPECTROGRAM IMAGES WITH labels

# make sure you have an annotation file like belugaAnnotations.csv or humpback_annotations.csv
## for beluga, use the file belugaAnnotations.py
## for humpback, and probably killer whale, use singleAnnotationsFile() in this script
## Required columns are
### Begin Time (s)
### End Time (s)
### location
### audiofile
### labelfile




# if __name__ == "__main__":
kwLocNaMES='Chinitna','Iniskin','PtGraham','SWCorner'

for name in kwLocNaMES:
    # name = kwLocNaMES[0]
    sourceAudioFolder = f'./DataInput/Whales_Classification_2022/KillerWhale/{name}'
    destinationImageFolder = f'./DataInput/KillerWhale/SpectrogramsOverlap400ms/{name}'
    allAnnotationsCsv = '/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/killerwhale_annotations.csv'


    ## Combine raw selections files for killer whales into one file
    # Find all the text files in the folders

    # Run the combined process for a specific location
    #df is the label file, each png will have 0 or 1
    df = make_spectrograms_with_labels(sourceAudioFolder, destinationImageFolder, allAnnotationsCsv, sdur=2, overlap=0.4)
    df.to_csv(f'/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/{name}_killerwhale_overlap400ms_spectrogram_labels.csv', index=False)


## combine all location-based label files into one file
# Get a list of all the csv files
file_list = glob.glob('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/*killerwhale_overlap400ms_spectrogram_labels.csv')

# Combine all the csv files into one dataframe
label = pd.concat([pd.read_csv(file) for file in file_list])

#add columns to label file to make it easier to split into train - validation - test
label['location'] = label['filename'].str.split('_').str[0].str.replace('AU-', '')
label['dirpath'] = label.apply(lambda x: os.path.join('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/SpectrogramsOverlap400ms', x['location']), axis=1)


label['fullpath'] = label.apply(lambda x: os.path.join('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/SpectrogramsOverlap400ms', x['location'], x['filename']), axis=1)
# label.groupby('location')['label'].count()
# label.groupby('location')['label'].sum()
# label.groupby('location')['label'].mean()

# Save the combined dataframe to a new csv file
label.to_csv('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/KillerWhale/killerwhale_spectrogram_labels_overlap400ms.csv', index=False)

