import os
import pandas as pd
import numpy as np
import librosa
from multiprocessing import Pool
import glob

def combine_audio_timestamp_files(sourceAudioFolder):
    """
    Combine all txt files in the sourceAudioFolder into a single file.
    """
    locations = [i for i in os.listdir(sourceAudioFolder) if os.path.isdir(os.path.join(sourceAudioFolder, i)) and 'selections' in i]
    
    out=pd.DataFrame()
    for loc in locations:
        print(loc)
        for i in os.listdir(os.path.join(sourceAudioFolder, loc)):
            if i.endswith('.txt'):
                temp=pd.read_csv(os.path.join(sourceAudioFolder,loc,i),sep='\t')
                temp['location']=loc
                temp['labelfile']=i
                try:
                    temp['audiofile']=temp['Begin File']   
                except:
                    temp['audiofile'] =  i.split(".")[0] + ".wav"
                out = pd.concat([out,temp],ignore_index=True)
    return out

'''
def load_audio_metadata(file_path):
    """
    Load metadata for a single audio file using librosa
    Set sr=None otherwise sampel rate changed to 22050
    """
    y, sr = librosa.load(file_path, sr=None)  # Load using the original sample rate
    total_length = len(y) / sr
    return file_path, os.path.basename(file_path), total_length, sr
'''

def process_audio_file(args):
    """
    Generate labels for each segment of an audio file and construct the filename for spectrogram.
    """
    file_path, time_segments, sdur, overlap, total_length, sample_rate = args
    segments = [(start, start + sdur) for start in np.arange(0, total_length, sdur - overlap)]
    result = []
    for i, segment in enumerate(segments):
        #check if time segment in labels file overlaps by at least 0.2 seconds with time segment based os sdur, overlap, total_length, and sample_rate
        overlap_check = any(
            max(0, min(seg[1], segment[1]) - max(seg[0], segment[0])) >= 0.2
            for seg in time_segments
        )
        label = 1 if overlap_check else 0
        #
        filestem = os.path.basename(file_path).split('.')[0]
        loc = os.path.basename(os.path.dirname(file_path)).replace(' ', '').lower()
 #      If segment is in sample rate counts
 #      filename = f"{loc}/{filestem}_{i}_{round(segment[0]*sample_rate)}_{round(segment[1]*sample_rate)}.png"
        filename = f"{loc}/{filestem}_{i}_{round(segment[0]*1000)}_{round(segment[1]*1000)}.png"
        result.append((filename, label))
    return result

def create_spectrogram_label_dataframe_parallel(sourceAudioFolder, allLabelsCsv, sdur=2, overlap=0):
    """
    Process each audio file in parallel to generate labels and filenames for spectrograms.
    """
    audio_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(sourceAudioFolder) for f in filenames if f.endswith('.wav')]
    labels = pd.read_csv(allLabelsCsv)
    labels['time_segments'] = list(zip(labels['Begin Time (s)'], labels['End Time (s)']))

    tasks = [
        (
            file_path,
            list(labels[labels['audiofile'] == os.path.basename(file_path)]['time_segments']),
            sdur,
            overlap,
            librosa.get_duration(path=file_path),  # Get total length using librosa
            librosa.get_samplerate(file_path)  # Get sample rate using librosa
        )
        for file_path in audio_files 
        ]
    
    #results will be a list of lists
    with Pool(os.cpu_count()) as pool:
        results = pool.map(process_audio_file, tasks)
    
    # Flatten the results and create a DataFrame
    flat_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_results, columns=['filename', 'label'])
    return df

#change root_dir to path for Humpback, Orca, Beluga
root_dir = '/home/radodhia/ssdprivate/NOAA_Whales/DataInput/Humpback/SpectrogramsOverlap400ms'
existing_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in glob.glob(os.path.join(dirpath, '*.png')):
        # Normalize the file path to create a comparable identifier, e.g., remove the root_dir part
        normalized_path = file.replace(root_dir + '/', '')
        existing_files.append(normalized_path)
existing_files_set = set(existing_files)

# Normalize identifiers in df['filename']
df_files = ['_'.join(i.split('/')[1].split('_')[:3]) for i in list(df['filename'])]
df_files = set(list(df['filename']))

# Compare the sets
files_in_df_not_in_directory = df_files - existing_files_set
files_in_directory_not_in_df = existing_files_set - df_files

print("Files in df not in directory:", len(files_in_df_not_in_directory))
print("Files in directory not in df:", len(files_in_directory_not_in_df))






# sourceAudioFolder = './DataInput/Whales_Classification_2022/Killer_whale/'
# allLabelsCsv = './DataInput/Whales_Classification_2022/Killer_whale/killer_whale_labels.csv'
sourceAudioFolder = '/home/radodhia/ssdprivate/NOAA_Whales/DataInput/Whales_Classification_2022/Humpback/'
allLabelsCsv = '/home/radodhia/ssdprivate/NOAA_Whales/DataInput/Humpback/humpback_labels.csv'

#Run this first to create a single database of input labels
temp=combine_audio_timestamp_files(sourceAudioFolder)
temp.to_csv('DataInput/Humpback/humpback_labels.csv')
#Run this second to attach labels to spectrograms
df = create_spectrogram_label_dataframe_parallel(sourceAudioFolder, allLabelsCsv, sdur=2, overlap=0.4)
df.to_csv('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/Humpback/humpball_all_spectrogram_labels.csv')
###
image_folder = './DataInput/Humpback/SpectrogramsOverlap400ms'
# Check if filenames in the label file exist in the subfolders of the image folder, and vice versa.
# Get the list of filenames from the label file
label_filenames = df['filename'].tolist()
label_filenames = [i.split('/')[1] for i in label_filenames]

# Get the list of filenames from the image folder
image_filenames = []
locations=os.listdir(image_folder)
for loc in locations:
    temp = os.listdir(os.path.join(image_folder,loc))
    image_filenames.append(temp)
# Flatten image_filenames into a single list
image_filenames = [filename for sublist in image_filenames for filename in sublist]

f=set(label_filenames)
i=set(image_filenames)

df = df[~df['filename'].str.split('/').str[1].isin(f - i)]
df['encounter'] = df['filename'].str.split('/').str[1].str.split('_').str[:3].str.join('_')

df.to_csv(os.path.join(sourceAudioFolder, 'killer_whale_spectrogram_labels_overlap400ms.csv'), index=False)

