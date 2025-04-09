import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import bisect
import librosa

###
'''
The annotation files need an extra column indicating which wav file the annotations belong to.
From the Local_Time column in the annotations file, get the datetime stamp
From the wav file names, also get the datetime stamp 
Associate the annotation time stamp to the wav file
'''

def parsedt(dat,withDecSec=False):
    try:
        a=datetime.strptime(dat, '%Y-%m-%d %H:%M:%S.%f%z')
    except:
        try:
            a=datetime.strptime(dat, '%Y-%m-%d %H:%M:%S%z')
        except:
            None
    if withDecSec==True:
        return a
    else: 
        return a.strftime('%y%m%d%H%M%S')

def find_min_diff_sorted(a, b):
    # First, sort list 'a'
    a.sort()
    result = []
    
    for b_val in b:
        # Find the insertion point in 'a' for b_val using binary search
        pos = bisect.bisect_left(a, b_val)
        
        # We need to check the element just before the insertion point
        if pos > 0:  # There is an element less than b_val
            closest_val = a[pos - 1]
        else:
            closest_val = None  # No element in 'a' is less than b_val
        
        result.append(closest_val)
    
    return result

def makeAnnotationsDf(folder_name):
    base = f'/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/Beluga/'
    ann_path = f'{base}Annotations'
    wav_path = f'{base}{folder_name}'
    
    # Find all the wav files recursively under the given folder
    wav_files = []
    for root, dirs, files in os.walk(wav_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    #Get sample rates of each wav file, needed to calculate duration
    sample_rates = []
    for file in wav_files:
        # Get the sample rate of the current wav file
        sample_rate = librosa.get_samplerate(file)
        sample_rates.append([file.split('/')[-1],sample_rate])
    sample_rates_df = pd.DataFrame(sample_rates, columns=['audiofile', 'sampleRate'])

    #get wav file datetimestamp and suffix
    wav_file_stamps = [file.split('.')[1] for file in wav_files]
    wav_filename_prefix = wav_files[0].split('.')[0].split('/')[-1]

    # Find the annotation file starting with the given folder_name
    ann_files = [file for file in os.listdir(ann_path) if file.startswith(folder_name) and file.endswith('.csv')]
    if len(ann_files) == 0:
        return pd.DataFrame()
    ann_fn = ann_files[0]
    
    fn = os.path.join(ann_path, ann_fn)
    annotations = pd.read_csv(fn)
    annotations = annotations[annotations.Species=='B']    
    annotation_times = list(annotations.Local_Time)
    # Convert the annotation times to the desired format
    annotation_stamps = [parsedt(time) for time in annotation_times]
    annotations['location']=folder_name
    annotations['annotationstamp'] = annotation_stamps
    annotations['labelfile']=ann_fn
                
    #determine which wav files correspond to the annotation
    closest_stamps = find_min_diff_sorted(wav_file_stamps, annotation_stamps)
    annotations['audiofile'] = [f"{wav_filename_prefix}.{stamp}.wav" for stamp in closest_stamps]
    #add end time, get sample rate, divide duration by sample rate to get duration in seconds

    #add sample rate to annotations, get duration in seconds, calculate end time, and rename startSeconds to begin time
    annotations = pd.merge(annotations, sample_rates_df, on='audiofile', how='left')
    annotations['durationSeconds'] = annotations[['duration', 'sampleRate']].apply(lambda x: x['duration'] / x['sampleRate'], axis=1)
    annotations['Begin Time (s)'] = annotations['Local_Time'].apply(lambda x: parsedt(x,withDecSec=True))
    annotations['End Time (s)'] = annotations.apply(lambda row: row['Begin Time (s)'] + timedelta(seconds=row['durationSeconds']), axis=1)

    # Convert Begin Time (s) and End Time (s) to datetime
    timezone_utc_minus_8 = pytz.timezone('Etc/GMT+8')
    annotations['filestarttime'] = annotations['audiofile'].apply(lambda x: datetime.strptime(x.split('.')[1], '%y%m%d%H%M%S').replace(tzinfo=timezone_utc_minus_8))
    
    annotations['Begin Time (s)'] = pd.to_datetime(annotations['Begin Time (s)'],utc=True)
    annotations['End Time (s)'] = pd.to_datetime(annotations['End Time (s)'],utc=True)
    
    
    annotations['startseconds'] = (annotations['Begin Time (s)'] - annotations['filestarttime']).dt.total_seconds()
    # Create a dictionary to map wav file names to their datetime stamps

    print(f'annotations file: {ann_fn}    wav folder: {root}    {dirs}')
    
    return annotations
    


wavFolders=[folder for folder in os.listdir('DataInput/Beluga') if folder.startswith('2') and os.path.isdir(os.path.join('DataInput/Beluga', folder))]
#Ignoring 205D and 207D as per Manolo in notes.txt
wavFolders.remove('205D')
wavFolders.remove('207D')
wavFolders.sort()

#combine all annotation files into one dataframe
annotationsdf=pd.DataFrame()
for i in wavFolders:
    print(i)
    temp=makeAnnotationsDf(i)
    if not temp.empty:
        # Append the current dataframe to the main dataframe
        if not annotationsdf.empty:
            annotationsdf = pd.concat([annotationsdf,temp], ignore_index=True)
        else:
            annotationsdf = temp


#drop unnecessary colums
annotationsdf=annotationsdf.drop('detections_not_on_datamap',axis=1)
annotationsdf=annotationsdf.drop('detection_not_on_datamap',axis=1)

annotationsdf.to_csv('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/Beluga/Annotations/belugaAnnotations.csv')


