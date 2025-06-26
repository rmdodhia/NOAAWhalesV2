import os
import pandas as pd
import numpy as np
import logging
import datetime
import pytz

# Set up logging
os.makedirs('Logs', exist_ok=True)
log_file = f'Logs/gather_humpback_orca_annotations{datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")}.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

def singleAnnotationsFile(species):
    ''' 
    Combines selections.txt and .csv files (time segments of detected calls) into one file
    '''
    annotations_folder_path = f"/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/{species}/"

    if not os.path.exists(annotations_folder_path):
        logging.error(f"Source folder {annotations_folder_path} does not exist.")
        return

    logging.info(f'Starting on {species}, looking here: {annotations_folder_path}')
    file_list = []
    for root, dirs, files in os.walk(annotations_folder_path):
        for file in files:
            if file.endswith(('.txt', '.csv')) and file != f"{species}_annotations.csv":
                file_list.append(os.path.join(root, file))

    if not file_list:
        logging.error("No annotation files found.")
        return
    else: 
        logging.info(f'Found {len(file_list)} files. {file_list[:(min(4,len(file_list)-1))]}')

    combined_df = pd.DataFrame()
    for file in file_list:
        if not os.path.exists(file):
            logging.warning(f"File {file} does not exist, skipping.")
            continue

        try:
            # Read the file
            initial_lines = sum(1 for _ in open(file))
            if file.endswith('.csv'):
                df = pd.read_csv(file, header=0, dtype=str, on_bad_lines='skip')
            else:
                df = pd.read_csv(file, sep='\t', header=0, dtype=str, on_bad_lines='skip')
            final_lines = df.shape[0]
            skipped = initial_lines - final_lines - 1  # subtract header
            if skipped > 0:
                warning_msg = f"⚠️  Skipped {skipped} bad line(s) in file: {file}"
                print(warning_msg)
                logging.warning(warning_msg)

            # Drop repeated headers
            df = df[df[df.columns[0]] != df.columns[0]]

            # Add metadata
            df['location'] = os.path.basename(os.path.dirname(file))
            df['annotationfile'] = file

            # Derive full audiofile path
            # Derive full audiofile path
            annotation_name = os.path.basename(file)
            if annotation_name.endswith('.Table.1.selections.txt'):
                audiofile_name = annotation_name.replace('.Table.1.selections.txt', '.wav')
            elif annotation_name.endswith('.selections.txt'):
                audiofile_name = annotation_name.replace('.selections.txt', '.wav')
            elif annotation_name.endswith('.csv'):
                audiofile_name = annotation_name.replace('.csv', '.wav')
            else:
                audiofile_name = annotation_name.replace('.txt', '.wav')

            annotation_dir = os.path.dirname(file)
            parent_dir = os.path.dirname(annotation_dir)
            location_name = os.path.basename(annotation_dir)

            if '_selections' in location_name:
                audio_dirname = location_name.replace('_selections', '_data')
            else:
                audio_dirname = location_name

            audio_dir = os.path.join(parent_dir, audio_dirname)
            full_audio_path = os.path.join(audio_dir, audiofile_name)

            df['audiofile'] = full_audio_path

            combined_df = pd.concat([combined_df, df], ignore_index=True)

        except Exception as e:
            logging.warning(f"Failed to process file {file}: {e}")

    if combined_df.empty:
        logging.error("No data combined, resulting dataframe is empty.")
        return

    # Remove duplicated columns (keep first occurrence)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    combined_df['startSeconds'] = combined_df['Begin Time (s)'].astype(float)

    output_path = os.path.join(annotations_folder_path, f'{species}_annotations.csv')
    combined_df.to_csv(output_path, index=False)
    logging.info(f'{output_path} created')

if __name__ == "__main__":
    species = ['Humpback', 'Orca', 'Beluga']
    for s in species:
        if s != "Beluga":
            singleAnnotationsFile(species=s)
        # else:
        #     os.system('python gather_beluga_annotations.py')
