import os
import pandas as pd
import logging
import datetime
import pytz

# Set up logging
os.makedirs('Logs', exist_ok=True)
log_file = f'Logs/make_spectrograms{datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")}.log'
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
            if file.endswith(('.txt', '.csv')):
                file_list.append(os.path.join(root, file))
# Filter files containing '5.1' in their names
file_list = [file for file in file_list if '5.1' in file]

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
            if file.endswith('.csv'):
                df = pd.read_csv(file, header=0, dtype=str, on_bad_lines='skip')
            else:
                df = pd.read_csv(file, sep='\t', header=0, dtype=str, on_bad_lines='skip')

            # Drop rows that repeat column names (common when headers reappear as data)
            df = df[df[df.columns[0]] != df.columns[0]]

            # Add metadata columns
            df['location'] = file.split('/')[-2]
            df['annotationfile'] = file
            df['audiofile'] = file.replace('.selections.txt', '.wav').split('/')[-1]

            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")

    if combined_df.empty:
        logging.error("No data combined, resulting dataframe is empty.")
        return

    # Remove duplicated columns (keep only the first occurrence)
    _, idx = np.unique(combined_df.columns, return_index=True)
    combined_df = combined_df.iloc[:, sorted(idx)]

    output_path = f'{annotations_folder_path}/{species}_annotations.csv'
    combined_df.to_csv(output_path, index=False)
    logging.info(f'{output_path} created')


# if __name__ == "__main__":
#     species = ['Humpback','Orca','Beluga']
#     for s in species:
#         if s != "Beluga":
#             singleAnnotationsFile(species=s)
#         # else:
#         #     os.system('python gather_beluga_annotations.py')