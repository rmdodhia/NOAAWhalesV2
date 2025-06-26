import pandas as pd
from collections import Counter

b= pd.read_csv('DataInput/Beluga/Beluga_annotations.csv', low_memory=False)
b.columns
b.shape

unique_wav_files = b['audiofile'].unique()
print(len(unique_wav_files))

a=Counter(b['audiofile'])
print(a.most_common(10))

print(a['604536840.171001003002.wav'])