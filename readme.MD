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