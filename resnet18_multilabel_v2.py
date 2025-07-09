import os
import logging
import pandas as pd
import time
import datetime
import pytz
import traceback
import csv
from itertools import combinations
import random
from collections import Counter
from PIL import Image
import numpy as np

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities import rank_zero_only
import torch.nn.functional as F

# Function to get data subset from labels df

def get_data_from_csv(df, location, data_use_proportion=1.0):
    logging.info(f'Reading {data_use_proportion} of location {location} from labels dataframe')
    df_location = df[df['location'] == location]
    if data_use_proportion < 1.0:
        df_location = df_location.sample(frac=data_use_proportion, random_state=42)
    image_paths = df_location['fullpath'].tolist()
    labels = df_location['multilabel'].tolist()
    logging.info(f'Returning {len(image_paths)} image_paths and labels\n')
    return image_paths, labels

## Custom tensor-based augmentations for spectrograms
class SpectrogramAugmentations:
    def __init__(self, 
                 horizontal_shift_range=0.1,
                 occlusion_prob=0.1,
                 occlusion_max_lines=3,
                 noise_prob=0.1,
                 noise_std=0.01,
                 buffer_prob=0.1,
                 buffer_max_ratio=0.1):
        self.horizontal_shift_range = horizontal_shift_range
        self.occlusion_prob = occlusion_prob
        self.occlusion_max_lines = occlusion_max_lines
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.buffer_prob = buffer_prob
        self.buffer_max_ratio = buffer_max_ratio
    
    def horizontal_shift(self, spec):
        """Randomly shift spectrogram horizontally (time axis)"""
        if torch.rand(1) < 0.1:  # 50% chance
            _, _, time_dim = spec.shape
            shift_pixels = int(torch.randint(-int(time_dim * self.horizontal_shift_range), 
                                           int(time_dim * self.horizontal_shift_range) + 1, (1,)))
            if shift_pixels != 0:
                spec = torch.roll(spec, shifts=shift_pixels, dims=2)
        return spec
    
    def add_occlusions(self, spec):
        """Add random thin line occlusions (frequency masking)"""
        if torch.rand(1) < self.occlusion_prob:
            _, freq_dim, time_dim = spec.shape
            num_lines = torch.randint(1, self.occlusion_max_lines + 1, (1,)).item()
            
            for _ in range(num_lines):
                # Random frequency line
                if torch.rand(1) < 0.7:  # 70% frequency lines, 30% time lines
                    freq_start = torch.randint(0, freq_dim, (1,)).item()
                    line_width = torch.randint(1, max(2, freq_dim // 20), (1,)).item()
                    freq_end = min(freq_start + line_width, freq_dim)
                    spec[:, freq_start:freq_end, :] = 0
                else:
                    # Random time line
                    time_start = torch.randint(0, time_dim, (1,)).item()
                    line_width = torch.randint(1, max(2, time_dim // 20), (1,)).item()
                    time_end = min(time_start + line_width, time_dim)
                    spec[:, :, time_start:time_end] = 0
        return spec
    
    def add_gaussian_noise(self, spec):
        """Add gaussian noise to spectrogram"""
        if torch.rand(1) < self.noise_prob:
            noise = torch.randn_like(spec) * self.noise_std
            spec = spec + noise
            # Clamp to maintain reasonable range
            spec = torch.clamp(spec, 0, spec.max())
        return spec
    
    def add_buffer_simulation(self, spec):
        """Simulate lower sampling rate by adding buffer/padding"""
        if torch.rand(1) < self.buffer_prob:
            _, freq_dim, time_dim = spec.shape
            # Simulate lower resolution by downsampling and upsampling
            downsample_factor = torch.rand(1) * self.buffer_max_ratio + 0.9  # 0.9-1.0
            
            new_time_dim = int(time_dim * downsample_factor)
            new_freq_dim = int(freq_dim * downsample_factor)
            
            # Downsample
            spec_down = F.interpolate(spec.unsqueeze(0), 
                                    size=(new_freq_dim, new_time_dim), 
                                    mode='bilinear', align_corners=False).squeeze(0)
            
            # Upsample back to original size
            spec = F.interpolate(spec_down.unsqueeze(0), 
                               size=(freq_dim, time_dim), 
                               mode='bilinear', align_corners=False).squeeze(0)
        return spec
    
    def __call__(self, spec, is_training=True):
        """Apply up to 2 random augmentations during training only"""
        if not is_training:
            return spec
        
        # List of augmentation methods (as bound methods)
        augmentations = [
            self.horizontal_shift,
            self.add_occlusions,
            self.add_gaussian_noise,
            self.add_buffer_simulation
        ]
        # Randomly select up to 2 augmentations (could be 0, 1, or 2)
        num_to_apply = torch.randint(1, 3, (1,)).item()  # 1 or 2
        selected = random.sample(augmentations, num_to_apply)
        random.shuffle(selected)
        for aug in selected:
            spec = aug(spec)
        return spec

# Set up logging
if pl.utilities.rank_zero.rank_zero_only.rank == 0:
    log_file = f'Logs/model_training_{datetime.datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d_%H-%M-%S")}.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

# Decorator to ensure logging is only performed by the main process
@rank_zero_only
def log_message(message):
    logging.info(message)

# Custom Dataset class to load spectrogram images
class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=False, species=None, locations=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.species = species if species is not None else [''] * len(image_paths)
        self.locations = locations if locations is not None else [''] * len(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the spectrogram tensor from .pt file
        spec = torch.load(self.image_paths[idx])  # shape: [freq, time] or [1, freq, time]
        if spec.ndim == 2:
            # Add channel dimension if missing
            spec = spec.unsqueeze(0)  # [1, freq, time]
        # Repeat to 3 channels if needed for ResNet
        if spec.shape[0] == 1:
            spec = spec.repeat(3, 1, 1)  # [3, freq, time]
        
        # Resize to (3, 224, 224) using torch.nn.functional.interpolate
        if spec.shape[-2:] != (224, 224):
            spec = F.interpolate(spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # Apply tensor-based augmentations if training
        if self.transform is not None:
            spec = self.transform(spec, self.is_training)
        
        # Ensure tensor is float and normalized to [0, 1] if needed
        if spec.dtype != torch.float32:
            spec = spec.float()
        
        # Normalize if needed (assuming spectrograms might need normalization)
        if spec.max() > 1.0:
            spec = spec / spec.max()
        
        label = self.labels[idx]
        species = self.species[idx]
        location = self.locations[idx]
        return spec, label, self.image_paths[idx], species, location
# Define image transformations
# For .pt tensor data, we handle transformations manually in __getitem__
# These transforms are kept for reference but not used with tensor data

train_transform = SpectrogramAugmentations(
    horizontal_shift_range=0.1,
    occlusion_prob=0.1,
    occlusion_max_lines=3,
    noise_prob=0.1,
    noise_std=0.01,
    buffer_prob=0.1,
    buffer_max_ratio=0.1
)

val_transform = SpectrogramAugmentations()  # No augmentation during validation

test_transform = val_transform

##Model definition and output specs 
class ResNet18Classifier(pl.LightningModule):
    def __init__(self, image_paths, learning_rate=0.001, test_location=None, val_location=None, class_weights=None):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)  # 4 output classes
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.test_location = test_location
        self.val_location = val_location
        self.val_outputs = []
        self.test_outputs = []
        self.image_paths = image_paths
        # Create the directory if it doesn't exist
        os.makedirs('/home/radodhia/ssdprivate/NOAAWhalesV2/TestResults', exist_ok=True)
        self.csv_file_path = f'/home/radodhia/ssdprivate/NOAAWhalesV2/TestResults/test_results_testlocation_{self.test_location}_vallocation_{self.val_location}.csv'        
        # Write the CSV header
        with open(self.csv_file_path, 'w', newline='') as file:
            fieldnames = ['image_filename', 'predicted_label', 'actual_label', 'species', 'location']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, *_ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.eq(preds, labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, *_ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        self.val_outputs.append({'preds': preds.detach(), 'labels': labels.detach()})

        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels, image_paths, species, locations = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.test_outputs.append({'preds': preds, 'labels': labels})
        
        # Collect data for saving to CSV
        results = []
        for i in range(len(labels)):
            results.append({
                'image_filename': os.path.basename(image_paths[i]),
                'predicted_label': preds[i].item(),
                'actual_label': labels[i].item(),
                'species': species[i] if species[i] is not None else '',
                'location': locations[i] if locations[i] is not None else ''
            })
        
        # Append the results to the CSV file
        with open(self.csv_file_path, 'a', newline='') as file:
            fieldnames = ['image_filename', 'predicted_label', 'actual_label', 'species', 'location']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(results)
        
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True, batch_size=inputs.size(0))
        return {'test_loss': loss, 'test_acc': acc}

    def on_train_epoch_start(self):
        logging.info(f'Starting epoch {self.current_epoch + 1}')
        # Log learning rate(s)
        if hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
            for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
                logging.info(f'Learning rate for group {i}: {param_group["lr"]}')
        # Log GPU memory usage if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem = torch.cuda.memory_allocated(i) / 1024**3
                logging.info(f'GPU {i} memory allocated: {mem:.2f} GB')

    def on_train_epoch_end(self):
        logging.info(f'Finished epoch {self.current_epoch + 1}')

    def on_validation_epoch_start(self):
        self.val_outputs = []
        logging.info(f'Starting validation epoch {self.current_epoch + 1}')

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.val_outputs])
        all_labels = torch.cat([x['labels'] for x in self.val_outputs])

        # Convert to CPU tensors
        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Compute metrics
        precision = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        recall = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(all_labels_np, all_preds_np, multi_class='ovo')  # multiclass support
        except ValueError:
            auc = float('nan')  # fallback if only one class present

        # Compute val_loss and val_acc from logged values
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_acc = self.trainer.callback_metrics.get('val_acc')

        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds))
        self.log('val_recall', recall, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds))
        self.log('val_auc', auc, prog_bar=True, logger=True, sync_dist=True, batch_size=len(all_preds))

        # Log class distribution in validation set
        unique, counts = np.unique(all_labels_np, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logging.info(f'Validation class distribution: {class_dist}')

        if val_loss is not None:
            val_loss_value = val_loss.item()
        else:
            val_loss_value = float('nan')
        logging.info(f'Validation Metrics - Epoch {self.current_epoch+1}: Loss={val_loss_value:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc:.4f}')


    def on_test_epoch_start(self):
        self.test_outputs = []
        logging.info(f'Starting test epoch')

    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_outputs])

        precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        try:
            auc = roc_auc_score(all_labels.cpu(), all_preds.cpu(), multi_class='ovo')
        except ValueError:
            auc = float('nan')
        test_acc = (all_preds == all_labels).float().mean().item()

        # Log class distribution in test set
        unique, counts = np.unique(all_labels.cpu().numpy(), return_counts=True)
        class_dist = dict(zip(unique, counts))
        logging.info(f'Test class distribution: {class_dist}')

        self.log('test_precision', round(precision, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_recall', round(recall, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_auc', round(auc, 3), sync_dist=True, batch_size=len(all_preds))
        self.log('test_accuracy', round(test_acc, 3), sync_dist=True, batch_size=len(all_preds))
        logging.info(f'Test Metrics - Location: {self.test_location}, Val Location: {self.val_location}, Acc: {test_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, AUC: {auc:.3f}')
        print(f'Test Metrics - Location: {self.test_location}, Val Location: {self.val_location}, Acc: {test_acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, AUC: {auc:.3f}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# After ModelCheckpoint callback definition, add a custom callback for logging best model saves and early stopping
class LoggingCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.info(f'Checkpoint saved at epoch {trainer.current_epoch}')
    def on_train_end(self, trainer, pl_module):
        if hasattr(trainer, 'early_stopping_callback') and trainer.early_stopping_callback.stopped_epoch > 0:
            logging.info(f"Early stopping triggered at epoch {trainer.early_stopping_callback.stopped_epoch}")

# Main loop for training and validation

# Set model train parameters
data_use_proportion = 1.0
num_epochs = 15
batch_size = 64


# Ensure CUDA is available and devices are properly initialized
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

# Labels file path
label_files = ['/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/Beluga/LabelsOverlap400ms/Beluga_labels.csv'
               ,'/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/Humpback/LabelsOverlap400ms/Humpback_labels.csv'
               ,'/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/Orca/LabelsOverlap400ms/Orca_labels.csv']

dfs = []
for label_file in label_files:
    df = pd.read_csv(label_file)
    dfs.append(df)
labelsdf = pd.concat(dfs)

# Assign integer labels: 0=nothing, 1=humpback, 2=orca, 3=beluga
labelsdf['multilabel'] = 0  # default to nothing
labelsdf.loc[(labelsdf['species'] == 'humpback') & (labelsdf['label'] != 0), 'multilabel'] = 1
labelsdf.loc[(labelsdf['species'] == 'orca') & (labelsdf['label'] != 0), 'multilabel'] = 2
labelsdf.loc[(labelsdf['species'] == 'beluga') & (labelsdf['label'] != 0), 'multilabel'] = 3

logging.info(f'Label files read')
# Log label counts by species
label_counts = labelsdf.groupby('species')['label'].value_counts()
logging.info(f'Label counts by species:\n{label_counts}')
labelsdf.to_csv('/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/labels_overlap400ms_three_species.csv')

# Get unique locations from the CSV file
locations = labelsdf['location'].unique()

# runs=[{'test':'AL16_BS4','val':'Iniskin'},{'test':'Iniskin','val':'AL16_NM1'},{'test':'Chinitna','val':'PtGraham'}]
runs=[{'val':['AL16_NM1','213D','Iniskin'],'test':['PtGrahama','215D','SWCorner']}]

for run in runs:
    test_locations = run['test']
    val_locations = run['val']
    logging.info(f'\n\n')
    logging.info(f'Setting test locations to {test_locations}')
    logging.info(f'Setting validation locations to {val_locations}')

    logging.info('Getting test data')
    test_image_paths, test_labels, test_species, test_locations_list = [], [], [], []
    for test_location in test_locations:
        logging.info('Getting testing data')
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=test_location, data_use_proportion=data_use_proportion)
        test_image_paths.extend(img_paths)
        test_labels.extend(lbls)
        # For each image, lookup species/location in df_loc
        df_loc = labelsdf[labelsdf['location'] == test_location]
        df_loc_indexed = df_loc.set_index('fullpath')
        for img_path in img_paths:
            if img_path in df_loc_indexed.index:
                test_species.append(df_loc_indexed.loc[img_path, 'species'])
                test_locations_list.append(df_loc_indexed.loc[img_path, 'location'])
            else:
                test_species.append(None)
                test_locations_list.append(None)

    train_locations = [loc for loc in locations if loc not in test_locations and loc not in val_locations]
    logging.info(f'Setting train locations to {train_locations}')
    train_image_paths, train_labels = [], []

    for train_location in train_locations:
        logging.info('Getting training data')
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=train_location, data_use_proportion=data_use_proportion)
        train_image_paths.extend(img_paths)
        train_labels.extend(lbls)

    val_image_paths, val_labels = [], []
    for val_location in val_locations:
        logging.info('Getting validation data')
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=val_location, data_use_proportion=data_use_proportion)
        logging.info(f'Adding {len(img_paths)} validation samples from {val_location}')
        val_image_paths.extend(img_paths)
        val_labels.extend(lbls)

    # Proportional data split for validation
    p = 0.2  # Proportion for validation
    all_train_idx, val_idx_from_train = train_test_split(range(len(train_image_paths)), test_size=p, stratify=train_labels)
    logging.info(f'Split {len(train_image_paths)} training samples into {len(all_train_idx)} for training and {len(val_idx_from_train)} for validation')

    # Prepare species and locations for train and val using efficient dictionary lookup
    logging.info('Creating lookup dictionaries for species and locations')
    path_to_species = dict(zip(labelsdf['fullpath'], labelsdf['species']))
    path_to_location = dict(zip(labelsdf['fullpath'], labelsdf['location']))
    
    train_species = [path_to_species.get(p, '') for p in train_image_paths]
    logging.info(f'Created train_species list with {len(train_species)} entries')
    train_locations_list = [path_to_location.get(p, '') for p in train_image_paths]
    logging.info(f'Created train_locations_list with {len(train_locations_list)} entries')
    val_species = [path_to_species.get(p, '') for p in val_image_paths]
    logging.info(f'Finished creating validation species list with {len(val_species)} entries')
    val_locations_list = [path_to_location.get(p, '') for p in val_image_paths]
    logging.info(f'Finished creating validation locations list with {len(val_locations_list)} entries')

    train_subset = Subset(SpectrogramDataset(train_image_paths, train_labels, transform=train_transform, species=train_species, locations=train_locations_list), all_train_idx)
    val_subset_from_train = Subset(SpectrogramDataset(train_image_paths, train_labels, transform=val_transform, species=train_species, locations=train_locations_list), val_idx_from_train)
    val_dataset_from_val_location = SpectrogramDataset(val_image_paths, val_labels, transform=val_transform, species=val_species, locations=val_locations_list)
    logging.info(f'Created training and validation subsets')

    # Combine the val subsets
    val_dataset = ConcatDataset([val_subset_from_train, val_dataset_from_val_location])
    logging.info(f'Created validation dataset with {len(val_dataset)} samples from train and validation locations')
    

    '''
    Create dataframes containing file paths, whether they are train, validation, or test, and their labels
    '''
    # Create a DataFrame of training image paths, labels, species, and locations
    train_image_paths = [train_subset.dataset.image_paths[i] for i in all_train_idx]
    train_labels = [train_subset.dataset.labels[i] for i in all_train_idx]
    train_species = [train_subset.dataset.species[i] for i in all_train_idx]
    train_locations = [train_subset.dataset.locations[i] for i in all_train_idx]
    train_df = pd.DataFrame({
        'image_path': [os.path.basename(p) for p in train_image_paths],
        'assigned': 'train',
        'multilabel': train_labels,
        'species': train_species,
        'location': train_locations
    })

    # Create a DataFrame of all validation image paths, labels, species, and locations
    val_sample_image_paths = [val_subset_from_train.dataset.image_paths[i] for i in val_idx_from_train]
    val_sample_labels = [val_subset_from_train.dataset.labels[i] for i in val_idx_from_train]
    val_sample_species = [val_subset_from_train.dataset.species[i] for i in val_idx_from_train]
    val_sample_locations = [val_subset_from_train.dataset.locations[i] for i in val_idx_from_train]
    # Add val_dataset_from_val_location
    val_sample_image_paths += [i for i in val_dataset_from_val_location.image_paths]
    val_sample_labels += val_dataset_from_val_location.labels
    val_sample_species += [s for s in getattr(val_dataset_from_val_location, 'species', ['']*len(val_dataset_from_val_location.image_paths))]
    val_sample_locations += [l for l in getattr(val_dataset_from_val_location, 'locations', ['']*len(val_dataset_from_val_location.image_paths))]
    val_df = pd.DataFrame({
        'image_path': [os.path.basename(p) for p in val_sample_image_paths],
        'assigned': 'validation',
        'multilabel': val_sample_labels,
        'species': val_sample_species,
        'location': val_sample_locations
    })

    # Create a DataFrame of test image paths, labels, species, and locations
    test_df = pd.DataFrame({
        'image_path': [os.path.basename(i) for i in test_image_paths],
        'assigned': 'test',
        'multilabel': test_labels,
        'species': test_species,
        'location': test_locations_list
    })
    
    # Combine train_df, val_df, and test_df and save as csv
    assigned_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    os.makedirs("Assignations", exist_ok=True)
    # Convert list locations to string for filenames
    test_location = '_'.join(test_locations)
    val_location = '_'.join(val_locations)
    assigned_filepath = os.path.join("Assignations", f"Run_test_{test_location}_val_{val_location}.csv")
    assigned_df.to_csv(assigned_filepath, index=False)
    logging.info(f'Saved data assignations to {assigned_filepath}')

    # Log train/val/test class, species, and location counts at the start of training
    def log_split_stats(split_name, df):
        logging.info(f"{split_name} split: {len(df)} samples")
        if 'species' in df.columns:
            species_counts = df['species'].value_counts()
            logging.info(f"{split_name} species counts: {species_counts.to_dict()}")
        if 'location' in df.columns:
            location_counts = df['location'].value_counts()
            logging.info(f"{split_name} location counts: {location_counts.to_dict()}")
        if 'multilabel' in df.columns:
            label_counts = df['multilabel'].value_counts()
            logging.info(f"{split_name} label counts: {label_counts.to_dict()}")

    log_split_stats('Train', train_df)
    log_split_stats('Validation', val_df)
    log_split_stats('Test', test_df)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(SpectrogramDataset(test_image_paths, test_labels, transform=test_transform, is_training=False, species=test_species, locations=test_locations_list), batch_size=batch_size, shuffle=False, num_workers=0)
    logging.info(f'Data loaders created')

    # Compute class weights from training labels
    from collections import Counter
    train_label_counts = Counter(train_labels)
    num_classes = 4
    total = sum(train_label_counts.values())
    class_weights = []
    for i in range(num_classes):
        count = train_label_counts.get(i, 0)
        if count > 0:
            class_weights.append(total / (num_classes * count))
        else:
            class_weights.append(0.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logging.info(f'Class weights for loss: {class_weights}')

    model = ResNet18Classifier(image_paths=test_image_paths, class_weights=class_weights_tensor)
    logging.info(f'Model set to ResNet18Classifier()')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='BestModels',
        filename=f'best_model_test_{test_location}_val_{val_location}',
        save_top_k=1,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
    mode='min'
)
    
    log_dir = os.path.join(
        "lightning_logs",
        datetime.datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M-%S')
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name='')
    # Log test/val locations as hyperparameters
    tb_logger.log_hyperparams({'test_locations': test_locations, 'val_locations': val_locations})

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices='auto',  # Use all available GPUs
        num_nodes=1,
        accelerator='gpu',
        strategy='ddp',  # Use DistributedDataParallel for multi-GPU
        precision='16-mixed',
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, LoggingCallback()]
    )
    
    logging.info(f'Starting model training')    
    trainer.fit(model, train_loader, val_loader)

    # Test the best model
    best_model_path = checkpoint_callback.best_model_path
    # best_model_path = './BestModels/best_model_test_Iniskin_val_ALNM01-v1.ckpt'
    best_model = ResNet18Classifier.load_from_checkpoint(best_model_path, learning_rate=0.001, test_location=test_location, val_location=val_location, image_paths=test_image_paths)
    logging.info(f'best model loaded from {best_model_path}')

    # Perform the test
    trainer.test(model=best_model, dataloaders=test_loader)


'''
    # Read the test results from the CSV file
    test_results = pd.read_csv(csv_file_path)

    # Extract the predicted and actual labels
    predicted_labels = test_results['predicted_label']
    actual_labels = test_results['actual_label']

    # Calculate recall, precision, and auc
    recall = recall_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels)
    auc = roc_auc_score(actual_labels, predicted_labels)

    # Print the metrics
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"AUC: {auc:.4f}")
'''