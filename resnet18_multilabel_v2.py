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

# Custom tensor-based augmentations for spectrograms
class SpectrogramAugmentations:
    def __init__(self, 
                 horizontal_shift_range=0.1,
                 occlusion_prob=0.3,
                 occlusion_max_lines=3,
                 noise_prob=0.5,
                 noise_std=0.01,
                 buffer_prob=0.3,
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
        if torch.rand(1) < 0.5:  # 50% chance
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
    def __init__(self, image_paths, labels, transform=None, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths)

    # def __getitem__(self, idx):
    #     image = Image.open(self.image_paths[idx])
    #     label = self.labels[idx]
    #     if self.transform:
    #         image = self.transform(image)
    #     return image, label, self.image_paths[idx] 

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
        return spec, label, self.image_paths[idx]
# Define image transformations
# For .pt tensor data, we handle transformations manually in __getitem__
# These transforms are kept for reference but not used with tensor data

train_transform = SpectrogramAugmentations(
    horizontal_shift_range=0.1,
    occlusion_prob=0.5,
    occlusion_max_lines=3,
    noise_prob=0.5,
    noise_std=0.01,
    buffer_prob=0.3,
    buffer_max_ratio=0.1
)

val_transform = SpectrogramAugmentations()  # No augmentation during validation

test_transform = val_transform


class ResNet18Classifier(pl.LightningModule):
    def __init__(self, image_paths, learning_rate=0.001, test_location=None, val_location=None):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.test_location = test_location
        self.val_location = val_location
        self.val_outputs = []
        self.test_outputs = []
        self.image_paths = image_paths
        # Create the directory if it doesn't exist
        os.makedirs('/home/radodhia/ssdprivate/NOAA_Whales/TestResults', exist_ok=True)
        self.csv_file_path = f'/home/radodhia/ssdprivate/NOAA_Whales/TestResults/test_results_testlocation_{self.test_location}_vallocation_{self.val_location}.csv'        
        # Write the CSV header
        with open(self.csv_file_path, 'w', newline='') as file:
            fieldnames = ['image_filename', 'predicted_label', 'actual_label']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.eq(preds, labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        
        self.val_outputs.append({'preds': preds.detach(), 'labels': labels.detach()})

        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels, image_paths = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.test_outputs.append({'preds': preds, 'labels': labels})
        
        # Collect data for saving to CSV
        results = []
        for i in range(len(labels)):
            # The image paths should be passed as part of the batch
            results.append({
                'image_filename': os.path.basename(image_paths[i]),
                'predicted_label': preds[i].item(),
                'actual_label': labels[i].item()
            })
        
        # Append the results to the CSV file
        with open(self.csv_file_path, 'a', newline='') as file:
            fieldnames = ['image_filename', 'predicted_label', 'actual_label']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(results)
        
        self.log('test_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, logger=True, sync_dist=True)
        return {'test_loss': loss, 'test_acc': acc}

    def on_train_epoch_start(self):
        logging.info(f'Starting epoch {self.current_epoch + 1}')

    def on_train_epoch_end(self):
        logging.info(f'Finished epoch {self.current_epoch + 1}')
    
    def on_validation_epoch_start(self):
        self.val_outputs = []

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

        self.log('val_precision', precision, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', recall, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_auc', auc, prog_bar=True, logger=True, sync_dist=True)

        logging.info(f'Validation Metrics - Epoch {self.current_epoch+1}: Precision={precision:.4f}, Recall={recall:.4f}, AUC={auc:.4f}')

    def on_test_epoch_start(self):
        self.test_outputs = []

    def on_test_epoch_end(self):
        all_preds = torch.cat([x['preds'] for x in self.test_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_outputs])

        precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='macro', zero_division=0)
        auc = roc_auc_score(all_labels.cpu(), all_preds.cpu())
        test_acc = (all_preds == all_labels).float().mean().item()

        self.log('test_precision', precision, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_auc', auc, sync_dist=True)
        self.log('test_accuracy', test_acc, sync_dist=True)
        logging.info(f'Test Metrics - Location: {self.test_location}, Val Location: {self.val_location}, Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')
        print(f'Test Metrics - Location: {self.test_location}, Val Location: {self.val_location}, Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Function to get data subset from labels df
def get_data_from_csv(df, location, data_use_proportion=1.0):
    logging.info(f'Reading {data_use_proportion} of location {location} from labels dataframe')
    df = df.sample(frac=data_use_proportion, random_state=42)
    df_location = df[df['location'] == location]
    image_paths = df_location['fullpath'].tolist()
    labels = df_location['label'].tolist()
    logging.info(f'Returning {len(image_paths)} image_paths and labels\n')
    return image_paths, labels


# Main loop for training and validation

# Set model train parameters
data_use_proportion = .02
num_epochs = 2
batch_size = 32


# Ensure CUDA is available and devices are properly initialized
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
# else:
#     for gpu in range(len(gpus)):
#         try:
#             _ = torch.cuda.get_device_properties(gpu)
#         except RuntimeError as e:
#             raise RuntimeError(f"Error initializing GPU {gpu}: {e}")


# Labels file path
label_files = ['/home/radodhia/ssdprivate/NOAAWhalesV2/DataInput/Beluga/201d_beluga_overlap400ms_spectrogram_labels.csv']
# Read label_files into a dataframe
# beluga labels were created by make_spectrograms_v3_beluga.py and belugaInputSelection.py

dfs = []
for label_file in label_files:
    df = pd.read_csv(label_file)
    dfs.append(df)
labelsdf = pd.concat(dfs)



# Add column 'species' and Remap labels in df
labelsdf.loc[(labelsdf['dirpath'].str.contains('Humpback')), 'species'] = 'humpback'
labelsdf.loc[(labelsdf['dirpath'].str.contains('KillerWhale')), 'species'] = 'killerwhale'
labelsdf.loc[(labelsdf['dirpath'].str.contains('Beluga')), 'species'] = 'beluga'

# Count labels by species
# label_counts = labelsdf.groupby('species')['label'].value_counts()
# print(label_counts)

labelsdf.loc[labelsdf['label']==0, 'multilabel'] = 'nothing'
labelsdf.loc[(labelsdf['species'] == 'humpback') & (labelsdf['label'] == 1), 'multilabel'] = 'humpback'
labelsdf.loc[(labelsdf['species'] == 'killerwhale') & (labelsdf['label'] == 1), 'multilabel'] = 'killerwhale'
labelsdf.loc[(labelsdf['species'] == 'beluga') & (labelsdf['label'] == 1), 'multilabel'] = 'beluga'

labelsdf.to_csv('/home/radodhia/ssdprivate/NOAA_Whales/DataInput/labels_overlap400ms_three_species.csv')

# Get unique locations from the CSV file
locations = labelsdf['location'].unique()
locations = locations[::-1]

# Get frequency count of location by species
labelsdf.groupby(['location', 'species']).size().reset_index(name='count')

# runs=[{'test':'ALBS04','val':'Iniskin'},{'test':'Iniskin','val':'ALNM01'},{'test':'Chinitna','val':'PtGraham'}]
# runs=[{'test':['ALBS04','201D'],'val':['Iniskin','215D']}]
runs=[{'test':['201D'],'val':['201D']}]

for run in runs:
    test_locations = run['test']
    val_locations = run['val']
    logging.info(f'\n\nSetting test locations to {test_locations}')
    logging.info(f'Setting validation locations to {val_locations}')

    logging.info('Getting test data')
    test_image_paths, test_labels = [], []
    for test_location in test_locations:
        logging.info('Getting testing data')
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=test_location, data_use_proportion=data_use_proportion)
        test_image_paths.extend(img_paths)
        test_labels.extend(lbls)

    train_locations = [loc for loc in locations if loc != test_locations and loc != val_locations]
    logging.info(f'Setting train locations to {train_locations}')
    train_image_paths, train_labels = [], []

    for train_location in train_locations:
        logging.info('Getting training data')
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=train_location, data_use_proportion=data_use_proportion)
        train_image_paths.extend(img_paths)
        train_labels.extend(lbls)

    val_image_paths, val_labels = [], []
    for val_location in val_locations:
        logging.info('Getting valing data')
        img_paths, lbls = get_data_from_csv(df=labelsdf, location=val_location, data_use_proportion=data_use_proportion)
        val_image_paths.extend(img_paths)
        val_labels.extend(lbls)

    # Proportional data split for validation
    p = 0.2  # Proportion for validation
    all_train_idx, val_idx_from_train = train_test_split(range(len(train_image_paths)), test_size=p, stratify=train_labels)

    train_subset = Subset(SpectrogramDataset(train_image_paths, train_labels, transform = train_transform), all_train_idx)
    val_subset_from_train = Subset(SpectrogramDataset(train_image_paths, train_labels, transform = val_transform), val_idx_from_train)
    val_dataset_from_val_location = SpectrogramDataset(val_image_paths, val_labels, transform = val_transform)

    # Combine the val subsets
    val_dataset = ConcatDataset([val_subset_from_train, val_dataset_from_val_location])

    '''
    Create dataframes containing file paths, whether they are train, validation, or test, and their labels
    '''
    # Create a DataFrame of training image paths and labels
    train_image_paths = [train_subset.dataset.image_paths[i] for i in all_train_idx]
    train_labels = [train_subset.dataset.labels[i] for i in all_train_idx]
    train_df = pd.DataFrame({'image_path': [os.path.basename(p) for p in train_image_paths], 'assigned':'train','label': train_labels})

    # Create a DataFrame of all validation image paths and labels
    val_image_paths = [val_subset_from_train.dataset.image_paths[i] for i in val_idx_from_train]
    val_labels = [val_subset_from_train.dataset.labels[i] for i in val_idx_from_train]
    val_image_paths += [i for i in val_dataset_from_val_location.image_paths]
    val_labels += val_dataset_from_val_location.labels
    val_df = pd.DataFrame({'image_path': [os.path.basename(p) for p in val_image_paths], 'assigned':'validation', 'label': val_labels})

    # Create a DataFrame of test image paths and labels
    test_df = pd.DataFrame({'image_path': [os.path.basename(i) for i in test_image_paths], 'assigned':'test', 'label': test_labels})
    
    # Combine train_df, val_df, and test_df and save as csv
    assigned_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    os.makedirs("DataAssignations", exist_ok=True)
    assigned_filepath = os.path.join("DataAssignations", f"Run_test_{test_location}_val_{val_location}.csv")
    assigned_df.to_csv(assigned_filepath, index=False)
    logging.info(f'Saved data assignations to {assigned_filepath}')

    train_loader = DataLoader(SpectrogramDataset(train_image_paths, train_labels, transform=train_transform, is_training=True), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(SpectrogramDataset(val_image_paths, val_labels, transform=val_transform, is_training=False), batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(SpectrogramDataset(test_image_paths, test_labels, transform=test_transform, is_training=False), batch_size=batch_size, shuffle=False, num_workers=4)
    logging.info(f'Data loaders created')

    model = ResNet18Classifier(image_paths=test_image_paths)
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
    patience=3,
    verbose=True,
    mode='min'
)
    
    log_dir = os.path.join("lightning_logs", f"test_{test_location}_val_{val_location}")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices=[0],
        num_nodes=1,
        accelerator='gpu',
        strategy='auto',
        precision='16-mixed',
        logger=pl_loggers.TensorBoardLogger(save_dir=log_dir,name=''),
        callbacks=[checkpoint_callback, early_stop_callback]
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