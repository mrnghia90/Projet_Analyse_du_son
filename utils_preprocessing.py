import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Librairies for file openings
import os
import pickle
import glob

# Audio Librairies
import librosa
import librosa.display
import IPython.display as idp

#Librairies for better plotting
from itertools import cycle
sns.set_theme(style="white", palette=None)
#color palette
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


class Loader:
    
    """ Loader purpose: load audio (.wav) to Array-like """
    
    def __init__ (self, sample_rate: int, duration: int):
        
        self.sample_rate=sample_rate # Number of sample for 1 second audio
        self.duration=duration # Duration of the audio
        
    def load(self,file_path):
        signal,_=librosa.load(file_path,
                           sr=self.sample_rate,
                           duration=self.duration)
        return signal

class Padder:
    
    """ Padder purpose: Padding Array-like with different methods """
    def __init__ (self, mode="constant"):
        self.mode=mode
        
    def left_padding(self,array: list, num_missing_items: int) -> list:
        padded_array=np.pad(array, 
                           (num_missing_items,0),
                            mode=self.mode)
        return padded_array
    
    def right_padding(self,array:list, num_missing_items: int) -> list:
        padded_array=np.pad(array,
                           (0,num_missing_items),
                           mode=self.mode)
        return padded_array
    
    def equal_padding(self,array:list, num_missing_items: int) -> list:
        
        if num_missing_items%2==1: # if odd number then padded_array can miss 1 value
            before= int(num_missing_items/2) + 1
            after= int(num_missing_items/2)
        padded_array=np.pad(array,
                           (before,after),
                           mode=self.mode)
        return padded_array


        
class LogSpectrogramExtractor:
    
    """LogSpectrogramExtractor purpose: Extract Array-like audio files to Spectrogram data"""
    
    def __init__ (self, frame_size: int, hop_length: int):
        
        self.frame_size=frame_size
        self.hop_length=hop_length
    
    def extract_stft (self, signal):
        stft=librosa.stft(signal,
                          n_fft=self.frame_size,
                          hop_length=self.hop_length) #[:-1]  #(1 +frame_size/2,num_frames)
                                                 #1024->513->512
        spectrogram=np.abs(stft)
        log_spectrogram=librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
    
class MelSpectrogramExtractor:
    
    """MelSpectrogramExtractor purpose: Extract Array-like audio files to MelSpectrogram data"""

    
    def __init__(self, hop_length, n_mels, sample_rate=22050):
        
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.sample_rate=sample_rate
    
    
    def extract_mel (self, signal):
        
        Mel_spectrogram= librosa.feature.melSpectrogram(y=signal,
                                                       sr=self.sample_rate,
                                                       n_mels=self.n_mels,
                                                       hop_length=self.hop_length)
        Mel_spectrogram=librosa.amplitude_to_db(Mel_spectrogram)
        return Mel_spectrogram


class MinMaxScaler:
    
    """"""
    def __init__(self, min_val, max_val):
        self.min_val=min_val
        self.max_val=max_val
    
    def normalize(self, array):
        array_scaled= (array-array.min())/(array.max()-array.min())
        array_scaled= array_scaled*(self.max_val - self.min_val) + self.min_val
        
        return array_scaled
    
    def denormalize(self, array_scaled, original_min, original_max):
        array= (array_scaled - self.min_val) / (self.max_val - self.min_val)
        array= array * (original_max -original_min) + original_min
        
    
    
    
class MaxSample:
    
    """ PEUT ETRE ENLEVER (A LAISSER POUR L'INSTANT)"""
    def __init__(self, initialize_sample=50000, max_sample=0):
        
        self.max_sample=max_sample
        self.initialize_sample=initialize_sample
        
    def maxSample(self,signal):
        
        num_sample = signal.shape[0]
        
        if self.max_sample<num_sample:
            self.max_sample=num_sample
    

class Saver:
    """saver is responsible to save features, and the min max values."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path
        


class PreprocessingPipeline:
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self):
        
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None
        

    @property
    def loader(self):
        
        return self._loader
    
    

    @loader.setter
    def loader(self, loader):
        
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
        

    def process(self, audio_files_dir):
        
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                if file.endswith('.wav'):  # Adjust the prefix as needed
                    file_path = os.path.join(root, file)
                    self._process_file(file_path)
                    print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)
        

    def _process_file(self, file_path):
        
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract_stft(signal) # CHANGE CLASS SO YOU CAN EXTRACT SFTF AND MELSPEC !!!!
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
        

    def _is_padding_necessary(self, signal):
        
        if len(signal) < self._num_expected_samples:
            return True
        return False
    

    def _apply_padding(self, signal):
        
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_padding(signal, num_missing_samples)
        return padded_signal
    

    def _store_min_max_value(self, save_path, min_val, max_val):
        
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }