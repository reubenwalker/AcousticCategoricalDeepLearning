#PSEUDOCODE
#Load DataFrame
#Run through Code Sandbox for all Test5 files
#Calculate:
    #Mel Spectrogram
    #MFCCs
    #Dynamic Time Warped Versions to Reference
    #Pad/Truncate to Match Reference
#Save 

#Necessary Libraries:
from parselmouth.praat import call
from parselmouth import Sound 
import scipy
from scipy import stats
import librosa
from scipy.signal import hilbert, butter, sosfilt, find_peaks, resample, find_peaks_cwt
from scipy.io.wavfile import read, write
import statsmodels.api as sm 
from statsmodels.tsa.stattools import acovf, acf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pickle
from scipy import signal
from librosa.feature import rms
import os
import glob
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import MaxAbsScaler
import noisereduce as nr 
import torch
import math
import random
from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from matplotlib.lines import Line2D
import speechpy
###Normalization based on messa di voce
import librosa
import pyloudnorm as pyln
from librosa.sequence import dtw
import random
from datetime import datetime

import torchaudio  # For SpecAugment masking
import torch
import numpy as np
import librosa
from librosa.sequence import dtw
# from fastdtw import fastdtw  # or use dtw f

def freqExtract(data, samplerate, freqLow=60, freqHigh=700):
    sound = Sound(data, samplerate)
    pitch = call(sound, "To Pitch", 0.1, freqLow, freqHigh) #c4-g5 
    pitch_contour = pitch.selected_array['frequency']
    if pitch_contour[pitch_contour != 0].size == 0:
        return np.nan, np.nan, np.nan, np.nan
    pitchContLength = pitch.selected_array['frequency'].size
    wavLength = len(data)
    f_s_contour = pitchContLength/wavLength*samplerate
    ###Pitch contour correction
        #From https://praat-users.yahoogroups.co.narkive.com/swLrgWcR/eliminate-octave-jumps-and-annotate-pitch-contour
    q1 = call(pitch, "Get quantile", 0.0, 0.0, 0.25, "Hertz")
    q3 = call(pitch, "Get quantile", 0.0, 0.0, 0.75, "Hertz")
    floor = q1*0.75
    ceiling = q3*2
    pitchCorrect = call(sound, "To Pitch", 0.1, floor, ceiling) #c4-g5 
    contourCorrect = pitchCorrect.selected_array['frequency']
    #plt.plot(pitch_contour)
    #plt.plot(contourCorrect)
    contourCorrect[contourCorrect == 0] = np.nan
    neighbor = contourCorrect[1:]/contourCorrect[:-1]
    idx = neighbor > 1.5
    mask = np.where(idx)
    #Eliminate octave jumps?
    # contourCorrect[1:][mask] = contourCorrect[1:][mask]/2
    #plt.plot(contourCorrect)
    f_0_min = np.nanmin(contourCorrect)
    f_0_max = np.nanmax(contourCorrect)
    f_0_mean = np.nanmean(contourCorrect)
    f_0_median = np.nanmedian(contourCorrect)
    return f_0_min, f_0_max, f_0_median, f_0_mean
    
# --- Compute Mel Spectrogram ---
def get_mel(y, sr, n_mels=128, fmin=1000, fmax=5000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
    
def pad_or_truncate_mfcc(mfcc, target_len):
    """Pad or truncate an MFCC matrix to match target_len time steps."""
    current_len, n_mfcc = mfcc.shape
    if current_len >= target_len:
        return mfcc[:target_len, :]
    else:
        pad_width = target_len - current_len
        return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')

# --- Pad or truncate ---
def pad_or_truncate(mel, target_frames):
    if mel.shape[1] < target_frames:
        pad_width = target_frames - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel = mel[:, :target_frames]
    return mel

# --- Align input to reference using waveform-based MFCC DTW with smoothing and constraints ---
def align_mel_with_mfcc_smooth(ref_mfcc, ref_mel, input_y, sr, n_mfcc=20, hop_length_smooth=3, band_rad=20):
    # Compute Mel spectrogram of input for visualization/warping
    input_mel = get_mel(input_y, sr)

    # Compute MFCCs directly from waveform
    input_mfcc = librosa.feature.mfcc(y=input_y, sr=sr, n_mfcc=n_mfcc)

    # --- Smooth MFCCs over time to reduce jitter ---
    if hop_length_smooth > 1:
        input_mfcc = np.convolve(input_mfcc.flatten(), np.ones(hop_length_smooth)/hop_length_smooth, mode='same')
        input_mfcc = input_mfcc.reshape(n_mfcc, -1)

    # --- DTW on MFCCs with Sakoe-Chiba band ---
    D, wp = dtw(X=ref_mfcc, Y=input_mfcc, metric='euclidean', global_constraints=True, band_rad=band_rad)
    wp = np.array(wp[::-1])  # forward time

    # --- Warp original Mel spectrogram using linear interpolation ---
    n_mels = input_mel.shape[0]
    aligned_input_mel = np.zeros((n_mels, ref_mel.shape[1]))
    for i in range(ref_mel.shape[1]):
        idxs = wp[wp[:, 0] == i, 1]
        if len(idxs) > 0:
            # Linear interpolation between first and last mapped frame
            aligned_input_mel[:, i] = np.mean(input_mel[:, idxs], axis=1)

    # Pad or truncate to match reference length
    aligned_input_mel = pad_or_truncate(aligned_input_mel, ref_mel.shape[1])

    return input_mel, aligned_input_mel

# --- Compute best DTW path from pitch-shifted original input ---
def compute_best_dtw(ref_mel, input_y, sr, pitch_shifts=[0,2,4,12,14,16]):
    # ref_mel = get_mel(ref_y, sr)
    best_cost = np.inf
    best_wp = None
    best_shift = None
    best_input_mel = None
    
    for n_steps in pitch_shifts:
        shifted_y = librosa.effects.pitch_shift(input_y, n_steps=n_steps, sr=sr)
        shifted_mel = get_mel(shifted_y, sr)
        if n_steps == 0:
            input_mel_0 = shifted_mel
        
        D, wp = dtw(X=ref_mel, Y=shifted_mel, metric='euclidean')
        cost = D[-1, -1]
        
        if cost < best_cost:
            best_cost = cost
            best_wp = np.array(wp[::-1])  # forward time
            best_shift = n_steps
            best_input_mel = shifted_mel
    
    # Return saved path and reference info
    return best_wp #, ref_mel

def apply_saved_dtw(input_y, sr, wp_saved, ref_mel, n_steps_aug=0):
    # Random pitch-shift (augmentation)
    aug_y = librosa.effects.pitch_shift(input_y, n_steps=n_steps_aug, sr=sr)
    aug_mel = get_mel(aug_y, sr)
    
    # Warp augmented Mel using saved DTW path
    aligned_mel = np.zeros((aug_mel.shape[0], ref_mel.shape[1]))
    for i in range(ref_mel.shape[1]):
        idxs = wp_saved[wp_saved[:,0]==i, 1]
        if len(idxs) > 0:
            aligned_mel[:, i] = aug_mel[:, idxs].mean(axis=1)
    
    aligned_mel = pad_or_truncate(aligned_mel, ref_mel.shape[1])
    return aligned_mel, aug_mel



# --- Visualization ---
def plot_alignment(ref_mel, input_mel, aligned_input_mel, sr, fmin=1000, fmax=5000):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    librosa.display.specshow(ref_mel, x_axis='time', y_axis='mel', sr=sr, ax=axs[0], fmin=fmin, fmax=fmax)
    axs[0].set_title('Reference Mel Spectrogram')

    librosa.display.specshow(input_mel, x_axis='time', y_axis='mel', sr=sr, ax=axs[1], fmin=fmin, fmax=fmax)
    axs[1].set_title('Original Input Mel Spectrogram')

    librosa.display.specshow(aligned_input_mel, x_axis='time', y_axis='mel', sr=sr, ax=axs[2], fmin=fmin, fmax=fmax)
    axs[2].set_title('Waveform-MFCC DTW Aligned Input Mel Spectrogram')

    plt.tight_layout()
    # plt.savefig("waveform_mfcc_dtw.png")
    # plt.show()

#What are our frequency thresholds for avezzo?
#DDur / DisDur
d = 146.8324
dis = 155.5635
d1 = 293.6648
dis1 = 311.1270
d2 = 587.3295
dis2 = 622.2540

#For sustained fifth
e1 = 329.6276
e2 = 659.2551

#For vowel
g = 195.9977
g1 = 391.9954
a = 220.0000
a1 = 440

freqPairs = [[-np.inf,311],[311,350],[554,622],[622,np.inf]]
targetFreqs = [d1,e1,d2,e2]

medComp1 = np.array([g,a,g1,a1])

medComp2 = np.array([d1,e1,d2,e2])

path = os.getcwd()
wav_files = glob.glob(os.path.join(path, "*.wav*"))#"*.xlsx"))
mancaFiles = []
avezzoFiles = []
dreiklangFiles = []
vokalFiles = []
comeFiles = []
for i in wav_files:
    if i[-5] == '1':
        vokalFiles.append(i)
    if i[-5] == '2':
        dreiklangFiles.append(i)
    if i[-5] == '5':
        mancaFiles.append(i)
    if i[-5] == '6':
        avezzoFiles.append(i)
    if i[-5] == '7':
        comeFiles.append(i)
        
singingSamples =  mancaFiles #+ dreiklangFiles #+ #vokalFiles #+  + comeFiles vokalFiles +  avezzoFiles + 

root2 = np.sqrt(2)
root12 = np.power(2, 1/12)
prompt = ''

i = 0
df = pd.DataFrame({})
ref_path = '0011&2009_01_27&test5.wav' # sicher, Solo soprano with 0.06 s difference from median duration
ref_y, sr = librosa.load(ref_path, sr=16000)
ref_mel = get_mel(ref_y, sr)
# --- Extract MFCCs (transposed so time is axis 0) ---
ref_mfcc = librosa.feature.mfcc(y=ref_y, sr=sr, n_mfcc=13).T
ref_mfcc20 = librosa.feature.mfcc(y=ref_y, sr=sr, n_mfcc=20)

for wavFilename in singingSamples:

    #About 50% of the samples are 16 kHz samplerate, 50% 41kHz and a few 48 kHz
    #Let's downsample to 16 kHz.
    data, samplerate = librosa.load(wavFilename, mono=False, sr=16000)   
    sr = samplerate
    
    #Convert to floating point if int
    if data.dtype == 'int16': #Convert to float
        data = data.astype(np.float32) / np.iinfo(np.int16).max
    
    duration = len(data)/samplerate #seconds
    trialNum = wavFilename.split('\\')[-1][-5]
    idNum = int(wavFilename.split('\\')[-1].split('&')[0])
    date = wavFilename.split('\\')[-1].split('&')[1]
     
    if duration < 2:
        duration = np.nan
        resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      'date':wavFilename.split('\\')[-1].split('&')[1],
                      'trialNum':wavFilename.split('\\')[-1][-5],
                      'duration':duration
                      }
        df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
        continue

    if len(data.shape) > 1: #Convert to mono
        data = data[:,0]
    

    
    ###Now we shift the pitch randomly twice 
    choices = [-2,-1,1,2]
    augment = random.sample(choices,2) #Randomly raise/lower pitch
    augment.append(0) # Keep original
    wp_saved = compute_best_dtw(ref_mel, data, samplerate)
    for pitchChange in augment:
        augmented = True
        if pitchChange == 0:
            augmented = False
        
        ### Pitch Shift
        #Shift the data randomly between -2 to 2 halfsteps or retain the same
        # data_shift = librosa.effects.pitch_shift(data, sr=sr, n_steps=pitchChange)
  
        ## Dynamic Time Warp Audio to ReferenceError
        # --- Compute MFCCs or mel spectrograms in dB ---
        # input_mel = get_mel(data_shift, samplerate)
        
        # --- DTW alignment ---
        # input_mel, aligned_input_mel = align_mel_with_mfcc_smooth(ref_mfcc20, ref_mel, data, samplerate)

        aligned_input_mel, input_mel = apply_saved_dtw(data, samplerate, wp_saved, ref_mel, n_steps_aug=pitchChange)
        # plot_alignment(ref_mel, input_mel, aligned_input_mel, sr)

        
        # melSpec = melSpectrogram(data, samplerate)
        f_0_min, f_0_max, f_0_median, f_0_mean = freqExtract(data, samplerate)

        resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      'date':wavFilename.split('\\')[-1].split('&')[1],
                      'trialNum':wavFilename.split('\\')[-1][-5],
                      'duration':duration,
                      # 'pitchMin':f_0_min,
                      'pitchMed':f_0_median,
                      # 'mfcc_a':mfcc_a,
                      # 'mfcc_ei':mfcc_ei,
                      # 'pitchMax':f_0_max,
                      # 'pitchMean':f_0_mean,
                      # 'mfcc_ou':mfcc_ou,  
                      # 'meanMFCC_a':np.median(mfcc_a, axis=0),
                      # 'meanMFCC_ei':np.median(mfcc_ei, axis=0),
                      # 'meanMFCC_ou':np.median(mfcc_ou, axis=0),
                      'melSpec':aligned_input_mel,
                      # 'freqBand':freqBand,
                      # 'meanSpec':meanSpec,
                      # 'mfcc':final_mfcc,
                      'augmented':augmented,
                      'pitchShift':pitchChange,

                      'sr':samplerate
                     
                     }
                     
        df = pd.concat([df, pd.DataFrame.from_records([resultDict])])

        plt.close('all')
    # if prompt != 'done':
        # prompt = input("Press Enter to continue, q to quit...")
        # if prompt == 'q':
            # break

df.to_pickle('melSpecAug20250827.pkl')
