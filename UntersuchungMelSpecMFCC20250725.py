#PSEUDOCODE
#Load DataFrame
#Run through Code Sandbox for all Test2 files
#Calculate:
    #FINAL Trial 
        #Beginning Frame
        #End Frame
    #Maximum Frequency
    #Frames within a minor third of this frequency
        #Beginning Frame
        #End Frame
    #Middle 50% of sustained pitch
        #Beginning Frame
        #End Frame
    #Rolling Mean/Std Vibrato Frequency
    #Rolling Mean/Std Vibrato Amplitude
#Save all four measures

#Train K Nearest neighbors algorithm on stabil, labil, ohne
#Rerun the classifier on 90% of sustained pitch. Only record values if they return as "stable"


###Let's clean this up.

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
# from fastdtw import fastdtw  # or use dtw f

def get_nonzero_bounds(arr):
    arr = np.asarray(arr)
    nonzero_indices = np.where(arr != 0)[0]
    if nonzero_indices.size == 0:
        return None, None  # or raise an error if needed
    return nonzero_indices[0], nonzero_indices[-1]

###Normalization to quietest 
def get_quietest_edge_rms(audio, sr, segment_duration=0.1):
    segment_samples = int(segment_duration * sr)
    start_segment = audio[:segment_samples]
    end_segment = audio[-segment_samples:]

    start_rms = np.sqrt(np.mean(start_segment**2))
    end_rms = np.sqrt(np.mean(end_segment**2))

    return min(start_rms, end_rms)
    
def normalize_to_target_rms(audio, target_rms):
    audio_rms = np.sqrt(np.mean(audio**2))
    if audio_rms == 0:
        return audio
    return audio * (target_rms / audio_rms)

def reference_rms(wavdata, sr):
    sound = Sound(wavdata, sr)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #create a praat pitch object
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency


def loudness_normalize(audio, sr, target_lufs=-23.0):
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
    return normalized_audio, loudness



def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    # fig, ax = subplots(4)
    # ax[0].plot(norm)
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return r, lag

def tremorRateCalc(wavData, samplerate):
    analytic_signal = hilbert(wavData)
    amplitude_envelope = np.abs(analytic_signal)
    #resample takes number of samples. 
    #50 Hz would be number of samples at samplerate * 50/samplerate rounded to the nearest integer
    samples = round(len(amplitude_envelope)*50/samplerate)
    amplitude_env_downsampled = resample(amplitude_envelope, samples)
    # Local maxima of the envelopes were detected
    # using the peak-finding function (https://terpconnect.umd.edu
    # /~toh/spectrum/). 
    peaks = find_peaks(amplitude_env_downsampled, distance=round(50/10))[0]
    plt.close('all')
    plot(np.arange(len(amplitude_env_downsampled))/50,amplitude_env_downsampled)
    plot(peaks/50, amplitude_env_downsampled[peaks], 'x')
    # Peaks were assessed by downward zerocrossings in the smoothed first derivative using a pseudogaussian smoothing algorithm. A peak was classified as a group
    # of points with amplitude that exceeded the amplitude of
    # neighboring points on either side. The rate of tremor was
    # calculated by the total number of peaks divided by the duration
    # of the audio sample (Figure 1).
    sampleDuration = len(wavData)/samplerate
    tremorRate = len(peaks)/sampleDuration
    #Both samples show lower tremorRate than the Madde simulation.
    #Cut sample duration from first peak to last peak, 
        #then subtract the peaks by one.
    peakDuration = len(amplitude_env_downsampled[peaks[0]:peaks[-1]])/len(amplitude_env_downsampled)*sampleDuration
    tremorRate_final = len(peaks[:-1])/peakDuration
    return tremorRate_final

def vibratoCalc8(stableWavArray, samplerate):
#0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #create a praat pitch object
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(0.5*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    pandasContour = pd.Series(pitch_contour)
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3Hz(x, f_s_contour))
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.np.std()
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    return vibrato_Frequency, vibratoPercentage, vibratoStd
    
###Massive Calculation
def vibratoCalcMF(stableWavArray, samplerate, gender=np.nan, windowSecs=0.5):
#0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's go A-G
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(windowSecs*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    pandasContour = pd.Series(pitch_contour)
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3Hz(x, f_s_contour))
    rollingAmp = pandasContour.rolling(window).apply(lambda x: vibAmpRoll(x, f_s_contour,rollingVib, windowFactor=0.75))
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.np.std()
    vibratoAmplitude = rollingAmp.mean()
    vibAmpStd = rollingAmp.np.std()
    vibFreqTotal = autocorrVib3Hz(pandasContour, f_s_contour)
    vibAmpTotal, vibAmpStdTotal = vibAmp(pandasContour, f_s_contour, vibFreqTotal, windowFactor=0.75)
    
    # vibFreq = vibrato_Frequency
    # wavLengthWindow = 0.75*1/vibFreq*f_s_contour
    # meanFreq = pitch_contour.mean()
    # maxPeaks = find_peaks(pitch_contour, distance=wavLengthWindow)[0]
    # prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    # contour_heights = pitch_contour[maxPeaks] - prominences
    # plt.close()
    # plt.plot(pitch_contour)
    # plt.plot(maxPeaks, pitch_contour[maxPeaks], "x")
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Amplitude Calculation')
    # plt.vlines(x=maxPeaks, ymin=contour_heights, ymax=pitch_contour[maxPeaks],color='r')
    # plt.show()
    # print(str(vibratoAmplitude) + ' cents')
    # prompt = input("Press Enter to continue...")
    # plt.close()
    
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    #vibratoAmplitude, vibAmpStd = vibAmp(pitch_contour, f_s_contour, vibrato_Frequency, windowFactor=0.33)
    #Amplitude Calculations
    intensityAmplitude = np.nan#intensityAmp(stableWavArray, samplerate, vibrato_Frequency, windowFactor=0.33)
    #ampWindow = math.ceil(windowSecs*samplerate)
    #pandasContourAmp = pd.Series(stableWavArray)
    #rollingVibAmp = pandasContourAmp.rolling(ampWindow).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    #amplitudeFreq_rolling = rollingVibAmp.mean()
    #amplitudeFreq_simple = np.nan#autocorrVib3HzAmp(stableWavArray, samplerate)
    
    return vibrato_Frequency, vibratoPercentage, vibratoStd, vibratoAmplitude, vibAmpStd, vibFreqTotal, vibAmpTotal, vibAmpStdTotal

def predictVibFrame(rate, extent, non_normedVibArray, classifier):
    frameArray = np.array([rate, extent]).reshape(1,-1)
    #print(frameArray)
    if np.isnan(frameArray).any():
        return 0
    vibState = classifier.predict(preprocessing.StandardScaler().fit(non_normedVibArray).transform(frameArray))[0]
    return vibState

###Training Data Calculation
def vibratoCalcTraining(stableWavArray, samplerate, normedTrainingSet, classifier, gender=np.nan, windowSecs=0.5):
    ###Training Data Calculation
    #0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #sound = Sound(highestPitch, samplerate)
    #gender=geschlecht
    #windowSecs = 1#0.5, let's try one second.
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's go A-G
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(windowSecs*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    pandasContour = pd.Series(pitch_contour)
    rollingDF = pd.DataFrame({})
    #This only does one way
    rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #Calculate the initial window backwards by taking final "window" frames of pitch contour calculated in reverse
    rollingDF['rollingVib'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingVib'].iloc[:window] = rollingDF['rollingVib'].iloc[:window][::-1]

    rollingDF['rollingAmp'] = pandasContour.rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75))
    rollingDF['rollingAmp'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingAmp'].iloc[:window] = rollingDF['rollingAmp'].iloc[:window][::-1]

    #Check Non-Vibrato:
    rollingDF['vibState'] = rollingDF[['rollingVib', 'rollingAmp']].apply(lambda x: predictVibFrame(x.rollingVib, x.rollingAmp, X_0[:,:2], classifier), axis=1)
    
    #Let's only calculate values for windows classified as vibrato:
    vibrato_Frequency = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].mean()
    vibratoStd = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].np.std()
    vibratoAmplitude = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].mean()
    vibAmpStd = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].np.std()
    #vibFreqTotal = autocorrVib3Hz(pandasContour, f_s_contour)
    #vibAmpTotal, vibAmpStdTotal = vibAmp(pandasContour, f_s_contour, vibFreqTotal, windowFactor=0.75)
    
    # vibFreq = vibrato_Frequency
    # wavLengthWindow = 0.75*1/vibFreq*f_s_contour
    # meanFreq = pitch_contour.mean()
    # maxPeaks = find_peaks(pitch_contour, distance=wavLengthWindow)[0]
    # prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    # contour_heights = pitch_contour[maxPeaks] - prominences
    # plt.close()
    # plt.plot(pitch_contour)
    # plt.plot(maxPeaks, pitch_contour[maxPeaks], "x")
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Amplitude Calculation')
    # plt.vlines(x=maxPeaks, ymin=contour_heights, ymax=pitch_contour[maxPeaks],color='r')
    # plt.show()
    # print(str(vibratoAmplitude) + ' cents')
    # prompt = input("Press Enter to continue...")
    # plt.close()
    
    #vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    vibratoPercentage = len(rollingDF[rollingDF['vibState'] == 1])/len(rollingDF)
    #vibratoAmplitude, vibAmpStd = vibAmp(pitch_contour, f_s_contour, vibrato_Frequency, windowFactor=0.33)
    #Amplitude Calculations
    #intensityAmplitude = np.nan#intensityAmp(stableWavArray, samplerate, vibrato_Frequency, windowFactor=0.33)
    #ampWindow = math.ceil(windowSecs*samplerate)
    #pandasContourAmp = pd.Series(stableWavArray)
    #rollingVibAmp = pandasContourAmp.rolling(ampWindow).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    #amplitudeFreq_rolling = rollingVibAmp.mean()
    #amplitudeFreq_simple = np.nan#autocorrVib3HzAmp(stableWavArray, samplerate)
    
    return vibrato_Frequency, vibratoStd, vibratoAmplitude, vibAmpStd, vibratoPercentage


###Training Data Calculation
def vibratoCalcTrainingSimple(stableWavArray, samplerate, normedTrainingSet, classifier, gender=np.nan, windowSecs=0.5):
    ###Training Data Calculation
    #0.5 s window, 3Hz autoCorr
    sound = Sound(stableWavArray, samplerate)
    #sound = Sound(highestPitch, samplerate)
    #gender=geschlecht
    #windowSecs = 1#0.5, let's try one second.
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    #We actually just need the single pitches for this calculation, between C-E
        #Let's go A-G
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 784) #c4-g5
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    #r, lag = autocorr(pitch_contour)
    #vibrato_Frequency = 1/lag*f_s_contour
    #if (r < 0.5) | (vibrato_Frequency > 12):
    #    vibrato_Frequency = np.nan
    window = math.ceil(windowSecs*f_s_contour) # 0.5 s * sampling frequency of pitch contour
    if window > pitch_contour.shape[0]:
        window = pitch_contour.shape[0]-1
    
    vibrato_frequency =  autocorrVibNoThresh(pitch_contour, f_s_contour)
    vibratoAmplitude = vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75).iloc[(len(pandasContour)-window):]
    
    pandasContour = pd.Series(pitch_contour)
    rollingDF = pd.DataFrame({})
    #This only does one way
    rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #rollingDF['rollingVib'] = pandasContour.rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour))
    #Calculate the initial window backwards by taking final "window" frames of pitch contour calculated in reverse
    rollingDF['rollingVib'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: autocorrVibNoThresh(x, f_s_contour)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingVib'].iloc[:window] = rollingDF['rollingVib'].iloc[:window][::-1]

    rollingDF['rollingAmp'] = pandasContour.rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75))
    rollingDF['rollingAmp'].iloc[:window] = pandasContour[::-1].rolling(window).apply(lambda x: vibAmpRollNoThresh(x, f_s_contour,rollingDF['rollingVib'], windowFactor=0.75)).iloc[(len(pandasContour)-window):]
    rollingDF['rollingAmp'].iloc[:window] = rollingDF['rollingAmp'].iloc[:window][::-1]

    #Check Non-Vibrato:
    rollingDF['vibState'] = rollingDF[['rollingVib', 'rollingAmp']].apply(lambda x: predictVibFrame(x.rollingVib, x.rollingAmp, X_0[:,:2], classifier), axis=1)
    
    #Let's only calculate values for windows classified as vibrato:
    vibrato_Frequency = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].mean()
    vibratoStd = rollingDF[rollingDF['vibState'] == 1]['rollingVib'].np.std()
    vibratoAmplitude = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].mean()
    vibAmpStd = rollingDF[rollingDF['vibState'] == 1]['rollingAmp'].np.std()
    #vibFreqTotal = autocorrVib3Hz(pandasContour, f_s_contour)
    #vibAmpTotal, vibAmpStdTotal = vibAmp(pandasContour, f_s_contour, vibFreqTotal, windowFactor=0.75)
    
    # vibFreq = vibrato_Frequency
    # wavLengthWindow = 0.75*1/vibFreq*f_s_contour
    # meanFreq = pitch_contour.mean()
    # maxPeaks = find_peaks(pitch_contour, distance=wavLengthWindow)[0]
    # prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    # contour_heights = pitch_contour[maxPeaks] - prominences
    # plt.close()
    # plt.plot(pitch_contour)
    # plt.plot(maxPeaks, pitch_contour[maxPeaks], "x")
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Amplitude Calculation')
    # plt.vlines(x=maxPeaks, ymin=contour_heights, ymax=pitch_contour[maxPeaks],color='r')
    # plt.show()
    # print(str(vibratoAmplitude) + ' cents')
    # prompt = input("Press Enter to continue...")
    # plt.close()
    
    #vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    vibratoPercentage = len(rollingDF[rollingDF['vibState'] == 1])/len(rollingDF)
    #vibratoAmplitude, vibAmpStd = vibAmp(pitch_contour, f_s_contour, vibrato_Frequency, windowFactor=0.33)
    #Amplitude Calculations
    #intensityAmplitude = np.nan#intensityAmp(stableWavArray, samplerate, vibrato_Frequency, windowFactor=0.33)
    #ampWindow = math.ceil(windowSecs*samplerate)
    #pandasContourAmp = pd.Series(stableWavArray)
    #rollingVibAmp = pandasContourAmp.rolling(ampWindow).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    #amplitudeFreq_rolling = rollingVibAmp.mean()
    #amplitudeFreq_simple = np.nan#autocorrVib3HzAmp(stableWavArray, samplerate)
    
    return vibrato_Frequency, vibratoStd, vibratoAmplitude, vibAmpStd, vibratoPercentage




def autocorrVibNoThresh(pitch_contour, f_s_contour):
    x = pitch_contour
    n = len(x)
    acorr = sm.tsa.acf(x, nlags = n-1)
    #95% Confidence interval is +- 1.96/np.sqrt(n)
    ###DON'T NEED FOR TRAINING DATA
    #highCI = 1.96/np.sqrt(n)
    #lowCI = -highCI
    #Desired range is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = np.floor(1/10*f_s_contour)
    frame3Hz = np.floor(1/3*f_s_contour)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    vibratoFreq = 1/maxLag*f_s_contour
    #if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
    #    vibratoFreq = 1/maxLag*f_s_contour
    #else:
    #    vibratoFreq = np.nan
    #r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    #fig, ax = subplots(2)
    #ax.plot(1/np.arange(len(covariance[frame12Hz:frame3Hz]))*f_s_contour,covariance[frame12Hz:frame3Hz])
    #ax[0].plot(np.arange(len(acorr[frame10Hz:frame4Hz]))+frame10Hz, acorr[frame10Hz:frame4Hz])
    #ax[0].plot(np.arange(len(acorr)), acorr)
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #plt.vline
    #prompt = input("Press Enter to continue...")
    #plt.close()
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return vibratoFreq

def autocorrVib3Hz(pitch_contour, f_s_contour):
    x = pitch_contour
    n = len(x)
    acorr = sm.tsa.acf(x, nlags = n-1)
    #95% Confidence interval is +- 1.96/np.sqrt(n)
    highCI = 1.96/np.sqrt(n)
    lowCI = -highCI
    #Desired range is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = np.floor(1/10*f_s_contour)
    frame3Hz = np.floor(1/3*f_s_contour)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
        vibratoFreq = 1/maxLag*f_s_contour
    else:
        vibratoFreq = np.nan
    #r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    #fig, ax = subplots(2)
    #ax.plot(1/np.arange(len(covariance[frame12Hz:frame3Hz]))*f_s_contour,covariance[frame12Hz:frame3Hz])
    #ax[0].plot(np.arange(len(acorr[frame10Hz:frame4Hz]))+frame10Hz, acorr[frame10Hz:frame4Hz])
    #ax[0].plot(np.arange(len(acorr)), acorr)
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #plt.vline
    #prompt = input("Press Enter to continue...")
    #plt.close()
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return vibratoFreq

def vibratoCalcAmplitude(stableWavArray, samplerate):
#0.5 s window, 3Hz autoCorr
    sample = stableWavArray
    window = math.ceil(0.5*samplerate) # 0.5 s * sampling frequency of pitch contour
    pandasContour = pd.Series(sample)
    rollingVib = pandasContour.rolling(window).apply(lambda x: autocorrVib3HzAmp(x, samplerate))
    vibrato_Frequency = rollingVib.mean()
    vibratoStd = rollingVib.np.std()
    vibratoPercentage = len(rollingVib[window:][rollingVib[window:].notna()])/len(rollingVib[window:])
    return vibrato_Frequency, vibratoPercentage, vibratoStd

def autocorrVib3HzAmp(sustainedAudio, f_s):
    x = np.abs(hilbert(sustainedAudio))
    n = len(x)
    acorr = sm.tsa.acf(x, nlags = n-1)
    #95% Confidence interval is +- 1.96/np.sqrt(n)
    highCI = 1.96/np.sqrt(n)
    lowCI = -highCI
    #Desired range is 3 Hz - 10 Hz
        #=> Desired period is 1/10 s - 1/4 s
        #=> Desired lag times in frames are:
            #{1/10*f_s_contour:1/4*f_s_contour}
    frame10Hz = np.floor(1/10*f_s)
    frame3Hz = np.floor(1/3*f_s)
    #lag = np.abs(acorr)[frame12Hz:frame3Hz].argmax() + 1 + frame12Hz
    #maxLag = covariance[frame10Hz:frame4Hz].argmax() + frame10Hz
    maxLag = acorr[frame10Hz:frame3Hz].argmax() + frame10Hz
    if (acorr[:maxLag].min() < lowCI) & (acorr[maxLag] > highCI):
        vibratoFreq = 1/maxLag*f_s
    else:
        vibratoFreq = np.nan
    #r = acorr[lag-1]        
    # if np.abs(r) > 0.5:
      # print('Appears to be autocorrelated with r = {}, lag = {}'. format(r, lag))
    # else: 
      # print('Appears to be not autocorrelated')
    #fig, ax = subplots(2)
    #ax.plot(1/np.arange(len(covariance[frame12Hz:frame3Hz]))*f_s_contour,covariance[frame12Hz:frame3Hz])
    #ax[0].plot(np.arange(len(acorr[frame10Hz:frame4Hz]))+frame10Hz, acorr[frame10Hz:frame4Hz])
    #ax[0].plot(np.arange(len(acorr)), acorr)
    #pd.plotting.autocorrelation_plot(pitch_contour, ax=ax[1])
    #plt.vline
    #prompt = input("Press Enter to continue...")
    #plt.close()
    # ax[1].plot(result)
    # ax[2].plot(acorr)
    # ax[3].plot(r)
    return vibratoFreq

def vibAmp(pitch_contour, f_s_contour, vibFreq, windowFactor=0):
    if math.isnan(vibFreq):
        #vibFreq = 5.5
        ampCents = np.nan
        ampStd = np.nan
        #print(str(ampCents))
        return ampCents, ampStd
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = np.floor(wavelength*windowFactor)
    except ValueError:
        window = np.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    ampEstimate = prominences.mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences.np.std()
    #Now we need that in cents. How?
    #100 cent is a HALF step(?)
    #(x)^n*f_0 = 2*f_0, for the full octave where n is 12*100
    #25 cent is 33/32
    #x^25*f_0 = 33/32*f_0
    #=> x^1200 = 2
        #=> 1200*ln(x) = 2
        #=> x = e^(2/1200)
    cent = power(math.e,(log(2)/1200)) # Ok, so this is an even tempered cent for the octave
    #cent = 1.0005777895065548
    #ampEstimate in Hz
    #meanFreq + ampEstimate = meanFreq*(cent)^n where n is the number of cents
    #cent^n = (meanFreq + ampEstimate)/meanFreq
    #n*ln(cent) = ln(1 + ampEstimate/meanFreq)
    #n = ln(1 + ampEstimate/meanFreq)/ln(cent)
    ampCents = 1200*log(1 + ampEstimate/meanFreq)/log(2)
    ampStd = 1200*log(1 + ampStd/meanFreq)/log(2)
    return ampCents, ampStd # amplitude in cents

def vibAmpRoll(pitch_contour, f_s_contour, rollingVib, windowFactor=0):
    vibFreq = rollingVib[pitch_contour.index.max()]
    if math.isnan(vibFreq):
        #vibFreq = 5.5
        ampCents = np.nan
        #print(str(ampCents))
        return ampCents
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = np.floor(wavelength*windowFactor)
    except ValueError:
        window = np.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    ampEstimate = prominences[1:-1].mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences[1:-2].np.std()
    #
    ampCents = 1200*log(1 + ampEstimate/meanFreq)/log(2)
    #vibStd = 1200*log(1 + ampStd/meanFreq)/log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents
    

def vibAmpRollNoThresh(pitch_contour, f_s_contour, rollingVib, windowFactor=0):
    #Let's take the highest vibrato rate possible to then have the smallest wavelength threshold possible.
    vibFreq = rollingVib.max()#[pitch_contour.index.max()]
    #if math.isnan(vibFreq):
    #    #vibFreq = 5.5
    #    ampCents = np.nan
    #    #print(str(ampCents))
    #    return ampCents
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = np.floor(wavelength*windowFactor)
    except ValueError:
        window = np.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    ampEstimate = prominences[1:-1].mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences[1:-2].np.std()
    #
    ampCents = 1200*log(1 + ampEstimate/meanFreq)/log(2)
    #vibStd = 1200*log(1 + ampStd/meanFreq)/log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents

def intensityAmp(stableWav, samplerate, vibFreq, windowFactor=0):
    ampEnv = np.abs(stableWav)
    if vibFreq == np.nan:
        vibFreq = 5.5
    wavelength = 1/vibFreq*samplerate # in frames
    try:
        window = np.floor(wavelength*windowFactor)
    except ValueError:
        window = np.floor(1.0/5.5*f_s_contour*0.33)
    if window == 0:
        window = 1
    maxPeaks = find_peaks(ampEnv, distance=window)[0]
    minPeaks = find_peaks(ampEnv*-1,distance=window)[0]
    maxMean = ampEnv[maxPeaks].mean()
    minMean = ampEnv[minPeaks].mean()
    ampEstimate = (maxMean - minMean)/2 # in Hz
    meanAmplitude = ampEnv.mean()
    dB = np.np.log10(ampEstimate/meanAmplitude + 1) # np.np.log10((ampEstimate+meanAmplitude)/meanAmplitude)
    return dB # amplitude in cents

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    

  
#wavFilename = wavFilePath
#samplerate, data = read(wavFilename)
def selectMiddleTrial(data, samplerate):
    #load wav file.
    #wavFilename = wavFilename
    # samplerate, data = read(wavFilename)
    #Load audio file
    #If audio is stereo, take the left channel
    try:
        if data.shape[1] > 1:
            data = data[:,0]
    except:
        pass
    #There are some local zeros. Take rolling average and find max closest to midpoint
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    #Need a big moving average to not cut out mid-trial lows.
    #Let's use a half-second moving average (samplerate/2)
    a = moving_average(amplitude_envelope, n=round(samplerate/2))
    #Try to find larger peaks
    #Set all values where the amplitude_envelope is a factor of 2^3 lower to zero (-3dB)
    #Why -3 dB? Had one error, so let's lower the threshold to -4 dB
    a1 = np.where(a < a.max()/np.power(2,3), 0, a) 
    #Could be a problem with clipping^
    
    ###Find the amplitude envelope closest to the midpoint.
        #Could be a nonzero section with dB >-3 of max or a section closest to it.
        #If you find the sections where amplitude envelope != 0, you have an array of x tests
        #Choose the middle one.
    peaks = find_peaks(a1)
    #Calculate the distance from each to the midpoint
    distance_from_midpoint = abs(peaks[0] - round(len(a1)/2))
    minIndex = peaks[0][np.argmin(distance_from_midpoint)]
    #Ok, that gives us the local max closest to the midpoint.
    #Set zeros of audio intensity to -4 dB below this point.
    a = np.where(a < a[minIndex]/np.power(2,4), 0, a)
    #How do we find the section of the signal around that local max?
    #First, find zero prior to that point.
    if np.where(a[:minIndex] == 0)[0].size != 0:
        startMiddleAttempt = np.where(a[:minIndex]==0)[0][-1]
    else:
        startMiddleAttempt = 0  
    
    #Then find zero after that point. 
        #In one case, the recording ends before a zero.
    if np.where(a[minIndex:] == 0)[0].size != 0:
        finishMiddleAttempt = minIndex + np.where(a[minIndex:]==0)[0][0]
    else:
        finishMiddleAttempt = len(data)
    selectedMiddleTrial = data[startMiddleAttempt:finishMiddleAttempt]
    #visualCheckSelection(data, startMiddleAttempt, finishMiddleAttempt)
    #prompt = input("Press Enter to continue...")
    #if #prompt == 'q':
    #    break
    return samplerate, selectedMiddleTrial    
    
def selectFinalTrial(wavFilename):
    #load wav file.
    #wavFilename = wavFilename
    samplerate, data = read(wavFilename)
    #Load audio file
    #If audio is stereo, take the left channel
    try:
        if data.shape[1] > 1:
            data = data[:,0]
    except:
        pass
    #There are some local zeros. Take rolling average and find max closest to midpoint
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    #Need a big moving average to not cut out mid-trial lows.
    #Let's use a half-second moving average (samplerate/2)
    a = moving_average(amplitude_envelope, n=round(samplerate/2))
    #Try to find larger peaks
    #Set all values where the amplitude_envelope is a factor of 2^3 lower to zero (-3dB)
    #Why -3 dB? Had one error, so let's lower the threshold to -4 dB
    a1 = np.where(a < a.max()/np.power(2,3), 0, a) 
    #Could be a problem with clipping^
    
    ###Find the amplitude envelope closest to the midpoint.
        #Could be a nonzero section with dB >-3 of max or a section closest to it.
        #If you find the sections where amplitude envelope != 0, you have an array of x tests
        #Choose the middle one.
    peaks = find_peaks(a1)
    #Calculate the distance from each to the end of the file
    distance_from_midpoint = abs(peaks[0] - len(a1))
    minIndex = peaks[0][np.argmin(distance_from_midpoint)]
    #Ok, that gives us the local max closest to the midpoint.
    #Set zeros of audio intensity to -4 dB below this point.
    a = np.where(a < a[minIndex]/np.power(2,4), 0, a)
    #How do we find the section of the signal around that local max?
    #First, find zero prior to that point.
    if np.where(a[:minIndex] == 0)[0].size != 0:
        startFinalAttempt = np.where(a[:minIndex]==0)[0][-1]
    else:
        startFinalAttempt = 0  
    
    #Then find zero after that point. 
        #In one case, the recording ends before a zero.
    if np.where(a[minIndex:] == 0)[0].size != 0:
        finishFinalAttempt = minIndex + np.where(a[minIndex:]==0)[0][0]
    else:
        finishFinalAttempt = len(data)
    selectedFinalTrial = data[startFinalAttempt:finishFinalAttempt]
    #visualCheckSelection(data, startMiddleAttempt, finishMiddleAttempt)
    #prompt = input("Press Enter to continue...")
    #if #prompt == 'q':
    #    break
    return samplerate, selectedFinalTrial

def isolateHighestPitch50MF(samplerate, selectedMiddleTrial, gender=np.nan):
    #Can we get the pitch contour?
    sound = Sound(selectedMiddleTrial, samplerate)
    #Create a praat pitch object,
    #Probably need upper frequency bound 2x potential sung frequency
    #Piano key frequencies:
    #g5: 784
    #g4: 392
    #c4: 261
    #c3: 131
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    #What is the new samplingrate?
    f_s_Audio = sound.sampling_frequency
    wavLength = sound.values.size
    pitchContLength = pitch.selected_array['frequency'].size
    f_s_contour = pitchContLength/wavLength*f_s_Audio
    if len(pitch_contour[pitch_contour != 0]) < 1*f_s_contour:
        return samplerate, np.nan, np.nan, np.nan
    
    #So we have an interval of a minor third between the highest note and the middle note. 
    #Yeah!
    ###Ok, now we want to find the corresponding interval in the pitch_contour array    
        #That are within a minor third of the maximum pitch. 
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection and hope.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)  
    #Let's take the middle fifty percent of this interval.
    #If you save the audio file here, you could use it for all data analysis.
    #close('all')
    begin50 = beginInterval + round((endInterval - beginInterval)*.25)
    #print(str(begin50))
    end50 =  beginInterval + round((endInterval - beginInterval)*.75)
    #print(str(end50))
    #visualCheckSelection(pitch_contour, begin50, end50)
    #prompt = input("Press Enter to continue...")
    #beginAudioInterval = startMiddleAttempt + round(begin50*f_s_Audio/f_s_contour)
    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    #endAudioInterval = startMiddleAttempt + round(end50*f_s_Audio/f_s_contour)
    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt  
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq
    
def isolateHighestPitch2024(f_s_Audio, selectedMiddleTrial, pitch_contour, f_s_contour):
    #So we have an interval of a minor third between the highest note and the middle note. 
    #Yeah!
    ###Ok, now we want to find the corresponding interval in the pitch_contour array    
        #That are within a minor third of the maximum pitch. 
    #This is a little sensitive to pitch artifacts.
    #maxFreq = max(pitch_contour)
    #maxIndex = argmax(pitch_contour)
    #Let's just grab the middle value of the selection and hope.
    maxIndex = round(len(pitch_contour)/2)
    maxFreq = pitch_contour[maxIndex]
    #Minor 3rd ratio is 6:5
    thresholdFreq = maxFreq*5/6
    beginInterval = np.where(pitch_contour[:maxIndex] < thresholdFreq)[0][-1]
    if np.where(pitch_contour[maxIndex:] < thresholdFreq)[0].size != 0:
        endInterval = maxIndex + np.where(pitch_contour[maxIndex:] < thresholdFreq)[0][0]
    else:
        endInterval = len(pitch_contour)  
    #Let's take the middle fifty percent of this interval.
    #If you save the audio file here, you could use it for all data analysis.
    #close('all')
    begin50 = beginInterval + round((endInterval - beginInterval)*.25)
    #print(str(begin50))
    end50 =  beginInterval + round((endInterval - beginInterval)*.75)
    #print(str(end50))
    #visualCheckSelection(pitch_contour, begin50, end50)
    #prompt = input("Press Enter to continue...")
    #beginAudioInterval = startMiddleAttempt + round(begin50*f_s_Audio/f_s_contour)
    beginAudioInterval = round(begin50*f_s_Audio/f_s_contour) #+ startMiddleAttempt
    #endAudioInterval = startMiddleAttempt + round(end50*f_s_Audio/f_s_contour)
    endAudioInterval = round(end50*f_s_Audio/f_s_contour) #+ startMiddleAttempt  
    middleFiftyPercentHighestPitch = selectedMiddleTrial[beginAudioInterval:endAudioInterval]
    #Let's get the mean pitch of this interval
    meanFreq = pitch_contour[begin50:end50].mean()
    return samplerate, middleFiftyPercentHighestPitch, maxFreq, meanFreq


def visualizeResults(wavFilename, middleTrial, isolatedHighestPitch, samplerate, gender=np.nan):
    samplerate, data = read(wavFilename)
    #Visualize results:ipy
    fig, ax = plt.subplots(4)
    plt.rcParams['font.size'] = '14'
    ax[0].plot(np.arange(len(data))/samplerate,data)
    ax[0].set_title('Original Audio Waveform')
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].plot(np.arange(len(middleTrial))/samplerate,middleTrial)
    ax[1].set_title('Middle Trial')
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    sound = Sound(middleTrial, samplerate)
    #Create a praat pitch object,
    if gender == 'männl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c3-g4
    elif gender == 'weibl.':
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5
    else:
        pitch = call(sound, "To Pitch", 0.0, 60, 1000) #c4-g5 
    #This provides the frequencies of the sample.
    pitch_contour = pitch.selected_array['frequency']
    pitchContLength = pitch.selected_array['frequency'].size
    wavLength = len(middleTrial)
    f_s_contour = pitchContLength/wavLength*samplerate
    ax[2].plot(np.arange(len(pitch_contour))/f_s_contour,pitch_contour)
    ax[2].set_title('Pitch Contour Middle Task')
    ax[2].set_ylabel('Freq (Hz)', fontsize=14)
    #ax[2].axes.xaxis.set_visible(False)
    sound2 = Sound(isolatedHighestPitch, samplerate)
    #Create a praat pitch object,
    if gender == 'männl.':
        pitch2 = call(sound2, "To Pitch", 0.0, 100, 390) #c3-g4
    elif gender == 'weibl.':
        pitch2 = call(sound2, "To Pitch", 0.0, 261, 784) #c4-g5
    else:
        pitch2 = call(sound2, "To Pitch", 0.0, 60, 784) #c4-g5
    #This provides the frequencies of the sample.
    pitch_contour2 = pitch2.selected_array['frequency']
    pitchContLength2 = pitch2.selected_array['frequency'].size
    wavLength2 = len(isolatedHighestPitch)
    f_s_contour2 = pitchContLength2/wavLength2*samplerate
    ax[3].plot(np.arange(len(pitch_contour2))/f_s_contour2,pitch_contour2)
    ax[3].set_title('Pitch Contour Highest Pitch')
    ax[3].set_ylabel('Freq (Hz)', fontsize=14)
    #ax[3].axes.xaxis.set_visible(False)



#This for a single array check.
def showPitchContour(wavArray, samplerate):
    sound = Sound(wavArray, samplerate)
    #Create a praat pitch object,
    #Probably need upper frequency bound 2x potential sung frequency
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) 
    pitch_contour = pitch.selected_array['frequency']
    plt.subplots(1)
    plot(pitch_contour)

#Visually check middle sample against full sample
def visualCheckSelection(sample0, beginSample, endSample):
    plt.close('all')
    plt.subplots(1)
    plt.plot(sample0, color='b')
    plt.plot(np.arange(beginSample,endSample),sample0[beginSample:endSample], color='r')

def vibAmp(pitch_contour, f_s_contour, vibFreq, windowFactor=0):
    if math.isnan(vibFreq):
        #vibFreq = 5.5
        ampCents = np.nan
        #print(str(ampCents))
        return ampCents
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = np.floor(wavelength*windowFactor)
    except ValueError:
        window = np.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(pitch_contour, distance=window)[0]
    prominences = scipy.signal.peak_prominences(pitch_contour, maxPeaks)[0]/2
    #minPeaks = find_peaks(pitch_contour*-1,distance=window)[0]
    #maxMean = pitch_contour[maxPeaks].mean()
    #minMean = pitch_contour[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    ampEstimate = prominences[1:-1].mean()
    meanFreq = pitch_contour.mean()
    ampStd = prominences[1:-2].np.std()
    #
    ampCents = 1200*log(1 + ampEstimate/meanFreq)/log(2)
    #vibStd = 1200*log(1 + ampStd/meanFreq)/log(2)
    #if vibFreq == np.nan:
    #    ampCents = np.nan
    return ampCents#, vibStd # amplitude in cents


def vibAmpDB(amplitude_envelope, f_s_contour, vibFreq, windowFactor=0):
    if math.isnan(vibFreq):
        #vibFreq = 5.5
        extentDB = np.nan
        #print(str(extentDB))
        return extentDB
    wavelength = 1.0/vibFreq*f_s_contour # in frames
    try:
        window = np.floor(wavelength*windowFactor)
    except ValueError:
        window = np.floor(1.0/5.5*f_s_contour*0.75)
    #
    if window == 0:
        window = 1
    maxPeaks = find_peaks(amplitude_envelope, distance=window)[0]
    prominences = scipy.signal.peak_prominences(amplitude_envelope, maxPeaks)[0]/2
    #minPeaks = find_peaks(amplitude_envelope*-1,distance=window)[0]
    #maxMean = amplitude_envelope[maxPeaks].mean()
    #minMean = amplitude_envelope[minPeaks].mean()
    #ampEstimate = (maxMean - minMean)/2 # in Hz
    #Do not calculate with the first and final peaks.
    extentEstimate = prominences[1:-1].mean()
    meanAmp = amplitude_envelope.mean()
    extentStd = prominences[1:-2].np.std()
    #
    #ampCents = 1200*log(1 + ampEstimate/meanFreq)/log(2)
    extentDB = 20 * np.log10(extentEstimate / meanAmp)
    #vibStd = 1200*log(1 + ampStd/meanFreq)/log(2)
    #if vibFreq == np.nan:
    #    extentDB = np.nan
    return extentDB#, vibStd # amplitude in cents

def vibTremor(wavFile, samplerate):
    ###CALCULATE vibRate
    sound = Sound(wavFile, samplerate)
    #Creates PRAAT sound file from .wav array, default 44100 Hz sampling frequency?
    pitch = call(sound, "To Pitch", 0.0, 60, 1000) #create a praat pitch object
    pitch_contour = pitch.selected_array['frequency']
    #Calculate the contour's sample rate from the differing array sizes
    f_s_contour = pitch.selected_array['frequency'].size/sound.values.size*sound.sampling_frequency
    vibRate = autocorrVib3Hz(pitch_contour, f_s_contour)
    
    ###CALCULATE vibExtent
    vibExtent = vibAmp(pitch_contour, f_s_contour, vibRate, 0.75)#window factor of 0.75 the wavelength
    
    ###CALCULATE ampRate
    wavData = wavFile/max(max(wavFile),abs(min(wavFile)))
    analytic_signal = hilbert(wavData)
    amplitude_envelope = np.abs(analytic_signal)
    vibAmpRate = autocorrVib3Hz(amplitude_envelope, samplerate)
    
    ###Calculate ampExtent
    vibAmpExtent = vibAmpDB(amplitude_envelope, samplerate, vibAmpRate, 0.75)
    #vibAmpExtent = vibAmpPercent(amplitude_envelope, samplerate, vibAmpRate, 0.75)
    return vibRate, vibExtent, vibAmpRate, vibAmpExtent



def H2H1(data, samplerate, meanFreq):
    #Number of sample points
    N = len(data)
    #Sample spacing
    T = 1/samplerate
    yf = scipy.fft.fft(data)
    xf = scipy.fft.fftfreq(N, T)[:N//2]
    yf_plot = 2.0/N *np.abs(yf[0:N//2])
    #plt.plot(xf, yf_plot)
    #Next goal is to find the 
        #Sung frequency,
        #Intensity of that frequency in the FFT
        #Intensity of first harmonic in the FFT
        #H2/H1 in DB
    #meanFreq = 130
    #Try to find a maximum within 1/4 harmonic increments
    harmonicIncrement = round(meanFreq*0.25)
    #Frequency range of xf is:
    mask = np.where(((meanFreq-harmonicIncrement)<xf)&(xf<(meanFreq+harmonicIncrement)))
    max_H1 = xf[mask][yf_plot[mask].argmax()]
    H1_amp = yf[mask].argmax()

    H2_guess = 2*max_H1
    mask2 = np.where(((H2_guess-harmonicIncrement)<xf)&(xf<(H2_guess+harmonicIncrement)))
    max_H2 = xf[mask2][yf_plot[mask2].argmax()]
    H2_amp = yf[mask2].argmax()
    H2_H1 = 20*np.log10(H2_amp/H1_amp) # in dB
    return H2_H1

def H3H1(data, samplerate, meanFreq):
    #Number of sample points
    N = len(data)
    #Sample spacing
    T = 1/samplerate
    yf = scipy.fft.fft(data)
    xf = scipy.fft.fftfreq(N, T)[:N//2]
    yf_plot = 2.0/N *np.abs(yf[0:N//2])
    #plt.plot(xf, yf_plot)
    #Next goal is to find the 
        #Sung frequency,
        #Intensity of that frequency in the FFT
        #Intensity of first harmonic in the FFT
        #H3/H1 in DB
    #meanFreq = 130
    #Try to find a maximum within 1/4 harmonic increments
    harmonicIncrement = round(meanFreq*0.25)
    #Frequency range of xf is:
    mask = np.where(((meanFreq-harmonicIncrement)<xf)&(xf<(meanFreq+harmonicIncrement)))
    max_H1 = xf[mask][yf_plot[mask].argmax()]
    H1_amp = yf[mask].argmax()

    H3_guess = 2*max_H1
    mask3 = np.where(((H3_guess-harmonicIncrement)<xf)&(xf<(H3_guess+harmonicIncrement)))
    max_H3 = xf[mask3][yf_plot[mask3].argmax()]
    H3_amp = yf[mask3].argmax()
    H3_H1 = 20*np.log10(H3_amp/H1_amp) # in dB
    return H3_H1

def cpp(data, samplerate):
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    # cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, freqLow, freqHigh, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    return cpps
    
def cppsDreiklang(data, samplerate, pitchContour, medComp):
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    # cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") #Default
    #Calculate whether pitch is closer to d1, e1, d2, e2
    #medComp = np.array([d1,e1,d2,e2])
    #Pitch contour has to be np.nan cleaned
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]
    medArray = np.abs(medComp - np.median(cleanedContour))
    keyGuess = medComp[medArray.argmin()]
    root2 = np.sqrt(2)
    #Calculate the octave around the pitch
    f0_min = keyGuess/root2
    f0_max = keyGuess*root2
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    return cpps

def cppsAvezzo(data, samplerate, pitchContour, medComp):
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    # cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") #Default
    #Calculate whether pitch is closer to d1, e1, d2, e2
    #medComp = np.array([d1,e1,d2,e2])
    #Pitch contour has to be np.nan cleaned
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]
    medArray = np.abs(medComp - np.median(cleanedContour))
    keyGuess = medComp[medArray.argmin()]
    root2 = np.sqrt(2)
    #Calculate the octave around the pitch
    f0_min = keyGuess/root2
    f0_max = keyGuess*root2
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    return cpps

def timbre(data, samplerate, meanFreq):
    #samplerate, data = wavfile.read(monoFiles[i])
    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)
    #plt.figure()
    #plt.semilogy(f, RMS)
    #plt.title(monoFiles[i][:-8])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('Linear spectrum [V RMS]')
    #plt.show()
    #plt.savefig(str(monoFiles[i][:-8] + '.png'))
    idx1 = (f>2000)*(f<5000)
    mask = np.where(idx1)
    idx2 = (f>2000)
    mask2 = np.where(idx2)
    mask2a = np.where(~idx2)
    idx3 = (f>2700)*(f<3600)
    mask3 = np.where(idx3)
    idx4 = (f<2700)
    mask4 = np.where(idx4)
    f1_max = f[RMS.argmax()]
    f_s_max = f[mask][RMS[mask].argmax()]
    hammarberg = RMS[mask].max()/RMS.max()
    energyRatio_pabon = trapz(RMS[mask2])/trapz(RMS[mask2a])
    energyRatio_muerbe = trapz(RMS[mask3])/trapz(RMS[mask4])
    
    H1_idx = (f < meanFreq*(1.25))
    maskH1 = np.where(H1_idx)
    f_H1 = f[maskH1][RMS[maskH1].argmax()]
    frames_H1 = RMS[maskH1].argmax()
    peaks = find_peaks(RMS, distance=(frames_H1-1))
    
    return f1_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe

def timbreLTAS(f, RMS):
    #samplerate, data = wavfile.read(monoFiles[i])
    #f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    #RMS = np.sqrt(Pxx_spec)
    #plt.figure()
    #plt.semilogy(f, RMS)
    #plt.title(monoFiles[i][:-8])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('Linear spectrum [V RMS]')
    #plt.show()
    #plt.savefig(str(monoFiles[i][:-8] + '.png'))
    idx1 = (f>2000)*(f<5000)
    mask = np.where(idx1)
    idx2 = (f>2000)
    mask2 = np.where(idx2)
    mask2a = np.where(~idx2)
    idx3 = (f>2700)*(f<3600)
    mask3 = np.where(idx3)
    idx4 = (f<2700)
    mask4 = np.where(idx4)
    f1_max = f[RMS.argmax()]
    f_s_max = f[mask][RMS[mask].argmax()]
    hammarberg = RMS[mask].max()/RMS.max()
    energyRatio_pabon = trapz(RMS[mask2])/trapz(RMS[mask2a])
    energyRatio_muerbe = trapz(RMS[mask3])/trapz(RMS[mask4])
    
    H1_idx = (f < meanFreq*(1.25))
    maskH1 = np.where(H1_idx)
    f_H1 = f[maskH1][RMS[maskH1].argmax()]
    frames_H1 = RMS[maskH1].argmax()
    peaks = find_peaks(RMS, distance=(frames_H1-1))
    
    return f1_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe


def LTAS(data, samplerate, freqResolution=400):
    #samplerate, data = wavfile.read(monoFiles[i])
    #We want a frequency resolution of 400 Hz
    #This is 400 Hz = samplerate/nperseg
    #nperseg = samplerate/400
    n = round(samplerate/freqResolution)
    # f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 353, scaling='spectrum')#1024, scaling='spectrum')
    # f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', nperseg=n, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)
    LTASarray = np.array([f,RMS])
    # plt.figure()
    # plt.semilogy(f, RMS)
    #plt.title(monoFiles[i][:-8])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('Linear spectrum [V RMS]')
    # plt.show()
    # plt.savefig(str(monoFiles[i][:-8] + '.png'))
    # idx1 = (f>2000)*(f<5000)
    # mask = np.where(idx1)
    # idx2 = (f>2000)
    # mask2 = np.where(idx2)
    # mask2a = np.where(~idx2)
    # idx3 = (f>2700)*(f<3600)
    # mask3 = np.where(idx3)
    # idx4 = (f<2700)
    # mask4 = np.where(idx4)
    # f1_max = f[RMS.argmax()]
    # f_s_max = f[mask][RMS[mask].argmax()]
    # hammarberg = RMS[mask].max()/RMS.max()
    # energyRatio_pabon = trapz(RMS[mask2])/trapz(RMS[mask2a])
    # energyRatio_muerbe = trapz(RMS[mask3])/trapz(RMS[mask4])]
    
    # return f_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe
    return LTASarray

def slopeFunction(f_min1, f_min2, amp1, amp2, sf):
    slope = (amp2-amp1)/(f_min2-f_min1)
    intercept = amp1 - slope*f_min1 
    amp_estimate = slope*sf + intercept
    amp_estimate_db = 10*np.log10(sf/amp_estimate)
    return amp_estimate_db

def amp_db(f, RMS):
    #f = test[0]
    #RMS = test[1]
    idx0 = (f>2000)*(f<4000)
    mask0 = np.where(idx0)
    f[mask0] #Gives the frequencies between 2000 and 5000
    RMS[mask0] #Gives the LTAS RMS for those frequencies
    #Find the frequency of the maximum in this frequency range 
        #and the magnitude?
    sf = f[mask0][RMS[mask0].argmax()] # Frequency
    amp_sf = RMS[mask0].max() # Magnitude

    #Now we need to find the minima to the left and right of that peak
    idx1 = (f>2000)*(f<sf)
    mask1 = np.where(idx1)
    try:
        f_min1 = f[mask1][RMS[mask1].argmin()]
    except ValueError:
        return 0
    amp1 = RMS[mask1].min()

    idx2 = (f>sf)*(f<4000)
    mask2 = np.where(idx2)
    try:
        f_min2 = f[mask2][RMS[mask2].argmin()]
    except ValueError:
        return 0
    amp2 = RMS[mask2].min()
    
    amp_estimate_db = slopeFunction(f_min1, f_min2, amp1, amp2, sf)
    return amp_estimate_db


def vowelSD(data, samplerate):

    #testfile = '0001&2015_12_01&test1.wav'
    #samplerate, data = read(testfile)
    sound = Sound(data, samplerate)
    f0min=75
    f0max=700
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    t_list = []
    # df = pd.DataFrame({})
    # for t in range(0,numPoints):
        # t += 1
        # df.append({'t':t},ignore_index=True)
        #df = pd.concat([df, pd.DataFrame(t,columns=['t'])])
    # df['f1'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 1, x, 'Hertz', 'Linear'))
    # df['f2'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 2, x, 'Hertz', 'Linear'))
    # df['f3'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 3, x, 'Hertz', 'Linear'))
    for point in range(0, numPoints):
        point +=1
        t = call(pointProcess, "Get time from index", point)
        t_list.append(t)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
    # f2Array = np.array([t_list, f2_list])
    # i_e = np.where(f2Array[1] > 1700)
    # o_u = np.where(f2Array[1] < 1150)
    # a = np.where((f2Array[1] < 1700) & (f2Array[1] > 1150))

    # Let's smooth the formants with a moving average.
    # def moving_average(a, n=3):
        # ret = np.cumsum(a, dtype=float)
        # ret[n:] = ret[n:] - ret[:-n]
        # return ret[n - 1:] / n

    # f2Test = f2Array
    # f2Test[1,99:] = moving_average(f2Test[1])

    # f2Array = f2Test
    # i_e = np.where(f2Array[1] > 1700)
    # o_u = np.where(f2Array[1] < 1150)
    # a = np.where((f2Array[1] < 1700) & (f2Array[1] > 1150))
    # plt.close('all')
    # plt.plot(f2Array[0][i_e], f2Array[1][i_e], 'x')
    # plt.plot(f2Array[0][o_u], f2Array[1][o_u], 'x')
    # plt.plot(f2Array[0][a], f2Array[1][a], 'x')
    # plt.plot(f2Array[0], f2Array[1])
    df_f = pd.DataFrame(np.array([t_list, f1_list, f2_list, f3_list]).T, columns=['t', 'f1', 'f2', 'f3'])
    df_f['f2'] = df_f['f2'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    def categorize(f2):
        if f2 < 1150:
            return 'o_u'
        if f2 > 1750:
            return 'e_i'
        else:
            return 'a'
    df_f['vowel'] = df_f['f2'].apply(lambda x: categorize(x))
        
    #Goal: find three longest unbroken segments of a, e_i, and o_u
    resultDF = pd.DataFrame({}, columns=['a', 'e_i', 'o_u'],index=['Max', 'Index'])
    for i in ['a', 'e_i', 'o_u']:
        #print(i)
        mask = df_f['vowel'] == i
        try:
            resultDF.loc['Max', i] = (~mask).cumsum()[mask].value_counts().max()
            resultDF.loc['Index',i] = (~mask).cumsum()[mask].value_counts().index[0]
        except IndexError:
            #print('oh no!')
            return np.nan, np.nan, np.nan
    i_a0 = resultDF.loc['Index', 'a']
    i_af = i_a0 + resultDF.loc['Max', 'a']
    quartSpan_a = round((i_af - i_a0)/4)
    i_a0 = i_a0 + quartSpan_a
    i_af = i_af - quartSpan_a
    i_ei0 = resultDF.loc['Index', 'e_i']
    i_eif = i_ei0 + resultDF.loc['Max', 'e_i']
    quartSpan_ei = round((i_eif - i_ei0)/4)
    i_ei0 = i_ei0 + quartSpan_ei
    i_eif = i_eif - quartSpan_ei
    i_ou0 = resultDF.loc['Index', 'o_u']
    i_ouf = i_ou0 + resultDF.loc['Max', 'o_u']
    quartSpan_ou = round((i_ouf - i_ou0)/4)
    i_ou0 = i_ou0 + quartSpan_ou
    i_ouf = i_ouf - quartSpan_ou

    #audio frames
    x_a0 = round(df_f.loc[i_a0,'t']*samplerate)
    x_af = round(df_f.loc[i_af, 't']*samplerate)
    x_ei0 = round(df_f.loc[i_ei0,'t']*samplerate)
    x_eif = round(df_f.loc[i_eif,'t']*samplerate)
    x_ou0 = round(df_f.loc[i_ou0,'t']*samplerate)
    x_ouf = round(df_f.loc[i_ouf,'t']*samplerate)

    #Compare the root mean squared energy
    # RMS_full = np.sqrt(mean(np.square(data)))
    # RMS_a = np.sqrt(mean(np.square(data[x_a0:x_af])))
    # RMS_ei = np.sqrt(mean(np.square(data[x_ei0:x_eif])))
    # RMS_ou = np.sqrt(mean(np.square(data[x_ou0:x_ouf])))
    RMS_full = np.mean(rms(y=data))
    RMS_a = np.mean(rms(y=data[x_a0:x_af]))
    RMS_ei = np.mean(rms(y=data[x_ei0:x_eif]))
    RMS_ou = np.mean(rms(y=data[x_ou0:x_ouf]))


    Hamm_a, slope_a, _, _, _ = hammerbergSlope(data[x_a0:x_af],samplerate)
    Hamm_ei, slope_ei, _, _, _  = hammerbergSlope(data[x_ei0:x_eif],samplerate)
    Hamm_ou, slope_ou, _, _, _  = hammerbergSlope(data[x_ou0:x_ouf],samplerate)

    RMSvowels = [RMS_a, RMS_ei, RMS_ou]
    aDB = 10*np.log10(RMS_a/min(RMSvowels))
    eiDB = 10*np.log10(RMS_ei/min(RMSvowels))
    ouDB = 10*np.log10(RMS_ou/min(RMSvowels))

    ampSD = np.std([aDB, eiDB, ouDB])
    hammSD = np.std([Hamm_a, Hamm_ei, Hamm_ou])
    slopeSD = np.std([slope_a, slope_ei, slope_ou])

    return ampSD, hammSD, slopeSD


def skCategorize(df_notna, vowelNum=5):

    prep = MaxAbsScaler()
    kmeans = KMeans(n_clusters=vowelNum, random_state=0)

    scaled_data = prep.fit_transform(df_notna)
    kmeans.fit(scaled_data)

    df_notna['label'] = kmeans.labels_
    labelOrder = df_notna.groupby('label')['t'].median().sort_values().index
    if vowelNum == 5:
        vowels = ['a', 'e', 'i', 'o', 'u']

    if vowelNum == 3:
        vowels = ['a', 'e_i', 'o_u']
    vowelDict = {}        
    for i in range(vowelNum):
        vowelDict[labelOrder[i]] = vowels[i]
    df_notna['vowel'] = df_notna['label'].apply(lambda x: vowelDict[x])
    return df_notna
    
def normalize_mfcc(mfcc):
    # Mean and variance normalization (CMVN)
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc_normalized = (mfcc - mfcc_mean) / mfcc_std
    
    return mfcc_normalized

def mfccVar(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_n = normalize_mfcc(mfcc)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mfcc_n.T, mel_db, sr, np.arange(0, len(mfcc[0]) * hop_length / sr, hop_length / sr)



def skVowelSD(data, samplerate, freqLow, freqHigh):

    #testfile = '0001&2015_12_01&test1.wav'
    #samplerate, data = read(testfile)
    sound = Sound(data, samplerate)
    df_mfcc, df_mfcc_n = mfccDF3(data, samplerate)
    f0min=freqLow
    f0max=freqHigh
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    t_list = []
    # df = pd.DataFrame({})
    # for t in range(0,numPoints):
        # t += 1
        # df.append({'t':t},ignore_index=True)
        #df = pd.concat([df, pd.DataFrame(t,columns=['t'])])
    # df['f1'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 1, x, 'Hertz', 'Linear'))
    # df['f2'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 2, x, 'Hertz', 'Linear'))
    # df['f3'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 3, x, 'Hertz', 'Linear'))
    for point in range(0, numPoints):
        point +=1
        t = call(pointProcess, "Get time from index", point)
        t_list.append(t)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
    # f2Array = np.array([t_list, f2_list])
    # i_e = np.where(f2Array[1] > 1700)
    # o_u = np.where(f2Array[1] < 1150)
    # a = np.where((f2Array[1] < 1700) & (f2Array[1] > 1150))

    # Let's smooth the formants with a moving average.
    # def moving_average(a, n=3):
        # ret = np.cumsum(a, dtype=float)
        # ret[n:] = ret[n:] - ret[:-n]
        # return ret[n - 1:] / n

    # f2Test = f2Array
    # f2Test[1,99:] = moving_average(f2Test[1])

    # f2Array = f2Test
    # i_e = np.where(f2Array[1] > 1700)
    # o_u = np.where(f2Array[1] < 1150)
    # a = np.where((f2Array[1] < 1700) & (f2Array[1] > 1150))
    # plt.close('all')
    # plt.plot(f2Array[0][i_e], f2Array[1][i_e], 'x')
    # plt.plot(f2Array[0][o_u], f2Array[1][o_u], 'x')
    # plt.plot(f2Array[0][a], f2Array[1][a], 'x')
    # plt.plot(f2Array[0], f2Array[1])
    df_f = pd.DataFrame(np.array([t_list, f1_list, f2_list, f3_list]).T, columns=['t', 'f1', 'f2', 'f3'])
    df_f['f1'] = df_f['f1'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    df_f['f2'] = df_f['f2'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    df_f['f3'] = df_f['f3'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    
    #Remove nans
    df_f = df_f[(df_f['f1'].notna()) & (df_f['f2'].notna()) & (df_f['f3'].notna())]
    
    # df_f['vowel'] = df_f['f2'].apply(lambda x: categorize(x))
    df_f = skCategorize(df_f, vowelNum=3)
    # df_f = hmmCategorize(df_f, vowelNum=3)

    #Goal: find three longest unbroken segments of a, e_i, and o_u
    resultDF = pd.DataFrame({}, columns=['a', 'e_i', 'o_u'],index=['Max', 'Index'])
    for i in ['a', 'e_i', 'o_u']:
        #print(i)
        mask = df_f['vowel'] == i
        try:
        #Find longest stretch of given vowel
            resultDF.loc['Max', i] = (~mask).cumsum()[mask].value_counts().max()
            resultDF.loc['Index',i] = (~mask).cumsum()[mask].value_counts().index[0]
        except IndexError:
            #print('oh no!')
            return np.nan, np.nan, np.nan
    i_a0 = resultDF.loc['Index', 'a']
    i_af = i_a0 + resultDF.loc['Max', 'a']
    quartSpan_a = round((i_af - i_a0)/4)
    i_a0 = i_a0 + quartSpan_a
    i_af = i_af - quartSpan_a
    i_ei0 = resultDF.loc['Index', 'e_i']
    i_eif = i_ei0 + resultDF.loc['Max', 'e_i']
    quartSpan_ei = round((i_eif - i_ei0)/4)
    i_ei0 = i_ei0 + quartSpan_ei
    i_eif = i_eif - quartSpan_ei
    i_ou0 = resultDF.loc['Index', 'o_u']
    i_ouf = i_ou0 + resultDF.loc['Max', 'o_u']
    quartSpan_ou = round((i_ouf - i_ou0)/4)
    i_ou0 = i_ou0 + quartSpan_ou
    i_ouf = i_ouf - quartSpan_ou

    #audio frames
    x_a0 = round(df_f.loc[i_a0,'t']*samplerate)
    x_af = round(df_f.loc[i_af, 't']*samplerate)
    x_ei0 = round(df_f.loc[i_ei0,'t']*samplerate)
    x_eif = round(df_f.loc[i_eif,'t']*samplerate)
    x_ou0 = round(df_f.loc[i_ou0,'t']*samplerate)
    x_ouf = round(df_f.loc[i_ouf,'t']*samplerate)

    #Compare the root mean squared energy
    # RMS_full = np.sqrt(mean(np.square(data)))
    # RMS_a = np.sqrt(mean(np.square(data[x_a0:x_af])))
    # RMS_ei = np.sqrt(mean(np.square(data[x_ei0:x_eif])))
    # RMS_ou = np.sqrt(mean(np.square(data[x_ou0:x_ouf])))
    RMS_full = np.mean(rms(y=data))
    RMS_a = np.mean(rms(y=data[x_a0:x_af]))
    RMS_ei = np.mean(rms(y=data[x_ei0:x_eif]))
    RMS_ou = np.mean(rms(y=data[x_ou0:x_ouf]))
    
    LTASarray_a = LTAS_5000(data[x_a0:x_af], samplerate, freqResolution=400)
    LTASarray_H1H2_a = LTAS_5000(data[x_a0:x_af], samplerate, freqResolution=30)
    LTASarray_ei = LTAS_5000(data[x_ei0:x_eif], samplerate, freqResolution=400)
    LTASarray_H1H2_ei = LTAS_5000(data[x_ei0:x_eif], samplerate, freqResolution=30)
    LTASarray_ou = LTAS_5000(data[x_ou0:x_ouf], samplerate, freqResolution=400)
    LTASarray_H1H2_ou = LTAS_5000(data[x_ou0:x_ouf], samplerate, freqResolution=30)

    alpha_a = calcAlphaRatio(LTASarray_a[0], LTASarray_a[1])
    alpha_ei = calcAlphaRatio(LTASarray_ei[0], LTASarray_ei[1])
    alpha_ou = calcAlphaRatio(LTASarray_ou[0], LTASarray_ou[1])
    


    Hamm_a, slope_a, _, _, _ = logHammarbergSlope(data[x_a0:x_af],samplerate)
    Hamm_ei, slope_ei, _, _, _  = logHammarbergSlope(data[x_ei0:x_eif],samplerate)
    Hamm_ou, slope_ou, _, _, _  = logHammarbergSlope(data[x_ou0:x_ouf],samplerate)
   

    RMSvowels = [RMS_a, RMS_ei, RMS_ou]
    aDB = 10*np.log10(RMS_a/min(RMSvowels))
    eiDB = 10*np.log10(RMS_ei/min(RMSvowels))
    ouDB = 10*np.log10(RMS_ou/min(RMSvowels))

    ampSD = np.std([aDB, eiDB, ouDB])
    hammSD = np.std([Hamm_a, Hamm_ei, Hamm_ou])
    slopeSD = np.std([slope_a, slope_ei, slope_ou])
    alphaSD = np.std([alpha_a, alpha_ei, alpha_ou])

    f1med = df_f['f1'].median()
    f2med = df_f['f2'].median()
    f3med = df_f['f3'].median()
    
    f1med_a = df_f.loc[i_a0:i_af, 'f1'].median()
    f2med_a = df_f.loc[i_a0:i_af, 'f2'].median()
    f1med_ei = df_f.loc[i_ei0:i_eif, 'f1'].median()
    f2med_ei = df_f.loc[i_ei0:i_eif, 'f2'].median()
    f1med_ou = df_f.loc[i_ou0:i_ouf, 'f1'].median()
    f2med_ou = df_f.loc[i_ou0:i_ouf, 'f2'].median()
    
    df_mfcc_a, _ = mfccDF3(data[x_a0:x_af], samplerate)
    df_mfcc_ei, _ = mfccDF3(data[x_ei0:x_eif], samplerate)
    df_mfcc_ou, _ = mfccDF3(data[x_ou0:x_ouf], samplerate)    

    resultList = ['mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6']
    vowelList = ['a', 'ei', 'ou']
    dfList = [df_mfcc_a, df_mfcc_ei, df_mfcc_ou]
    mfccResults = pd.DataFrame(index=vowelList)
    for i in resultList:
        for j in range(len(dfList)):
            mfccResults.loc[vowelList[j],i] = dfList[j][i].mean()
    
    mfccSD = mfccResults.std()
    
    data_a = data[x_a0:x_af]
    
    # plt.close('all')
    # for i in range(1,4):
        # plt.plot(df_f['t'], df_f['f'+str(i)], color='black')

    # colors = ['blue', 'green', 'orange', 'purple', 'yellow']
    # vowels = df_f.groupby('vowel')['label'].first().index
    # for i in range(3):
        # mask = df_f['vowel'] == vowels[i]
        # for j in range(1,4):
            # plt.plot(df_f.loc[mask, 't'], df_f.loc[mask, 'f'+str(j)], 'o' , color=colors[i])
            # plt.text(df_f.loc[mask, 't'].median(), 1500, vowels[i], horizontalalignment='center')
    # for k in range(1,4): 
        # plt.plot(df_f.iloc[i_a0:i_af]['t'], df_f.iloc[i_a0:i_af]['f'+str(k)], 'o', color='red')
    # plt.show()
    
    voiceMin = df_f['t'].min()
    voiceMax = df_f['t'].max()
    df_mfcc = df_mfcc[(df_mfcc['t'] >= voiceMin) & (df_mfcc['t'] <= voiceMax)]
    d_mfcc1 = df_mfcc['mfcc_1'].std()
    d_mfcc2 = df_mfcc['mfcc_2'].std()
    d_mfcc3 = df_mfcc['mfcc_3'].std()
    d_mfcc4 = df_mfcc['mfcc_4'].std()
    d_mfcc5 = df_mfcc['mfcc_5'].std()
    d_mfcc6 = df_mfcc['mfcc_6'].std()
    
    return ampSD, hammSD, slopeSD, alphaSD, f1med_a, f2med_a, f1med_ei, f2med_ei, f1med_ou, f2med_ou,  data_a, d_mfcc1, d_mfcc2, d_mfcc3, d_mfcc4, d_mfcc5, d_mfcc6, mfccSD


def skVowelMFCC(data, samplerate, freqLow, freqHigh):

    #testfile = '0001&2015_12_01&test1.wav'
    #samplerate, data = read(testfile)
    sound = Sound(data, samplerate)
    df_mfcc, df_mfcc_n = mfccDF3(data, samplerate)
    f0min=freqLow
    f0max=freqHigh
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    t_list = []
    # df = pd.DataFrame({})
    # for t in range(0,numPoints):
        # t += 1
        # df.append({'t':t},ignore_index=True)
        #df = pd.concat([df, pd.DataFrame(t,columns=['t'])])
    # df['f1'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 1, x, 'Hertz', 'Linear'))
    # df['f2'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 2, x, 'Hertz', 'Linear'))
    # df['f3'] = df['t'].apply(lambda x: praat.call(formants, "Get value at time", 3, x, 'Hertz', 'Linear'))
    for point in range(0, numPoints):
        point +=1
        t = call(pointProcess, "Get time from index", point)
        t_list.append(t)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
    # f2Array = np.array([t_list, f2_list])
    # i_e = np.where(f2Array[1] > 1700)
    # o_u = np.where(f2Array[1] < 1150)
    # a = np.where((f2Array[1] < 1700) & (f2Array[1] > 1150))

    # Let's smooth the formants with a moving average.
    # def moving_average(a, n=3):
        # ret = np.cumsum(a, dtype=float)
        # ret[n:] = ret[n:] - ret[:-n]
        # return ret[n - 1:] / n

    # f2Test = f2Array
    # f2Test[1,99:] = moving_average(f2Test[1])

    # f2Array = f2Test
    # i_e = np.where(f2Array[1] > 1700)
    # o_u = np.where(f2Array[1] < 1150)
    # a = np.where((f2Array[1] < 1700) & (f2Array[1] > 1150))
    # plt.close('all')
    # plt.plot(f2Array[0][i_e], f2Array[1][i_e], 'x')
    # plt.plot(f2Array[0][o_u], f2Array[1][o_u], 'x')
    # plt.plot(f2Array[0][a], f2Array[1][a], 'x')
    # plt.plot(f2Array[0], f2Array[1])
    df_f = pd.DataFrame(np.array([t_list, f1_list, f2_list, f3_list]).T, columns=['t', 'f1', 'f2', 'f3'])
    df_f['f1'] = df_f['f1'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    df_f['f2'] = df_f['f2'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    df_f['f3'] = df_f['f3'].iloc[::-1].rolling(100).mean() #Let's smooth the formants in reverse.
    
    #Remove nans
    df_f = df_f[(df_f['f1'].notna()) & (df_f['f2'].notna()) & (df_f['f3'].notna())]
    
    # df_f['vowel'] = df_f['f2'].apply(lambda x: categorize(x))
    df_f = skCategorize(df_f, vowelNum=3)
    # df_f = hmmCategorize(df_f, vowelNum=3)

    #Goal: find three longest unbroken segments of a, e_i, and o_u
    resultDF = pd.DataFrame({}, columns=['a', 'e_i', 'o_u'],index=['Max', 'Index'])
    for i in ['a', 'e_i', 'o_u']:
        #print(i)
        mask = df_f['vowel'] == i
        try:
        #Find longest stretch of given vowel
            resultDF.loc['Max', i] = (~mask).cumsum()[mask].value_counts().max()
            resultDF.loc['Index',i] = (~mask).cumsum()[mask].value_counts().index[0]
        except IndexError:
            #print('oh no!')
            return np.nan, np.nan, np.nan
    i_a0 = resultDF.loc['Index', 'a']
    i_af = i_a0 + resultDF.loc['Max', 'a']
    quartSpan_a = round((i_af - i_a0)/4)
    i_a0 = i_a0 + quartSpan_a
    i_af = i_af - quartSpan_a
    i_ei0 = resultDF.loc['Index', 'e_i']
    i_eif = i_ei0 + resultDF.loc['Max', 'e_i']
    quartSpan_ei = round((i_eif - i_ei0)/4)
    i_ei0 = i_ei0 + quartSpan_ei
    i_eif = i_eif - quartSpan_ei
    i_ou0 = resultDF.loc['Index', 'o_u']
    i_ouf = i_ou0 + resultDF.loc['Max', 'o_u']
    quartSpan_ou = round((i_ouf - i_ou0)/4)
    i_ou0 = i_ou0 + quartSpan_ou
    i_ouf = i_ouf - quartSpan_ou

    #audio frames
    x_a0 = round(df_f.loc[i_a0,'t']*samplerate)
    x_af = round(df_f.loc[i_af, 't']*samplerate)
    x_ei0 = round(df_f.loc[i_ei0,'t']*samplerate)
    x_eif = round(df_f.loc[i_eif,'t']*samplerate)
    x_ou0 = round(df_f.loc[i_ou0,'t']*samplerate)
    x_ouf = round(df_f.loc[i_ouf,'t']*samplerate)
    
    mfcc_a = rawMFCC(data[x_a0:x_af], samplerate)
    mfcc_ei = rawMFCC(data[x_ei0:x_eif], samplerate)
    mfcc_ou = rawMFCC(data[x_ou0:x_ouf], samplerate)    

    voiceMin = df_f['t'].min()
    voiceMax = df_f['t'].max()
    df_mfcc = df_mfcc[(df_mfcc['t'] >= voiceMin) & (df_mfcc['t'] <= voiceMax)]
    d_mfcc1 = df_mfcc['mfcc_1'].std()
    d_mfcc2 = df_mfcc['mfcc_2'].std()
    d_mfcc3 = df_mfcc['mfcc_3'].std()
    d_mfcc4 = df_mfcc['mfcc_4'].std()
    d_mfcc5 = df_mfcc['mfcc_5'].std()
    d_mfcc6 = df_mfcc['mfcc_6'].std()
    
    return mfcc_a, mfcc_ei, mfcc_ou

### SpeechPy version
def mfccDF3(data, samplerate):
    mfcc = speechpy.feature.mfcc(data, sampling_frequency=samplerate, frame_length=0.020, frame_stride=0.01,
                 num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
    mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
    # Number of frames
    num_frames = mfcc.shape[0]

    # Compute time stamps for each frame
    time_stamps = np.arange(0, num_frames) * 0.01 + (0.02 / 2)
    df_mfcc = pd.DataFrame(mfcc, columns=[f'mfcc_{i}' for i in range(1, mfcc.shape[1] + 1)])
    df_mfcc_n = pd.DataFrame(mfcc_cmvn, columns=[f'mfcc_{i}' for i in range(1, mfcc_cmvn.shape[1] + 1)])
    df_mfcc['t'] = time_stamps
    df_mfcc_n['t'] = time_stamps
    return df_mfcc, df_mfcc_n

def rawMFCC(data, samplerate):
    mfcc = speechpy.feature.mfcc(data, sampling_frequency=samplerate, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, num_cepstral=128)
    mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
    return mfcc#mfcc_cmvn
    
def melSpectrogram(data, samplerate):
    if data.dtype == 'int16': #Convert to float
        data = data.astype(np.float32) / np.iinfo(np.int16).max
    D = np.abs(librosa.stft(data))**2
    S = librosa.feature.melspectrogram(S=D, sr=samplerate)
    return S

def hammarbergSlope(data, samplerate):#, meanFreq):
    #samplerate, data = wavfile.read(monoFiles[i])
    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)
    #plt.figure()
    #plt.semilogy(f, RMS)
    #plt.title(monoFiles[i][:-8])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('Linear spectrum [V RMS]')
    #plt.show()
    #plt.savefig(str(monoFiles[i][:-8] + '.png'))
    
    #Let's normalize to the peak below 1000 Hz in dBs
    idx0 = (f<1000)
    mask0 = np.where(idx0)
    F1 = f[mask0][RMS[mask0].argmax()]
    idxF1 = RMS[mask0].argmax()
    maxF1 = RMS[mask0].max()
    RMS = 10*np.log10(RMS/maxF1)
    idx1 = (f>2000)*(f<5000)
    mask = np.where(idx1)
    # idx2 = (f>2000)
    # mask2 = np.where(idx2)
    # mask2a = np.where(~idx2)
    # idx3 = (f>2700)*(f<3600)
    # mask3 = np.where(idx3)
    # idx4 = (f<2700)
    # mask4 = np.where(idx4)
    # f1_max = f[RMS.argmax()]
    # f_s_max = f[mask][RMS[mask].argmax()]
    hammarberg = RMS[mask].max()/maxF1
    # energyRatio_pabon = trapz(RMS[mask2])/trapz(RMS[mask2a])
    # energyRatio_muerbe = trapz(RMS[mask3])/trapz(RMS[mask4])
    
    x = f[idxF1:]
    y = RMS[idxF1:]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # H1_idx = (f < meanFreq*(1.25))
    # maskH1 = np.where(H1_idx)
    # f_H1 = f[maskH1][RMS[maskH1].argmax()]
    # frames_H1 = RMS[maskH1].argmax()
    # peaks = find_peaks(RMS, distance=(frames_H1-1))
    
    return hammarberg, slope, r_value, p_value, RMS#f1_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe

def logHammarbergSlope(data, samplerate):#, meanFreq):
    #samplerate, data = wavfile.read(monoFiles[i])
    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)

    #Let's normalize to the peak below 1000 Hz in dBs
    idx0 = (f<1000)
    mask0 = np.where(idx0)
    F1 = f[mask0][RMS[mask0].argmax()]
    idxF1 = RMS[mask0].argmax()
    maxF1 = RMS[mask0].max()
    logRMS = 10*np.log10(RMS/maxF1)
    maxF1_dB = logRMS[idxF1]
    idx1 = (f>2000)*(f<5000)
    mask = np.where(idx1)
    # idx2 = (f>2000)
    # mask2 = np.where(idx2)
    # mask2a = np.where(~idx2)
    # idx3 = (f>2700)*(f<3600)
    # mask3 = np.where(idx3)
    # idx4 = (f<2700)
    # mask4 = np.where(idx4)
    # f1_max = f[logRMS.argmax()]
    # f_s_max = f[mask][logRMS[mask].argmax()]
    hammarberg = logRMS[mask].max() - maxF1_dB
    # energyRatio_pabon = trapz(logRMS[mask2])/trapz(logRMS[mask2a])
    # energyRatio_muerbe = trapz(logRMS[mask3])/trapz(logRMS[mask4])
    
    x = f[idxF1:]
    y = logRMS[idxF1:]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # H1_idx = (f < meanFreq*(1.25))
    # maskH1 = np.where(H1_idx)
    # f_H1 = f[maskH1][RMS[maskH1].argmax()]
    # frames_H1 = RMS[maskH1].argmax()
    # peaks = find_peaks(RMS, distance=(frames_H1-1))
    
    return hammarberg, slope, r_value, p_value, RMS#f1_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe


def LTAS_5000(data, samplerate, freqResolution=400):
    #samplerate, data = wavfile.read(monoFiles[i])
    #We want a frequency resolution of 400 Hz
    #This is 400 Hz = samplerate/nperseg
    #nperseg = samplerate/400
    n = round(samplerate/freqResolution)
    #Let's try 100
    # n = round(samplerate/100)
    
    # f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 353, scaling='spectrum')#1024, scaling='spectrum')
    # f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', nperseg=n, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)
    #LTASarray = np.array([f,RMS])
    idx1 = (f<5000)
    mask = np.where(idx1)
    LTASarray = np.array([f[mask],RMS[mask]])
    # plt.figure()
    # plt.semilogy(f, RMS)
    #plt.title(monoFiles[i][:-8])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('Linear spectrum [V RMS]')
    # plt.show()
    # plt.savefig(str(monoFiles[i][:-8] + '.png'))
    # idx1 = (f>2000)*(f<5000)
    # mask = np.where(idx1)
    # idx2 = (f>2000)
    # mask2 = np.where(idx2)
    # mask2a = np.where(~idx2)
    # idx3 = (f>2700)*(f<3600)
    # mask3 = np.where(idx3)
    # idx4 = (f<2700)
    # mask4 = np.where(idx4)
    # f1_max = f[RMS.argmax()]
    # f_s_max = f[mask][RMS[mask].argmax()]
    # hammarberg = RMS[mask].max()/RMS.max()
    # energyRatio_pabon = trapz(RMS[mask2])/trapz(RMS[mask2a])
    # energyRatio_muerbe = trapz(RMS[mask3])/trapz(RMS[mask4])]
    
    # return f_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe
    return LTASarray

def calcPropE500(f, RMS):
    idx0 = (f<500)
    mask0 = np.where(idx0)
    idx1 = (f<5000)
    mask1 = np.where(idx1)
    propE_500 = RMS[mask0].sum()/RMS[mask1].sum()
    return propE_500 # unitless
    
def calcPropE1000(f, RMS):
    idx0 = (f<1000)
    mask0 = np.where(idx0)
    idx1 = (f<5000)
    mask1 = np.where(idx1)
    propE_1000 = RMS[mask0].sum()/RMS[mask1].sum()
    return propE_1000 # unitless
    
def calcAlphaRatio(f, RMS):
    idx0 = (f<1000)
    mask0 = np.where(idx0)
    idx1 = (f>1000)
    mask1 = np.where(idx1)
    # alphaRatio = RMS[mask0].sum()/RMS[mask1].sum()
    #Alpha Ratio is high/low energy
    # alphaRatio = RMS[mask1].sum()/RMS[mask0].sum()
    #This needs to be converted to decibels!!!
    alphaRatio = 10*np.log10(RMS[mask1].sum()/RMS[mask0].sum())
    return alphaRatio # unitless, proportion

def calcH1H2LTAS(f, RMS, f_0_min, f_0_max):
    #Comparing energy in the energy band of the f_0 range
        #to the energy an octave above
    #In Scherer & Sundberg (2015) they removed top notes from analysis
    #Let's just compare frequency bands from [f_0_max/2, f_0_max]
    if f_0_min < f_0_max/2:
        f_0_min = f_0_max/2
    idx0 = (f>f_0_min)*(f<f_0_max)
    mask0 = np.where(idx0)
    idx1 = (f>f_0_min*2)*(f<f_0_max*2)
    mask1 = np.where(idx1)
    # H1H2LTAS = 10*np.log10(RMS[mask0].sum()/RMS[mask1].sum())
    return H1H2LTAS # dB

def reCalcH1H2LTAS(data, samplerate, f_0_min, f_0_max):
    #Comparing energy in the energy band of the f_0 range
        #to the energy an octave above
    #In Scherer & Sundberg (2015) they removed top notes from analysis.
    #Also used 35 Hz bins 
    #This is 35 Hz = samplerate/nperseg
    #nperseg = samplerate/400
    n = round(samplerate/35)
    # f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 353, scaling='spectrum')#1024, scaling='spectrum')
    # f, Pxx_spec = signal.welch(data, samplerate, 'flattop', 1024, scaling='spectrum')
    f, Pxx_spec = signal.welch(data, samplerate, 'flattop', nperseg=n, scaling='spectrum')
    RMS = np.sqrt(Pxx_spec)
    #Let's just compare frequency bands from [f_0_max/2, f_0_max]
    if f_0_min < f_0_max/2:
        f_0_min = f_0_max/2
    idx0 = (f>f_0_min)*(f<f_0_max)
    mask0 = np.where(idx0)
    idx1 = (f>f_0_min*2)*(f<f_0_max*2)
    mask1 = np.where(idx1)
    # H1H2LTAS = 10*np.log10(RMS[mask0].sum()/RMS[mask1].sum())
    #Need to use the AVERAGE level in these frequency bands
    H1H2LTAS = 10*np.log10(RMS[mask0].mean()/RMS[mask1].mean())
    return H1H2LTAS # dB
    
#For flatnessLTAS we need the geometric mean:
def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def calcFlatnessLTAS(f, RMS):
    flatnessLTAS = 10*np.log10(geo_mean(RMS)/np.mean(RMS))
    return flatnessLTAS # dB

def calcCentroidLTAS(f, RMS):
    #We want the weighted mean frequency of the LTAS
    centroidLTAS = sum(f*RMS/RMS.sum())
    return centroidLTAS #Hz
    

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
    
# test = freqExtract(data, samplerate)
    
# f_0_min, f_0_max, f_0_mean = freqExtract(data, samplerate)
    

    

def HNR(data, samplerate):
    sound = Sound(data, samplerate)
    #Harmonic to Noise Ratio
    # 1. Praat calls HNR "Harmonicity", and Parselmouth does expose it. `Sound.to_harmonicity` should work, and give you an object you can query (`Harmonicity.get_value_at_time`).
    harmonicity = sound.to_harmonicity()
    #-200 is the null value
    HNR = harmonicity.values[harmonicity.values != -200].mean()
    return HNR


def normLTAS(LTAS):    
    idx = (LTAS[0]<5000)
    mask = np.where(idx)
    normFactor = sum(LTAS[1][mask])
    normLTAS = LTAS[1]/normFactor
    return normLTAS

def corrGNE(s1, s2):
    corr = np.corrcoef(s1,s2)[0,1]
    return corr
    
def corrGNE_2(s1, s2):
    maxCorr = 0
    for i in range(1,4):
        #Perform correlation for lags [-3,3]
        test1 = np.corrcoef(s1[i:],s2[:-i])[0,1]
        test2 = np.corrcoef(s2[i:],s1[:-i])[0,1]
        if test1 > maxCorr:
            maxCorr = test1
            i_f = i
            #order = 's0s2'
        if test2 > maxCorr:
            maxCorr = test2
            i_f = i
            #order = 's2s0'
    return maxCorr

def calcGNE(highestPitch, samplerate):
        
    sound = Sound(highestPitch, samplerate)
    # scipy.io.wavfile.write("testSoprano.wav", samplerate, highestPitch)
    # glottalFlow = call(sound, "To glottalFlow", 1, 0.0, 1.0, 44100, "x^2 - x^3")
    # glottalFlow = call(sound, "To PointProcess glottalFlow", 1, 0.0, 1.0, 44100, "x^2 - x^3")
    # source = To Sound (phonation): 44100, 0.6, 0.05, 0.7, 0.03, 3.0, 4.0
    #glottalFlow = call(sound, "To Sound (phonation)", 44100, 0.6, 0.05, 0.7, 0.03, 3.0, 4.0)
    #To Sound (phonation) not available for given objects
    #Do we need Point Process?
    f0min=75
    f0max=700
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    numPoints = call(pointProcess, "Get number of points")
    glottalFlow = call(pointProcess, "To Sound (phonation)", 10000, 0.6, 0.05, 0.7, 0.03, 3.0, 4.0)

    t_list = []
    flowAmp = []
    for point in range(0, numPoints):
        point +=1
        t = call(pointProcess, "Get time from index", point)
        t_list.append(t)
        flow = call(glottalFlow, "Get value at time", 1, t,  'Linear')#, 'Hertz')
        flowAmp.append(flow)

    #3 Calculate hilbert envelopes of 0-1000, 1000-2000, and 2000-3000 frequency bands
    #Apply a real DFT
    fft = np.fft.fft(glottalFlow.values[0])
    freq = np.fft.fftfreq(len(fft), d=1/samplerate)
    #Select a frequency band from the fft and apply a hanning window:
    #0-1000
    mask0 = (freq < 1000) & (freq > 0)
    fft0 = fft[mask0]
    hann0 = fft0*np.hanning(len(fft0))
    #1000-2000
    mask1 = (freq < 2000) & (freq > 1000)
    fft1 = fft[mask1]
    hann1 = fft1*np.hanning(len(fft1))
    #2000-3000
    mask2 = (freq < 3000) & (freq > 2000)
    fft2 = fft[mask2]
    hann2 = fft2*np.hanning(len(fft2))
    #3000-4000
    mask3 = (freq < 4000) & (freq > 3000)
    fft3 = fft[mask3]
    hann3 = fft3*np.hanning(len(fft3))

    #Set non-window (including negative frequencies to zero
    #Compute inverse fourier transform
    #Take absolute value of complex signal
    fft0_ = np.copy(fft)
    fft0_[~mask0] = 0
    fft0_[mask0] = hann0
    s0 = np.abs(scipy.fft.ifft(fft0_))

    fft1_ = np.copy(fft)
    fft1_[~mask1] = 0
    fft1_[mask1] = hann1
    s1 = np.abs(scipy.fft.ifft(fft1_))

    fft2_ = np.copy(fft)
    fft2_[~mask2] = 0
    fft2_[mask2] = hann2
    s2 = np.abs(scipy.fft.ifft(fft2_))
    
    fft3_ = np.copy(fft)
    fft3_[~mask3] = 0
    fft3_[mask3] = hann3
    s3 = np.abs(scipy.fft.ifft(fft3_))
    
    

    #4 Consider pair of envelopes for which:
        #difference of center frequencies >= half the bandwidth
        #Calculate cross correlation between such envelopes
    signalList = [s0,s1,s2]#,s3]

    maxCorr = 0
    for x, y in combinations(signalList, 2):
        i_Corr = corrGNE_2(x,y)
        if i_Corr > maxCorr:
            maxCorr = i_Corr
    GNE1k2k = corrGNE_2(s0,s1)
    GNE1k3k = corrGNE_2(s0,s2)
    GNE2k3k = corrGNE_2(s2,s1)
    GNE1k4k = corrGNE_2(s0,s3)
    GNE = maxCorr
    return GNE, GNE1k2k, GNE1k3k, GNE2k3k, GNE1k4k

def calcPitchContour(data, samplerate, freqLow=60, freqHigh=700,gender=np.nan):
    sound = Sound(data, samplerate)
    pitch = call(sound, "To Pitch", 0.1, freqLow, freqHigh) #c4-g5 
    pitch_contour = pitch.selected_array['frequency']
    if pitch_contour[pitch_contour != 0].size == 0:
        return np.nan, np.nan#, np.nan, np.nan
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
    return contourCorrect, f_s_contour

def calcH1H2LTAS_thresh(f, RMS, f_0_min, f_0_max):
    #Comparing energy in the energy band of the f_0 range
        #to the energy an octave above
    #Let's just compare frequency bands from [f_0_max/2, f_0_max]
    if f_0_min < f_0_max/2:
        f_0_min = f_0_max/2
    idx0 = (f>f_0_min)*(f<f_0_max)
    mask0 = np.where(idx0)
    idx1 = (f>f_0_min*2)*(f<f_0_max*2)
    mask1 = np.where(idx1)
    #Since we're converting Root-Mean-Squared, a power measure, we use 10*log
    #Need to use the AVERAGE of the levels in these bands, not the sum
    H1H2LTAS = 10*np.log10(RMS[mask0].mean()/RMS[mask1].mean())
    return H1H2LTAS # dB
    

def dreiklangH1H2LTAS(pitch_contour, f, RMS, medComp):
    #Calculate whether pitch is closer to d1, e1, d2, e2
    #medComp = np.array([d1,e1,d2,e2])
    #Pitch contour has to be np.nan cleaned
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]
    medArray = np.abs(medComp - np.median(cleanedContour))
    keyGuess = medComp[medArray.argmin()]
    root2 = np.sqrt(2)
    #Calculate the octave around the pitch
    f0_min = keyGuess/root2
    f0_max = keyGuess*root2
    H1H2LTAS = calcH1H2LTAS_thresh(f,RMS, f0_min, f0_max)#keyGuess, 2*keyGuess)
    return H1H2LTAS
    
def avezzoH1H2LTAS(pitch_contour, f, RMS):
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]

    d_mask = (cleanedContour > d) & (cleanedContour < d1)
    d1_mask = (cleanedContour > d1) & (cleanedContour < d2)
    dis_mask = (cleanedContour > dis) & (cleanedContour < dis1)
    dis1_mask = (cleanedContour > dis1) & (cleanedContour < dis2)
    pLen = cleanedContour.shape[0]
    dPerc = cleanedContour[d_mask].shape[0]/pLen
    d1Perc = cleanedContour[d1_mask].shape[0]/pLen
    disPerc = cleanedContour[dis_mask].shape[0]/pLen
    dis1Perc = cleanedContour[dis1_mask].shape[0]/pLen

    keyArray = np.array([dPerc, disPerc, d1Perc, dis1Perc])
    freqArray = np.array([d, dis, d1, dis1])
    keyGuess = freqArray[keyArray.argmax()]
    f0_min = keyGuess
    f0_max = keyGuess*2
    H1H2LTAS = calcH1H2LTAS_thresh(f,RMS, keyGuess, 2*keyGuess)
    return H1H2LTAS
    
def dreiklangH1H2LTAS_CPPs(pitch_contour, f, RMS, medComp, data, samplerate):
    #Calculate whether pitch is closer to d1, e1, d2, e2
    #medComp = np.array([d1,e1,d2,e2])
    #Pitch contour has to be np.nan cleaned
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]
    medArray = np.abs(medComp - np.median(cleanedContour))
    keyGuess = medComp[medArray.argmin()]
    # print(keyGuess)
    root2 = np.sqrt(2)
    #Calculate the octave around the pitch
    f0_min = keyGuess/root2
    f0_max = keyGuess*root2
    H1H2LTAS = calcH1H2LTAS_thresh(f,RMS, f0_min, f0_max)#keyGuess, 2*keyGuess)
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
   
    return H1H2LTAS, cpps, keyGuess
    
def avezzoH1H2LTAS_CPPs(pitch_contour, f, RMS, data, samplerate):
    cleanedContour = pitch_contour[~np.isnan(pitch_contour)]

    d_mask = (cleanedContour > d) & (cleanedContour < d1)
    d1_mask = (cleanedContour > d1) & (cleanedContour < d2)
    dis_mask = (cleanedContour > dis) & (cleanedContour < dis1)
    dis1_mask = (cleanedContour > dis1) & (cleanedContour < dis2)
    pLen = cleanedContour.shape[0]
    dPerc = cleanedContour[d_mask].shape[0]/pLen
    d1Perc = cleanedContour[d1_mask].shape[0]/pLen
    disPerc = cleanedContour[dis_mask].shape[0]/pLen
    dis1Perc = cleanedContour[dis1_mask].shape[0]/pLen

    keyArray = np.array([dPerc, disPerc, d1Perc, dis1Perc])
    freqArray = np.array([d, dis, d1, dis1])
    keyGuess = freqArray[keyArray.argmax()]
    # print(keyGuess)
    f0_min = keyGuess
    f0_max = keyGuess*2
    H1H2LTAS = calcH1H2LTAS_thresh(f,RMS, keyGuess, 2*keyGuess)
    sound = Sound(data, samplerate)
    cepstogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(cepstogram, "Get CPPS", "yes", 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust") 
    
    return H1H2LTAS, cpps, keyGuess


def deNoise(data, samplerate, pitchDF, f_s_contour):
    #Find longest silence in pitch contour
    i_nonZero = (pitchDF[0].notna()).argmax()
    # contourCorrect[contourCorrect == 0] = np.nan
    # i_nonZero = (contourCorrect != np.nan).argmax()
    noiseFrame = int(np.floor(i_nonZero*samplerate/f_s_contour))
    noisy_part = data[:noiseFrame] 
    # perform noise reduction 
    reduced_noise = nr.reduce_noise(data,samplerate, y_noise=noisy_part)
    return reduced_noise

def calcSNR(data, samplerate, pitch_contour, f_s_contour):
    #Find longest "silence" in pitch contour
    i_nonZero = (pitchDF[0].notna()).argmax()
    # contourCorrect[contourCorrect == 0] = np.nan
    # i_nonZero = (contourCorrect != np.nan).argmax()
    noiseFrame = int(np.floor(i_nonZero*samplerate/f_s_contour))
    noisy_part = data[:noiseFrame] 
    #Calculate SNR in dB
    RMS = math.sqrt(np.mean(data**2))
    RMS_n = math.sqrt(np.mean(noisy_part**2))
    SNR = 10*np.log10(RMS/RMS_n)
    return SNR
    
    
def findNoise(data, samplerate, pitchDF, f_s_contour):
    #Find longest "silence" in pitch contour
    i_nonZero = (pitchDF[0].notna()).argmax()
    #Check the end of the file as well
    i_flipped = (np.flip(pitchDF[0]).notna()).argmax()
    if i_nonZero > i_flipped:
        noiseFrame = int(np.floor(i_nonZero*samplerate/f_s_contour))
        noisy_part = data[:noiseFrame] 
        plt.close()
        plt.plot(data)
        plt.plot(data[:noiseFrame])
    else:
        noiseFrame = int(np.floor(i_flipped*samplerate/f_s_contour))
        noisy_part = data[-noiseFrame:] 
        plt.close()
        plt.plot(np.arange(len(data)),data)
        plt.plot(np.arange(len(data)-noiseFrame,len(data)),data[-noiseFrame:] )

    
    return noisy_part

def findNoise2(data, samplerate, pitchDF, f_s_contour):
    #Find longest "silence" in pitch contour
    i_nonZero = (pitchDF[0].notna()).argmax()
    #Check the end of the file as well
    i_flipped = (np.flip(pitchDF[0]).notna()).argmax()
    beg0 = int(np.floor(i_nonZero*samplerate/f_s_contour*0.25))
    begF = int(np.floor(i_nonZero*samplerate/f_s_contour*0.75))
    begFrame = int(np.floor(i_nonZero*samplerate/f_s_contour))
    
    end0 = int(np.floor(i_flipped*samplerate/f_s_contour*0.25))
    endF = int(np.floor(i_flipped*samplerate/f_s_contour*0.75))
    endFrame = int(np.floor(i_flipped*samplerate/f_s_contour))
    #Find "quieter" noise (trying to avoid clipping)
    
    begPow = signalPower(data[beg0:begF])
    endPow = signalPower(data[-endF:-end0])

    if ((begPow > endPow) & (endFrame != 0)):
        noisy_part = data[-endF:-end0]
        # plt.close()
        # plt.plot(np.arange(len(data)),data)
        # plt.plot(np.arange(len(data)-endF,len(data)-end0),data[-endF:-end0] )
    elif ((begPow < endPow) & (begFrame != 0)):
        noisy_part = data[beg0:begF]
        # plt.close()
        # plt.plot(np.arange(len(data)), data)
        # plt.plot(np.arange(beg0,begF), data[beg0:begF])
    elif i_nonZero > i_flipped:
        noisy_part = data[beg0:begF] 
        # plt.close()
        # plt.plot(np.arange(len(data)), data)
        # plt.plot(np.arange(beg0,begF), data[beg0:begF])
    else:
        noisy_part = data[-endF:-end0]
        # plt.close()
        # plt.plot(np.arange(len(data)),data)
        # plt.plot(np.arange(len(data)-endF,len(data)-end0),data[-endF:-end0] )
    #Calculate SNR in dB
    # RMS = math.sqrt(np.mean(data**2))
    # RMS_n = math.sqrt(np.mean(noisy_part**2))
    # SNR = 10*np.log10(RMS/RMS_n)
    duration_n = len(noisy_part)/samplerate
    return noisy_part, duration_n
    


def calcSNR_simple(data, noisy_part):
    #A weird thing is happening with our arays where they need to be floats
    RMS = math.sqrt(np.mean(data.astype(float)**2))
    RMS_n = math.sqrt(np.mean(noisy_part.astype(float)**2))
    SNR = 10*np.log10(RMS/RMS_n)
    return SNR

def signalPower(x):
    return np.mean(x.astype(float)**2)
    
def calcSNR_simplest(signal, noise):
    powS = signalPower(signal)
    powN = signalPower(noise)
    return 10*np.log10((powS-powN)/powN)

def addNoise(signal):
    ###ADD RANDOM NOISE
    RMS = np.sqrt(np.mean(signal**2))
    SNR = np.random.random()*15 #Generate random value between 0(!) and 15 dB
    RMS_noise = np.sqrt(RMS**2/(10**(SNR/10)))
    STD_n = RMS_noise
    noise = np.random.normal(0, STD_n, signal.shape[0])
    data = signal+noise
    return data

def normalize_to_target_rms(audio, target_rms):
    audio_rms = np.sqrt(np.mean(audio**2))
    if audio_rms == 0:
        return audio
    return audio * (target_rms / audio_rms)

def singleHarmonicSpec(data, samplerate,freqResolution):
    
    # Parameters
    n_fft = int(samplerate / freqResolution)
    hop_length = n_fft // 4

    # Load audio

    # Compute spectrogram
    S = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))

    # Display
    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             # sr=sr, hop_length=hop_length, y_axis='linear', x_axis='time')
    # plt.title(f"Spectrogram with ~{freqResolution} Hz resolution")
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    meanSpec = np.mean(S, axis=1) #Calculate mean spectrum over time
    return meanSpec

def get_mel(y, sr, n_mels=128,f_min=1000,f_max=5000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=f_max, fmin=f_min)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def pad_or_truncate(mel, target_frames):
    if mel.shape[1] < target_frames:
        pad_width = target_frames - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel = mel[:, :target_frames]
    return mel

import numpy as np
import librosa
from librosa.sequence import dtw

def pad_or_truncate_mfcc(mfcc, target_len):
    """Pad or truncate an MFCC matrix to match target_len time steps."""
    current_len, n_mfcc = mfcc.shape
    if current_len >= target_len:
        return mfcc[:target_len, :]
    else:
        pad_width = target_len - current_len
        return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')




# def visualizeLTAS(LTAS1, LTAS2, pitchVerlauf):
    
# vibratoDF = pd.read_csv("VibratoUntersuchung.csv")
# with open('classVib_Anfang_Retro.pkl', 'rb') as f:
    # anamDB = pickle.load(f)  
# df = pd.merge(vibratoDF, anamDB[['audioID', 'stimme.lage.beginn', 'geschlecht']], left_on='id', right_on='audioID')
#Randfälle Untersuchung
#df = df[df['duration'] > 28]
#df = df[df['id'] == 31]
#for i in [121,372]:#df.index:
#Sort by date, then by id.

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
#indexArray = df.index
# df = pd.DataFrame({})
# for i in range(len(singingSamples)):
    # if singingSamples[i] == 'C:\\Users\\Reuben\\Documents\\Code\\Promotionsvorhaben\\Sandbox\\0072&2010_06_25&test 7.wav':
        # print(i)#1782
root2 = np.sqrt(2)
root12 = np.power(2, 1/12)
prompt = ''
# with open('SNR.pkl', 'rb') as f:
    # df = pickle.load(f) 
i = 0
df = pd.DataFrame({})
ref_path = '0011&2009_01_27&test5.wav' # sicher, Solo soprano with 0.06 s difference from median duration
ref_y, sr = librosa.load(ref_path, sr=16000)
ref_mel = get_mel(ref_y, sr)
# --- Extract MFCCs (transposed so time is axis 0) ---
ref_mfcc = librosa.feature.mfcc(y=ref_y, sr=sr, n_mfcc=13).T  

for wavFilename in singingSamples:
    # i += 1
    # if i >= 10:
        # break
    # wavFilename = random.choice(singingSamples)
    #wavFilename = df.loc[i].loc['newFilename']
    #geschlecht = df.loc[i].loc['geschlecht']
    #samplerate, data = read(wavFilename)
    #About 50% of the samples are 16 kHz samplerate, 50% 41kHz and a few 48 kHz
    #Let's downsample to 16 kHz.
    data, samplerate = librosa.load(wavFilename, mono=False, sr=16000)   
    sr = samplerate
    
    #Convert to floating point if int
    if data.dtype == 'int16': #Convert to float
        data = data.astype(np.float32) / np.iinfo(np.int16).max
    # data = librosa.to_mono(data)
    
    duration = len(data)/samplerate #seconds
    trialNum = wavFilename.split('\\')[-1][-5]
    idNum = int(wavFilename.split('\\')[-1].split('&')[0])
    date = wavFilename.split('\\')[-1].split('&')[1]
    
    ### Check to see if found in 
    # Step 2: Convert to datetime
    # try:
        # date_from_file = datetime.strptime(date.replace(' - Kopie','').replace('-','_'), '%Y_%m_%d')
    # except:
        # print('Continue')
        # continue
        
    # Step 3: Ensure 'date' column is datetime if not already
    # df0['date'] = pd.to_datetime(df0['date'])

    # Step 4: Check if (id, date) exists in the dataframe
    # match = ((df0['id'] == idNum) & (df0['date'] == date_from_file)).any()
    # if ~match:
        # continue
        
    # print("Match found?" , match, ': ',idNum,', ',date)
    
    if duration < 2:
        duration = np.nan
        resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      'date':wavFilename.split('\\')[-1].split('&')[1],
                      'trialNum':wavFilename.split('\\')[-1][-5],
                      'duration':duration
                      }
        df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
        continue
    # if duration > 59:
        # duration = np.nan
        # continue
    if len(data.shape) > 1: #Convert to mono
        data = data[:,0]
    
    ###Let's immediately normalize:
    ###Let's amplitude normalize right here:
    # if df[((df['id'] == idNum) & (df['date'] == date) & (df['trialNum'] == '4'))].size == 0:
        # resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      # 'date':wavFilename.split('\\')[-1].split('&')[1],
                      # 'trialNum':wavFilename.split('\\')[-1][-5],
                      # 'duration':duration,
                      # 'target_RMS':np.nan
                      # }
        # df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
        # continue
    # data = addNoise(data)
    
    ### Normalize data to quietest RMS of messa di voce:
    # target_RMS = df[((df['id'] == idNum) & (df['date'] == date) & (df['trialNum'] == '4'))]['ref_RMS'][0]
    # data = normalize_to_target_rms(data, target_RMS)
    
    #Select Middle Trial
    if (duration > 20) & (trialNum != '6') & (trialNum != '5'):
        samplerate, middleTrial = selectMiddleTrial(data, samplerate)
    else:
        middleTrial = data
    
    ###Now we shift the pitch randomly twice 
    choices = [-2,-1,1,2]
    augment = []#random.sample(choices,2) #Randomly raise 
    augment.append(0) # Keep original
    for pitchChange in augment:
        augmented = True
        if pitchChange == 0:
            augmented = False
        
        ### Pitch Shift
        #Shift the data randomly between -2 to 2 halfsteps or retain the same
        data_shift = librosa.effects.pitch_shift(data, sr=sr, n_steps=pitchChange)
        
        ### Dynamic Time Warp Audio to ReferenceError
        # --- Compute MFCCs or mel spectrograms in dB ---
        # input_mel = get_mel(data_shift, samplerate)

        # --- DTW alignment ---
        # D, wp = dtw(X=ref_mel, Y=input_mel, metric='euclidean')
        # wp = np.array(wp[::-1])  # forward time
        # aligned_input_mel = input_mel[:, wp[:, 1]]
        

        # Optionally pad or truncate to match reference length
        # melSpec = pad_or_truncate(aligned_input_mel, ref_mel.shape[1])

        input_mfcc = librosa.feature.mfcc(y=data_shift, sr=sr, n_mfcc=13).T

        # --- DTW Alignment ---
        # Note: We transpose to shape (n_mfcc, time) for DTW to work correctly
        D, wp = dtw(X=ref_mfcc.T, Y=input_mfcc.T, metric='euclidean')
        wp = np.array(wp[::-1])  # Reverse to forward path

        # Align input MFCC to reference using DTW path
        aligned_input_mfcc = input_mfcc[wp[:, 1], :]

        # --- Truncate or pad to match reference length ---
        final_mfcc = pad_or_truncate_mfcc(aligned_input_mfcc, ref_mfcc.shape[0])



        # if trialNum == '2':
            # samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch50MF(samplerate, middleTrial)
            # data = highestPitch
            # if type(data) == float:
                # continue
        if trialNum == '1':
            fLow = g/root2
            fHigh = a1*root2
        if trialNum == '2':
            # fLow = d1/root2
            fLow = g/root2
            fHigh = e2*root2
            ampSD = np.nan
            slopeSD = np.nan
            hammSD = np.nan
        if trialNum == '6':
            # fLow = d1/root2
            fLow = 60
            fHigh = e2*root2
            ampSD = np.nan
            slopeSD = np.nan
            hammSD = np.nan
        pitch_contour, f_s_contour = calcPitchContour(middleTrial, samplerate)#, freqLow=fLow, freqHigh=fHigh)

        if type(pitch_contour) == float:
            resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                          'date':wavFilename.split('\\')[-1].split('&')[1],
                          'trialNum':wavFilename.split('\\')[-1][-5],
                          'duration':duration,
                          'pitch_contour':pitch_contour
                          }
            df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
            continue                      
        pitchDF = pd.DataFrame(pitch_contour)
        if pitchDF[pitchDF[0].notna()].shape[0]/f_s_contour < 2:
            pitch_contour = np.nan
            resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                          'date':wavFilename.split('\\')[-1].split('&')[1],
                          'trialNum':wavFilename.split('\\')[-1][-5],
                          'duration':duration,
                          'pitch_contour':pitch_contour
                          }
            df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
            continue
        # noisy_part, duration_n = findNoise2(data, samplerate, pitchDF, f_s_contour)
        if trialNum == '1':
            # ampSD, hammSD, slopeSD, alphaSD, f1med_a, f2med_a, f1med_ei, f2med_ei, f1med_ou, f2med_ou, data, d_mfcc1, d_mfcc2, d_mfcc3, d_mfcc4, d_mfcc5, d_mfcc6, mfccSD = skVowelSD(middleTrial, samplerate, fLow, fHigh)
            mfcc_a, mfcc_ei, mfcc_ou = skVowelMFCC(middleTrial, samplerate, fLow, fHigh)
            pitch_contour, f_s_contour = calcPitchContour(data, samplerate)#, freqLow=fLow, freqHigh=fHigh)
            #data = middleTrial
        if trialNum == '2':
            samplerate, highestPitch, maxFreq, meanFreq = isolateHighestPitch2024(samplerate, middleTrial, pitch_contour, f_s_contour)
            data = highestPitch
            if trialNum == '2':
                fLow = d1/root2
                fHigh = e2*root2
            pitch_contour, f_s_contour = calcPitchContour(data, samplerate, freqLow=fLow, freqHigh=fHigh)
            pitchDF = pd.DataFrame(pitch_contour)
            if pitchDF[pitchDF[0].notna()].shape[0]/f_s_contour < 2:
                pitch_contour = np.nan
                resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                              'date':wavFilename.split('\\')[-1].split('&')[1],
                              'trialNum':wavFilename.split('\\')[-1][-5],
                              'duration':duration,
                              'pitch_contour':pitch_contour
                              }
                df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
                continue
        
        # melSpec = melSpectrogram(data, samplerate)
        f_0_min, f_0_max, f_0_median, f_0_mean = freqExtract(data, samplerate)
        for i in range(4):
            low, high = freqPairs[i]
            if ((low < f_0_median) & (f_0_median < high)):
                targetF0 = targetFreqs[i]
                break
        if trialNum == '2':
            freqBand = targetF0*root12**2 #Band up two semitones from targetF0
            meanSpec = singleHarmonicSpec(data, samplerate, freqResolution=freqBand)
        else:
            freqBand = np.nan
            meanSpec = np.nan
        # SNR = calcSNR_simplest(data, noisy_part)
        # RMS_data = librosa.feature.rms(y=data)
        # dSPL = 20*np.log10(RMS_data/target_RMS)
        resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      'date':wavFilename.split('\\')[-1].split('&')[1],
                      'trialNum':wavFilename.split('\\')[-1][-5],
                      # 'pitch_contour':pitch_contour,
                      # 'LTAS_400':LTASarray,
                      # 'LTAS_30':LTASarray_H1H2,
                      'duration':duration,
                      # 'pitchMin':f_0_min,
                      'pitchMed':f_0_median,
                      # 'keyGuess':keyGuess,
                      # 'hammarberg':hammarberg,
                      # 'slope':slope,
                      # 'HNR':hnr,
                      # 'CPPs':CPPs,
                      # 'propE_500':propE_500,
                      # 'propE_1000':propE_1000,
                      # 'alphaRatio':alphaRatio,
                      # 'H1H2LTAS':H1H2LTAS,
                      # 'H1H2LTASmed':H1H2LTASmed,
                      # 'flatnessLTAS':flatnessLTAS,
                      # 'centroidLTAS':centroidLTAS,
                      # 'mfcc_a':mfcc_a,
                      # 'mfcc_ei':mfcc_ei,
                      # 'duration_noise':duration_n,
                      #'sfEstimate':ampEstimate
                      # 'pitchMax':f_0_max,
                      # 'pitchMean':f_0_mean,
                      # 'mfcc_ou':mfcc_ou,  
                      # 'meanMFCC_a':np.median(mfcc_a, axis=0),
                      # 'meanMFCC_ei':np.median(mfcc_ei, axis=0),
                      # 'meanMFCC_ou':np.median(mfcc_ou, axis=0),
                      # 'melSpec':melSpec,
                      'freqBand':freqBand,
                      'meanSpec':meanSpec,
                      'mfcc':final_mfcc,
                      'augmented':augmented,
                      'pitchShift':pitchChange,
                      # 'ampSD':ampSD,
                      # 'hammSD':hammSD,
                      # 'slopeSD':slopeSD,
                      # 'alphaSD':alphaSD,
                      # 'f1med_a':f1med_a,
                      # 'f2med_a':f2med_a,
                      # 'f1med_ei':f1med_ei,
                      # 'f2med_ei':f2med_ei,
                      # 'f1med_ou':f1med_ou,
                      # 'f2med_ou':f2med_ou,
                      # 'd_mfcc1':d_mfcc1,
                      # 'd_mfcc2':d_mfcc2,
                      # 'd_mfcc3':d_mfcc3,
                      # 'd_mfcc4':d_mfcc4,
                      # 'd_mfcc5':d_mfcc5,
                      # 'd_mfcc6':d_mfcc6,
                      # 'mfccSD':mfccSD,
                      # 'GNE':GNE,
                      # 'GNE1k2k':GNE1k2k,
                      # 'GNE1k3k':GNE1k3k,
                      # 'GNE2k3k':GNE2k3k,
                      # 'GNE1k4k':GNE1k4k,
                      # 'vibRate':vibRate_f0,
                      # 'vibExtent':vibExtent_f0,
                      # 'SNR':SNR,
                      'sr':samplerate
                      # 'dSPL':dSPL
                     
                     }
        #df = df.append(resultDict, ignore_index=True)
        df = pd.concat([df, pd.DataFrame.from_records([resultDict])])
        #df.loc[i,'sfEstimate'] = ampEstimate
        
        
        #vibrato_Frequency, vibratoStd, amplitudeCents, amplitudeCentsStd, vibratoPercentage = vibratoCalcTraining(highestPitch, samplerate, X_0, classifier, gender='weibl.', windowSecs=1)
        #tremorRate = tremorRateCalc(highestPitch, samplerate)
        # vibRate_f0, vibExtent_f0, vibRate_amp, vibExtent_amp = vibTremor(highestPitch, samplerate)
        
        #f_H1, f1_max, f_s_max, hammarberg, energyRatio_pabon, energyRatio_muerbe = timbre(highestPitch, samplerate, meanFreq)
        #H2_H1 = H2H1(highestPitch, samplerate, meanFreq)
        #H3_H1 = H3H1(highestPitch, samplerate, meanFreq)
        #cpps = cpp(highestPitch, samplerate)
        
        # df.loc[i, 'Vibrato-Rate (F_0)'] = vibRate_f0
        # df.loc[i, 'Vibrato-Umfang (F_0)'] = vibExtent_f0
        # df.loc[i, 'Vibrato-Rate (Amp)'] = vibRate_amp
        # df.loc[i, 'Vibrato-Umfang (Amp)'] = vibExtent_amp
        
        # df.loc[i, 'f1_max'] = f1_max
        # df.loc[i, 'f_s_max'] = f_s_max
        # df.loc[i, 'Hammarberg-Index'] = hammarberg
        # df.loc[i, 'EnergyRatio_2000'] = energyRatio_pabon
        # df.loc[i, 'EnergyRatio_2.7-3.6'] = energyRatio_muerbe
        
        #df.loc[i, 'H2/H1'] = H2_H1
        #df.loc[i, 'H3/H1'] = H3_H1
        #df.loc[i, 'CPPS'] = cpps
        
        #print('TremorRate: ' + str(tremorRate))
        #plt.close('all')
        # df.loc[i, 'vibRate'] = vibrato_Frequency
        # df.loc[i, 'vibPercent'] = vibratoPercentage
        # df.loc[i, 'vibRateStd'] = vibratoStd
        # df.loc[i, 'vibExtent'] = amplitudeCents
        # df.loc[i, 'vibExtentStd'] = amplitudeCentsStd
        
        #plt.plot(finalTrial)
        # visualizeResults(wavFilename, finalTrial, finalTrial, samplerate, gender=geschlecht)
        # vibratoFreq3 = vibratoCalc3(highestPitch, samplerate)
        # print('id: ' + str(df.loc[i].loc['id']) + 
             # ', Vibrato4: ' + str(round(vibrato_Frequency, 2))+ 
             # ' Hz, Vibrato Percentage: ' + str(round(vibratoPercentage, 2)) + 
             # ', Vibrato Std: ' + str(round(vibratoStd,2)) +
             # ', Vibrato Ampitude: ' + str(round(amplitudeCents,2)))
        # scipy.io.wavfile.write(wavFilename[:-4] + "_autodenoised.wav", samplerate, data)
        # if prompt != 'done':
            # prompt = input("Press Enter to continue, q to quit...")
            # if prompt == 'q':
                # break
        # sampleFilename = wavFilename[:-4] + '_sample' + '.wav'
        # write(sampleFilename, samplerate, finalTrial)
        plt.close('all')
df.to_pickle('melSpecAug20250723.pkl')
# vibDB = df