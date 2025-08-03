#PSEUDOCODE
#Load DataFrame
#Run through Code Sandbox for all Test2 files
import os
import glob
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from fastdtw import fastdtw as dtw  # assuming you're using fastdtw
from scipy.spatial.distance import euclidean

def pad_or_truncate_mfcc(mfcc, target_len):
    """Pad or truncate an MFCC matrix to match target_len time steps."""
    current_len, n_mfcc = mfcc.shape
    if current_len >= target_len:
        return mfcc[:target_len, :]
    else:
        pad_width = target_len - current_len
        return np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')




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
        
        ### Dynamic Time Warp Audio to Reference

        input_mfcc = librosa.feature.mfcc(y=data_shift, sr=sr, n_mfcc=13).T

        # --- DTW Alignment ---
        # Note: We transpose to shape (n_mfcc, time) for DTW to work correctly
        D, wp = dtw(ref_mfcc, input_mfcc, dist=euclidean)
        wp = np.array(wp[::-1])  # Reverse to forward path

        # Align input MFCC to reference using DTW path
        aligned_input_mfcc = input_mfcc[wp[:, 1], :]

        # --- Truncate or pad to match reference length ---
        final_mfcc = pad_or_truncate_mfcc(aligned_input_mfcc, ref_mfcc.shape[0])


        resultDict = {'id':wavFilename.split('\\')[-1].split('&')[0], 
                      'date':wavFilename.split('\\')[-1].split('&')[1],
                      'trialNum':wavFilename.split('\\')[-1][-5],
                 
                      'duration':duration,
                     
                      'mfcc':final_mfcc,
                      'augmented':augmented,
                      'pitchShift':pitchChange,
                     
                      'sr':samplerate

                     
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
df.to_pickle('melSpecAug20250803.pkl')
# vibDB = df