import pickle
from datetime import datetime
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
###How to find the ID number of the audio files:

    #Two objects:
        #IDNumbers.csv
#IDNumbers and names from audio files
IDNames = pd.read_csv('IDNumbers.csv')
        #studDB
#IDNames.columns: id, Name
#Might need to add years 
#Deidentified ID Numbers with time stamps
deIDMeta = pd.read_csv('DeidentifizierteMetadatenTimestamp.csv')
from datetime import datetime
deIDMeta['date'] = deIDMeta.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
#deIDMeta['date'].min()
    #2002
#Let's grab someone from 2002
#studDB.loc[0, 'Nachname']
#Following provides us with the excel ID numbers, the names, and intake year
with open('studDB_IDs.pkl', 'rb') as f:
    studDB_IDs = pd.read_pickle(f)
#For the audio DB we just want years with an audio recording
audioDB = studDB_IDs[studDB_IDs['Jahr'].astype(int) >= 2002]
#Make a df with id, name, and each date of recording
IDNamesDate = pd.merge(IDNames[['id', 'name']], deIDMeta[['date', 'id']], on='id')
IDNamesDate['year'] = IDNamesDate['date'].apply(lambda x: x.year)
#nachname = audioDB['Nachname'].iloc[0]
#vorname = audioDB['Vorname'].iloc[0]
#Need to convert audioDB's years to int
IDNamesDate = IDNamesDate[['id', 'name', 'year']].groupby('id').min().reset_index()
#IDNamesDate[IDNamesDate['name'].str[:12] == 'Böhme, Julia']
    #There are two Julia Böhme

#With the exception of the single duplicate, we'll search for both Vorname and Nachname
def findIDNum(nachname, vorname, jahr, IDNames):
    idSer = IDNames[((IDNames['name'].str.contains(nachname)) & 
                             (IDNames['name'].str.contains(vorname)) #&
                             #(IDNamesDate['year'] == jahr)
                             #(abs(IDNamesDate['year'] - jahr)<= 2)
                             ) ]['id']
    #If there is more than one response, we'll look for a recording within a year of intake
    if idSer.size > 1:
        print(idSer)
        idSer = IDNames[((IDNames['name'].str.contains(nachname)) & 
                             (IDNames['name'].str.contains(vorname)) &
                             #(IDNamesDate['year'] == jahr)
                             (abs(IDNames['year'] - jahr) <= 1)
                             ) ]['id']
    if idSer.size < 1: 
        idSer = np.nan
    try:
        idNum = int(idSer)
        
    except ValueError:
        idNum = np.nan
    return idNum
    return idNum
    
audioDB['audioID'] = audioDB.apply(lambda x: findIDNum(x.Nachname, x.Vorname, int(x.Jahr), IDNamesDate), axis=1)
audioDB['audioID'].count()
#293 Entries
#214 ID'd Anamnesen
#69 Anamnese since 2002 without audio recordings
#29 Audio files (243-214) without Anamnesen??? Marie said this shouldn't be possible
###Find missing Anamnesen:
#Rejoin audioDB to IDNames
missingNames = pd.merge(IDNames, audioDB['audioID'], how='left', left_on='id', right_on='audioID')
missingNames = missingNames[missingNames['audioID'].isna()]
###Just do it manually:
#missingNames.to_csv('missingAudioAnamnesen.csv')
foundNames = pd.read_csv('missingAudioAnamnesen.csv', dtype='str')
foundNames = foundNames.rename(columns={'audioID':'anamnesenID'})
mask = audioDB['audioID'].isna()
#audioDB.loc[mask, 'audioID'] = audioDB.loc[mask, 'Nummer'].apply(lambda x: str(foundNames[foundNames['anamnesenID'] == x]))
#audioDB[mask]['audioID'] = audioDB.loc[mask, 'Nummer'].apply(lambda x: str(foundNames[foundNames['anamnesenID'] == x]))
foundAudio = pd.merge(audioDB[audioDB['audioID'].isna()], foundNames[['id', 'anamnesenID']], how='inner', left_on='Nummer', right_on= 'anamnesenID')
foundAudio = foundAudio[['Jahr', 'Nummer', 'Nachname', 'Vorname', 'id']].rename(columns={'id':'audioID'})
###Fill in the IDs from the found files
def fillFoundIDs(anamnesenID, foundAudio):
    try:
        audioID = int(foundAudio[foundAudio['Nummer'] == anamnesenID]['audioID'])
    except TypeError:
        audioID = np.nan
    return audioID
audioDB.loc[mask,'audioID'] = audioDB.loc[mask, 'Nummer'].apply(lambda x: fillFoundIDs(x, foundAudio))


#Let's exclude jazzPop
with open('jazzPopDB.pkl', 'rb') as f:
    jazzPopDB = pd.read_pickle(f)
noJazz = pd.merge(audioDB, jazzPopDB[jazzPopDB['jazzPop'] == 0], on='Nummer')
#Removed 27 students, 29 total in jazzPopDBFvib
#I think we need to mke all non-NA values in audioID integers.
#First remove NA values
noJazz = noJazz[noJazz['audioID'].notna()]
#change type to int
noJazz['audioID'] = noJazz['audioID'].astype('int')
#vibDB = pd.read_csv('Vibrato3Hz.csv')
#vibDB = pd.read_csv('VibratoMFAmpFinal.csv')#'VibratoMF1SecWindow.csv')
#vibDB = pd.read_csv('TimbreSF.csv')
# with open('LTASMed.pkl', 'rb') as f:

# with open('vokalAusgleich796.pkl', 'rb') as f:
    # vibDB1 = pd.read_pickle(f)


# with open('avezzo784.pkl', 'rb') as f:
    # vibDB2 = pd.read_pickle(f)

# with open('dreiklang794.pkl', 'rb') as f:
    # vibDB3 = pd.read_pickle(f)
    
# vibDB = pd.concat([vibDB1, vibDB2, vibDB3])

# with open('PEVOC_f.pkl', 'rb') as f:

# First dSPL run
# d = pd.read_pickle('Just_dSPL.pkl')
# with open('normedRun.pkl', 'rb') as f:
    # vibDB = pd.read_pickle(f)
# vibDB = vibDB.drop('dSPL', axis=1)
# vibDB = pd.merge(vibDB, d[['id','date','dSPL']], on=['id','date'])

# vibDB0 = pd.read_pickle('melSpecOriginal20250723.pkl')
vibDB = pd.read_pickle('melSpecMFCC20250723.pkl')
# vibDB = pd.concat([vibDB0,vibDB1])
# vibDB = pd.read_pickle('PEVOC_f.pkl')

# vibDB = vibDB.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])
vibDBTest = vibDB#.drop_duplicates()
vibDBTest['id'] = vibDBTest['id'].astype('int64')
classVib = pd.merge(noJazz[['Jahr', 'Nummer', 'audioID']], 
                    #vibDB[['id', 'date', 'meanFreq',
                    vibDBTest,#[['id', 'date', 'meanFreq',
                        #'vibratoFreqMF', 'vibratoPercentageMF', 'vibratoStdMF', 
                        #   'vibFreqAmp', 'vibFreqAmpStd']],
                    left_on='audioID', right_on='id')
#Let's rename the vibrato calculations:
vibDict = {'vibratoFreqMF':'vibFreq',
           'vibratoPercentageMF':'vibPerc',
           'vibratoStdMF':'vibStd'}
#classVib = classVib.rename(columns=vibDict)
#classVib = classVib.drop(column='id')
#classVib = classVib.drop_duplicates()


#Now let's grab the control variables from the categorical DB
with open('nullDB.pkl', 'rb') as f:
    nullDB = pd.read_pickle(f)
# nullDB = pd.read_csv('nullDB.csv',index_col=False)
#nullDB = nullDB.drop(columns='Unnamed: 0')
#classVib['Nummer'] = classVib['Nummer'].astype('int64')
nullDB.loc[nullDB['vpnummer'] == '00415','geschlecht'] = 'weibl.'
df = pd.merge(classVib, nullDB, left_on='Nummer', right_on='vpnummer')

###Sidebar: Let's group the singers into E5, D5, E4 and D4
# df.loc[df['meanFreq'] > 623.3, 'meanPitch'] = 'E5'
# df.loc[((df['meanFreq'] < 623.3) & (df['meanFreq'] > 440)), 'meanPitch'] = 'D5'
# df.loc[((df['meanFreq'] < 440) & (df['meanFreq'] > 311.64)), 'meanPitch'] = 'E4'
# df.loc[df['meanFreq'] < 311.64, 'meanPitch'] = 'D4'

# with open('SNRdraft.pkl', 'rb') as f:
    # snr = pd.read_pickle(f)

# snr['id'] = snr['id'].astype(int)
# df = pd.merge(df, snr[['id', 'trialNum','date','SNR']], on=['id', 'trialNum','date'])


df['date'] = df['date'].apply(lambda x: x.replace('_', '-'))

###Ok, let's find the first recording.
#Code taken from metaData exploration
from datetime import datetime
from datetime import timedelta
#datetime.strptime(df.date[0], "%Y-%m-%d")
#df.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
#Didn't work. Some aren't in that format.
df['dateLen'] = df.date.apply(lambda x: len(x))
#Still have a couple of holdovers. 
    #One 2012-6-6 and one DATE - Kopie
#Convert 2012-6-6, remove Kopie
mask = df['dateLen'] == 8
df.loc[mask, 'date'] = '2012-06-06'
###remove Kopie
df['dateLen'] = df.date.apply(lambda x: len(x))
df = df[df['dateLen'] == 10].copy()
#Transition date strings to date format
#df1.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
    #Didn't work.
#Find inconsistently formatted dates
df['dateLen2'] = df.date.apply(lambda x: len(x.split('-')[0]))
#Only one entry with %d-%m-%Y
df[df['dateLen2'] == 2]
mask = df['dateLen2'] == 2
df.loc[mask, 'date'] = '2011-11-23'
df['date'] = df.date.apply(lambda x: datetime.strptime(x,  "%Y-%m-%d"))
df = df.drop(columns=['dateLen', 'dateLen2'])

df['minDate'] = df['id'].apply(lambda x: df[['id', 'date']].loc[df['id'] == x].groupby('id').min().iloc[0])
#Calculate difference:
df['dateDiff'] = df['date'] - df['minDate']

df['beginDate'] = df.studienbeginn.apply(lambda x: datetime(int(x),9,25))
def negativeDate(beginDate, minDate):
    if (beginDate - minDate) < timedelta(0):
        return beginDate
    else:
        return minDate

df['beginDate'] = df[['beginDate', 'minDate']].apply(lambda x: negativeDate(x.beginDate, x.minDate), axis=1)
df['beginDiff'] = df['date'] - df['beginDate']
df['beginDiff'] = df['beginDiff'].apply(lambda x: x.days)

###Let's restrict this to singers with extant recordings between 3-5 years after the begin of studies
df['yearFloor'] = df['beginDiff'].apply(lambda x: np.floor(x/365))
df['yearCeiling'] = df['beginDiff'].apply(lambda x: np.ceil(x/365))
df['Year'] = df['date'].apply(lambda x: x.year)
#Fix 0 yearCeiling
maskCeil = df['yearCeiling'] == 0
df.loc[maskCeil, 'yearCeiling'] = 1
# yearFloor
# 0.0    220
# 1.0    166
# 2.0    150
# 3.0    120
# 4.0     73
# 5.0     34
# 6.0     16
# 7.0      7
# 8.0      1



df2 = df[df['trialNum'] == '2'].copy()
df2['Stimmfach'] = 'Sop/Mezzo/Alt'
df2.loc[df2['pitchMed'] < 450, 'Stimmfach'] = 'Ten/Bar/Bass'
df1 = df[df['trialNum'] == '1'].copy()
df1['Stimmfach'] = 'Sop/Mezzo/Alt'
df1.loc[df1['pitchMed'] < 325, 'Stimmfach'] = 'Ten/Bar/Bass'

df6 = df[df['trialNum'] == '6'].copy()
df6['Stimmfach'] = 'Sop/Mezzo/Alt'
df6.loc[df6['pitchMed'] < 325, 'Stimmfach'] = 'Ten/Bar/Bass'

df5 = df[df['trialNum'] == '5'].copy()
df5['Stimmfach'] = 'Sop/Mezzo/Alt'
df5.loc[df5['pitchMed'] < 325, 'Stimmfach'] = 'Ten/Bar/Bass'
df5.loc[df5['pitchMed'] >= 325, 'Stimmfach'] = 'Sop/Mezzo/Alt'


#Let's remove duplicate recordings:
df2 = df2.groupby(['id','date','trialNum']).first().reset_index()
df1 = df1.groupby(['id','date','trialNum']).first().reset_index()
df6 = df6.groupby(['id','date','trialNum']).first().reset_index()
df5 = df5.groupby(['id','date','trialNum','pitchShift']).first().reset_index()

###Subset of initial recordings:
# df2['melMean'] = df2['melSpec'].apply(lambda x: np.mean(x))

def pad_or_truncate(mel, target_shape=(128, 44)):
    mel_bins, time_frames = mel.shape
    target_bins, target_time = target_shape

    # Truncate
    mel = mel[:, :target_time]

    # Pad if too short
    if mel.shape[1] < target_time:
        pad_width = target_time - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')

    return mel

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            pred_labels = preds.argmax(dim=1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0
                
###Let's combine the non-frequent options:
def soloChor(text):
    if type(text) != str:
        return np.nan
    if 'Chor' in text:
        return 'Chor'
    elif 'Solo' in text:
        return 'Solo'
    else:
        return np.nan
df5['SoloChor'] = df5['zulassung.hno'].apply(lambda x: soloChor(x))
def nasal(text):
    if type(text) != str:
        return np.nan
    if 'nasal' in text:
        return 'nasal'
    elif 'unauffällig' in text:
        return 'unauffällig'
    else:
        return np.nan
df5['nasal'] = df5['nasalitaet.sing'].apply(lambda x: nasal(x))
def medDark(text):
    if type(text) != str:
        return np.nan
    if 'hell' in text:
        return 'hell'
    if 'dunkel' in text:
        return 'mittel/dunkel'
    elif 'mittel' in text:
        return 'mittel/dunkel'
    else:
        return np.nan
        
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_multiclass_metrics(model, dataloader, device, n_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(n_classes))
    # cm[i, j] = true class i predicted as class j

    sensitivities = []
    specificities = []

    for c in range(n_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)

    accuracy = np.trace(cm) / np.sum(cm)

    return accuracy, np.mean(sensitivities), np.mean(specificities)
    
def variable_level_balance_summary(df):
    summary = []

    for col in df.columns:
        series = df[col]
        non_na_count = series.notna().sum()
        value_counts = series.value_counts(dropna=True)
        n_levels = len(value_counts)
        
        p = value_counts / value_counts.sum()
        entropy = -(p * np.log2(p)).sum() / np.log2(n_levels) if n_levels > 1 else 1.0
        
        summary.append((col, n_levels, entropy, non_na_count))
    
    summary_df = pd.DataFrame(summary, columns=['variable', 'n_levels', 'balance', 'n_non_na'])
    summary_df.set_index('variable', inplace=True)
    summary_df.sort_values('n_levels', inplace=True)

    return summary_df



df5['timbre'] = df5['timbre.sing'].apply(lambda x: medDark(x))
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix, classification_report


use_augmented_data = False  # Include augmented data in training?
use_kfold = True           # Toggle K-Fold vs. 80/20 validation
k_folds = 5               # Number of folds if using K-Fold

###What are the final variable names?
abschluss = ['studiendauer',
             'note',
             'abschlussart',
             'stimme.lage.ende',
             'stimme.typ.ende',
             'atemvolumen.ende',
             'ophno',
             'umfang.phys.ende',
             'umfang.mus.ende',
             'atemtyp.ende']

numList = ['alter','größe', 'gewicht', 'achse.nacken', 'achse.steiß', 'hals.umfang','atemvolumen.beginn','studiendauer']
textList = ['studienbeginn','beruf', 'op','vpnummer','erkrankung', 'beurteilung.hno',
'umfang.physiologisch.tief',
'umfang.physiologisch.hoch',
'umfang.musikalisch.tief',
'umfang.musikalisch.hoch',
'registeruebergang.tief.mittel',
'registeruebergang.mittel.hoch',
'registeruebergang.hoch.grenz',
'registerzwischenbruch.tief',
'registerzwischenbruch.mittel',
'registerzwischenbruch.hoch']
vibList = list(vibDB.columns)
timbreColumns = ['resonanz.sing','timbre','SoloChor','nasal']

df0 = df5[df5['beginDiff'] < 365].groupby('id').first().reset_index()
# test = variable_level_balance_summary(df0)
test = pd.read_pickle('nullDistributions.pkl')
# df0 = df5[((df5['beginDiff'] > 365*3) & (df5['beginDiff'] < 365*4))].groupby('id').first().reset_index()
use_mlp = True  # set to False to use CNN
final_accuracies = []
for j in range(2):
    if j == 1:
        use_mlp = False
    for i in df5.columns:
        k = 5
        # if ~(i in abschluss):
            # continue
        if i in numList:
            continue
        if i in textList:
            continue
        if i in abschluss:
            continue
        if i in vibList:
            continue
        if test.loc[i, 'n_levels'] < 2:
            continue
        if test.loc[i, 'n_levels'] == 2: 
            if test.loc[i, 'balance'] < 0.4:
                continue
            if test.loc[i, 'n_non_na'] < 100:
                print(i,', n = ',test.loc[i, 'n_non_na'])
                k = 3
        if test.loc[i, 'n_levels'] == 3: 
            if test.loc[i, 'balance'] < 0.6:
                continue
            if test.loc[i, 'n_non_na'] < 100:
                print(i,', n = ',test.loc[i, 'n_non_na'])
                k = 3
        if test.loc[i, 'n_levels'] > 3:
            continue
        final_accuracies = []
        df0 = df5[df5['beginDiff'] < 365].groupby('id').first().reset_index()
        df0 = df0[((df0['mfcc'].notna()) & (df0[i].notna()))]

        non_aug_mask = df0['augmented'] == False
        df_real = df0[non_aug_mask]

        mel_list = df_real['mfcc'].to_list()
        shapes = [mel.shape[1] for mel in mel_list]
        min_shape = np.min(shapes)

        mel_tensors = torch.tensor(np.stack(mel_list), dtype=torch.float32).unsqueeze(1)
        if use_mlp:
            input_shape = mel_tensors.shape  # (N, 1, 128, T)
            flat_dim = input_shape[2] * input_shape[3]  # 128 * min_shape
            mel_tensors = mel_tensors.view(-1, flat_dim)
            # aug_mel_tensors = aug_mel_tensors.view(-1, flat_dim)


        labels = df_real[i].to_list()
        for t in range(1):
            # random.shuffle(labels)
            le = LabelEncoder()
            label_tensors = torch.tensor(le.fit_transform(labels), dtype=torch.long)
            n_classes = len(np.unique(label_tensors.numpy()))
            if use_augmented_data:
                df_aug = df0[df0['augmented'] == True]
                if not df_aug.empty:
                    aug_mels = [pad_or_truncate(mel, (128, min_shape)) for mel in df_aug['melSpec']]
                    aug_labels = torch.tensor(le.transform(df_aug[i]), dtype=torch.long)
                    aug_mel_tensors = torch.tensor(np.stack(aug_mels), dtype=torch.float32).unsqueeze(1)
                else:
                    aug_mel_tensors = None
                    aug_labels = None
            else:
                aug_mel_tensors = None
                aug_labels = None

            accuracies = []
            sensitivities = []
            specificities = []

            # === Main Splitting Logic ===
            if use_kfold:
                splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                splits = splitter.split(mel_tensors, label_tensors)
            else:
                train_idx, test_idx = train_test_split(
                    np.arange(len(label_tensors)),
                    stratify=label_tensors,
                    test_size=0.2,
                    random_state=42,
                )
                splits = [(train_idx, test_idx)]  # Single split

            for fold, (train_idx, test_idx) in enumerate(splits):
                # print(f"\n--- Fold {fold + 1} ---" if use_kfold else "\n--- 80/20 Split ---")

                X_train_real, y_train_real = mel_tensors[train_idx], label_tensors[train_idx]
                X_test, y_test = mel_tensors[test_idx], label_tensors[test_idx]

                if use_augmented_data and aug_mel_tensors is not None:
                    X_train = torch.cat([X_train_real, aug_mel_tensors])
                    y_train = torch.cat([y_train_real, aug_labels])
                else:
                    X_train = X_train_real
                    y_train = y_train_real

                train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
                test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
                


                if use_mlp:
                    model = nn.Sequential(
                        nn.Linear(flat_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, n_classes)
                    )
                else:
                    model = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(32, n_classes)
                    )


                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(10):
                    model.train()
                    total_loss = 0
                    for xb, yb in train_dl:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad()
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()

                # acc = evaluate_accuracy(model, test_dl)
                acc, sens, spec = evaluate_multiclass_metrics(model, test_dl, device, n_classes)

                
                accuracies.append(acc)
                sensitivities.append(sens)
                specificities.append(spec)

            # Take average over k folds
            mean_acc = np.mean(accuracies)
            mean_sens = np.mean(sensitivities)
            mean_spec = np.mean(specificities)

            final_accuracies.append(mean_acc)

    # Save to test DataFrame
        if use_mlp:
            test.loc[i, 'mlp_accuracy'] = mean_acc
            test.loc[i, 'mlp_std'] = np.std(accuracies)
            test.loc[i, 'mlp_sensitivity'] = mean_sens
            test.loc[i, 'mlp_specificity'] = mean_spec
            # test.loc[i, 'mlp_null'] = np.mean(final_accuracies)
            # test.loc[i, 'mlp_null_std'] = np.std(final_accuracies)
        else:
            test.loc[i, 'cnn_accuracy'] = mean_acc
            test.loc[i, 'cnn_std'] = np.std(accuracies)
            test.loc[i, 'cnn_sensitivity'] = mean_sens
            test.loc[i, 'cnn_specificity'] = mean_spec
            # test.loc[i, 'cnn_null'] = np.mean(final_accuracies)
            # test.loc[i, 'cnn_null_std'] = np.std(final_accuracies)
        print(f"{i}: Accuracy = {mean_acc:.4f} ± {np.std(accuracies):.4f} | Sensitivity = {mean_sens:.4f} | Specificity = {mean_spec:.4f}")
        print('Number of Classes: ',n_classes)
        # print(i,': ',t,': ',f"\nMean Accuracy: {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")

from scipy.stats import norm
test = test[test['mlp_null'].notna()]
test["p_value_mlp"] = test.apply(
    lambda row: 1 - norm.cdf(row["mlp_accuracy"], loc=row["mlp_null"], scale=row["mlp_null_std"])
    if row["mlp_null_std"] > 0 else 1.0,
    axis=1
)

from statsmodels.stats.multitest import multipletests

def apply_bh_correction(df, p_col="p_value", alpha=0.05, new_col="p_value_bh"):
    """
    Apply Benjamini-Hochberg FDR correction to a column of p-values in a DataFrame.

    Parameters:
    - df: pandas DataFrame containing p-values
    - p_col: column name with raw p-values
    - alpha: desired FDR level (default 0.05)
    - new_col: name for new column with adjusted p-values

    Returns:
    - Modified DataFrame with BH-adjusted p-values in new_col
    """
    pvals = df[p_col].values
    _, corrected_pvals, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    df[new_col] = corrected_pvals
    return df

test = apply_bh_correction(test, p_col="p_value_mlp", new_col="p_value_bh")



for i in ['resonanz.sing']:  # Add more label columns if needed
    final_accuracies = []

    df0 = df5[df5['beginDiff'] < 365].groupby('id').first().reset_index()
    df0 = df0[((df0['mfcc'].notna()) & (df0[i].notna()))]

    non_aug_mask = df0['augmented'] == False
    df_real = df0[non_aug_mask]

    mel_list = df_real['mfcc'].to_list()
    shapes = [mel.shape[1] for mel in mel_list]
    min_shape = np.min(shapes)
    mel_tensors = torch.tensor(np.stack(mel_list), dtype=torch.float32).unsqueeze(1)

    labels = df_real[i].to_list()
    random.shuffle(labels)

    for t in range(1):
        le = LabelEncoder()
        label_tensors = torch.tensor(le.fit_transform(labels), dtype=torch.long)
        n_classes = len(le.classes_)  # ✅ get number of unique classes

        if use_augmented_data:
            df_aug = df0[df0['augmented'] == True]
            if not df_aug.empty:
                aug_mels = [pad_or_truncate(mel, (128, min_shape)) for mel in df_aug['melSpec']]
                aug_labels = torch.tensor(le.transform(df_aug[i]), dtype=torch.long)  # ✅ match label column
                aug_mel_tensors = torch.tensor(np.stack(aug_mels), dtype=torch.float32).unsqueeze(1)
            else:
                aug_mel_tensors = None
                aug_labels = None
        else:
            aug_mel_tensors = None
            aug_labels = None

        accuracies = []

        # === Main Splitting Logic ===
        if use_kfold:
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            splits = splitter.split(mel_tensors, label_tensors)
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(label_tensors)),
                stratify=label_tensors,
                test_size=0.2,
                random_state=42,
            )
            splits = [(train_idx, test_idx)]  # Single split

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train_real, y_train_real = mel_tensors[train_idx], label_tensors[train_idx]
            X_test, y_test = mel_tensors[test_idx], label_tensors[test_idx]

            if use_augmented_data and aug_mel_tensors is not None:
                X_train = torch.cat([X_train_real, aug_mel_tensors])
                y_train = torch.cat([y_train_real, aug_labels])
            else:
                X_train = X_train_real
                y_train = y_train_real

            train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
            test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

            # ✅ Model updated to output correct number of classes
            cnn_model = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, n_classes)
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cnn_model.to(device)
            optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(10):
                cnn_model.train()
                total_loss = 0
                for xb, yb in train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    preds = cnn_model(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            acc = evaluate_accuracy(cnn_model, test_dl)
            accuracies.append(acc)

        final_accuracies.append(np.mean(accuracies))
        # print(i, ': ', f"\nMean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(t,': ',f"\nMean Accuracy: {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")


###Ful set timbre
# Set this flag to switch off k-fold and use full training set
df0 = df5[df5['beginDiff'] < 365].groupby('id').first().reset_index()
# Main dataset
df_main = df0[df0['mfcc'].notna() & df0['timbre'].notna()]
mel_list = df_main['mfcc'].to_list()
labels = df_main['timbre'].to_list()

# Encode labels
le = LabelEncoder()
label_tensors = torch.tensor(le.fit_transform(labels), dtype=torch.long)

# Shape padding/truncation
shapes = [mel.shape[1] for mel in mel_list]
min_shape = np.min(shapes)
fixed_mels = [pad_or_truncate(mel, (128, min_shape)) for mel in mel_list]
mel_tensors = torch.tensor(np.stack(fixed_mels), dtype=torch.float32).unsqueeze(1)

# Final training set
X_train = mel_tensors
y_train = label_tensors
train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

cnn_model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(32, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    cnn_model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = cnn_model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# acc = evaluate_accuracy(cnn_model, test_dl)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Put model in evaluation mode
cnn_model.eval()

# Collect all predictions
all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in DataLoader(TensorDataset(X_train, y_train), batch_size=32):
        xb = xb.to(device)
        preds = cnn_model(xb)
        pred_labels = preds.argmax(dim=1).cpu().numpy()
        all_preds.extend(pred_labels)
        all_true.extend(yb.numpy())

# Generate confusion matrix
cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix – Timbre Model (Full Set)")
plt.tight_layout()
plt.show()


import shap

# Sample background and test examples
background = next(iter(train_dl))[0][:10].to(device)
test_samples = next(iter(test_dl))[0][:5].to(device)

# Explain
explainer = shap.GradientExplainer(cnn_model, background)
shap_values = explainer.shap_values(test_samples)

import matplotlib.pyplot as plt
import numpy as np
import librosa

sample_idx = 0
sr = 16000
n_mels = shap_values.shape[2]
fmin = 1000#0
fmax = 5000#sr / 2

# Convert mel bin indices to real frequencies
mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

# Pick 10 frequencies to show as ticks
yticks = np.linspace(0, n_mels - 1, 10).astype(int)
ytick_labels = [f"{mel_frequencies[i]/1000:.2f} kHz" for i in yticks]

# Number of output classes
num_classes = shap_values.shape[-1]

# Plot SHAP heatmap per class
for class_idx in range(num_classes):
    # [sample, channel=0, mel, time, class]
    heatmap = shap_values[sample_idx, 0, :, :, class_idx]

    plt.figure(figsize=(10, 4))
    im = plt.imshow(heatmap, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar(im, label='SHAP Value')
    # If label encoder exists, use actual class name
    label_name = le.classes_[class_idx] if 'le' in globals() else f"Class {class_idx}"
    plt.title(f"SHAP Explanation - Class {class_idx} ({label_name})")
    plt.xlabel("Time")
    plt.ylabel("Frequency (kHz)")
    plt.yticks(yticks, ytick_labels)
    plt.tight_layout()
    plt.show()
    
    
###MFCCs
# Sample background and test examples
###TRAINTESTSPLIT
# background = next(iter(train_dl))[0][:10].to(device)
# test_samples = next(iter(test_dl))[0][:5].to(device)

### Explain
# explainer = shap.GradientExplainer(cnn_model, background)
# shap_values = explainer.shap_values(test_samples)

###FULL DATASET
# Assumes X_train is your full training tensor of shape [N, 1, MFCC, Time]
# and cnn_model is already trained

# Sample background and test samples directly from X_train
background = X_train[:10].to(device)     # 10 background examples
test_samples = X_train[10:15].to(device) # 5 test examples to explain

# Create SHAP GradientExplainer
explainer = shap.GradientExplainer(cnn_model, background)
shap_values = explainer.shap_values(test_samples)


# === MFCC Parameters ===
sample_idx = 0
n_mfcc = shap_values.shape[2]  # Typically 13

# Y-axis: MFCC coefficient index
yticks = np.arange(13)
# ytick_labels = [f"MFCC {i}" for i in yticks]

# Number of output classes
num_classes = shap_values.shape[-1]

# === Plot SHAP heatmap per class ===
for class_idx in range(num_classes):
    # Shape: [sample, channel, mfcc, time, class]
    heatmap = shap_values[sample_idx, 0, :, :, class_idx].T

    plt.figure(figsize=(10, 4))
    im = plt.imshow(heatmap, aspect='auto', origin='lower', cmap='coolwarm', vmin=-np.max(np.abs(heatmap)), vmax=np.max(np.abs(heatmap)))
    plt.colorbar(im, label='SHAP Value')
    
    label_name = le.classes_[class_idx] if 'le' in globals() else f"Class {class_idx}"
    plt.title(f"SHAP Explanation - Class {class_idx} ({label_name})")
    plt.xlabel("Time Frame")
    plt.ylabel("MFCC Coefficient")
    plt.yticks(yticks, fontsize=8, rotation=0) 
import matplotlib.pyplot as plt

# Use SHAP values for a specific class (e.g., class 0)
shap_class = shap_values[0]  # shape: [samples, 1, n_mfcc, time]

# Remove singleton channel dimension: shape → [samples, n_mfcc, time]
shap_class = shap_class[:, 0]

# Compute mean absolute SHAP per MFCC coefficient (averaging over time and samples)
mean_abs_shap = np.mean(np.abs(shap_class), axis=(0, 2))  # shape: [n_mfcc]

# Bar plot # Keep horizontal for MFCCs
    plt.tight_layout(pad=2.0) 
    plt.show()

                     # Add spacing to reduce overlap
import numpy as np
plt.figure(figsize=(10, 4))
plt.bar(np.arange(len(mean_abs_shap)), mean_abs_shap)
plt.xlabel("MFCC Coefficient")
plt.ylabel("Mean |SHAP value|")
plt.title("Global SHAP Importance by MFCC Coefficient")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

n_mfcc_to_plot = 3
samples_per_class = 3
classes_to_plot = le.classes_  # or subset like ['bright', 'dark']

mfccs = df0['mfcc'].tolist()
labels = df0['timbre'].tolist()

for cls in classes_to_plot:
    fig, axs = plt.subplots(n_mfcc_to_plot, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Timbre Class: {cls}", fontsize=14)

    class_idxs = [i for i, l in enumerate(labels) if l == cls]
    class_samples = [mfccs[i] for i in class_idxs[:samples_per_class]]

    for mfcc_idx in range(n_mfcc_to_plot):
        for sample in class_samples:
            sample = sample.T  # <-- transpose to shape (n_mfcc, time)
            axs[mfcc_idx].plot(sample[mfcc_idx], alpha=0.7)
        axs[mfcc_idx].set_ylabel(f"MFCC {mfcc_idx}")
        axs[mfcc_idx].grid(True)

    axs[-1].set_xlabel("Time Frame")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define which MFCCs to plot (1 through 5)
mfcc_indices = range(1, 6)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 10), sharex=True)

for subplot_idx, mfcc_idx in enumerate(mfcc_indices):
    ax = axes[subplot_idx]
    for label, mfcc_array in mfccs_by_label.items():
        # Shape: (n_samples, n_mfcc, n_time)
        mean = mfcc_array.mean(axis=0)   # (n_mfcc, n_time)
        std = mfcc_array.std(axis=0)

        time = np.arange(mean.shape[1])
        ax.plot(time, mean[mfcc_idx], label=f"{label}")
        ax.fill_between(time, mean[mfcc_idx] - std[mfcc_idx], mean[mfcc_idx] + std[mfcc_idx], alpha=0.2)

    ax.set_ylabel(f"MFCC {mfcc_idx}")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

axes[-1].set_xlabel("Time Frame")
fig.suptitle("MFCCs 1–5: Mean ± Std by Label", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

###Hypothesis: Singers darken their timbre over the course of studies:
year4 = df5[((df5['beginDiff'] > 365*3) & (df5['beginDiff'] < 365*4))]
new_tensor = torch.tensor(np.stack(df0['mfcc']), dtype=torch.float32).unsqueeze(1)  # sha
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_tensor = new_tensor.to(device)
cnn_model.eval()
with torch.no_grad():
    outputs = cnn_model(new_tensor)  # shape: [batch_size, num_classes]
    predicted_indices = outputs.argmax(dim=1).cpu().numpy()
    predicted_labels = le.inverse_transform(predicted_indices)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming:
# - labels: original ground truth labels from year one
# - predicted_labels: predicted labels on new data

# Convert to Series
true_series = pd.Series(labels, name='True')
pred_series = pd.Series(predicted_labels, name='Predicted')

# Normalize counts
true_dist = true_series.value_counts(normalize=True).sort_index()
pred_dist = pred_series.value_counts(normalize=True).sort_index()

# Combine into DataFrame
dist_df = pd.DataFrame({'Original Labels (Year 1)': true_dist,
                        'Predicted Labels (Year 4)': pred_dist}).fillna(0)

# Plot
dist_df.plot(kind='bar', figsize=(8, 5), width=0.7)
plt.title("Label Proportions: Original vs. Predicted")
plt.ylabel("Proportion")
plt.xlabel("Class Label")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

from scipy.stats import chi2_contingency

# Create a contingency table (counts, not proportions)
labels_ = sorted(set(labels) | set(predicted_labels))
contingency = pd.DataFrame(index=labels)
contingency['True'] = true_series.value_counts().reindex(labels_, fill_value=0)
contingency['Predicted'] = pred_series.value_counts().reindex(labels_, fill_value=0)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency.T)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p:.4f}")

import pandas as pd

# Step 1: Convert predicted_labels tensor to NumPy array
predicted_np = predicted_labels.cpu().numpy() if predicted_labels.is_cuda else predicted_labels.numpy()

# Step 2 (Optional): Convert class indices to label names
predicted_str_labels = le.inverse_transform(predicted_np)

# Step 3: Assign to your DataFrame
year4['predicted_timbre'] = predicted_labels


###Repeat for mean spectrogram
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
import numpy as np
import librosa
  
#Mask for PEVOC
df1 = df1[(df1['Year'] >= 2008) & (df1['Year'] <= 2018)]
df2 = df2[(df2['Year'] >= 2008) & (df2['Year'] <= 2018)]
df6 = df6[(df6['Year'] >= 2008) & (df6['Year'] <= 2018)]


###Let's combine the non-frequent options:
def soloChor(text):
    if type(text) != str:
        return np.nan
    if 'Chor' in text:
        return 'Chor'
    elif 'Solo' in text:
        return 'Solo'
    else:
        return np.nan
df2['SoloChor'] = df2['zulassung.hno'].apply(lambda x: soloChor(x))
def nasal(text):
    if type(text) != str:
        return np.nan
    if 'nasal' in text:
        return 'nasal'
    elif 'unauffällig' in text:
        return 'unauffällig'
    else:
        return np.nan
df2['nasal'] = df2['nasalitaet.sing'].apply(lambda x: nasal(x))
varList = ['timbre.sing', 'SoloChor', 'resonanz.sing', 'nasal']
freqPairs = [[-np.inf,311],[311,350],[554,622],[622,np.inf]]
for low, high in freqPairs:
    # print(str(low),' ',str(high))
    for v in varList:
        df0 = df2[df2['beginDiff'] < 365].groupby('id').first().reset_index()
        # df0 = df0[((df0['melSpec'].notna()) & (df0[v].notna()))]
        df0 = df0[((df0['melSpec'].notna()) & (df0[v].notna()))]
        df0['melMean'] = df0['melSpec'].apply(lambda x: np.mean(x, axis=1))
        df0 = df0[(( low < df0['pitchMed']) & (df0['pitchMed'] < high))] # Excluding mezzos
        #Should we remove "dark singers (n=6)?
        # df0 = df0[((df0[v] != 'dunkel') & (df0['Stimmfach'] == 'Sop/Mezzo/Alt') & (df0['stimme.lage.beginn'] == 'hoch'))]
        df0 = df0[df0[v] != 'Pädagogik']
        ### Remove all versions that have one label:
        # Get label counts
        label_counts = df0[v].value_counts()

        # Keep only labels with at least 2 samples
        valid_labels = label_counts[label_counts >= 10].index
        
        if len(np.unique(valid_labels)) < 2:
            continue
        
        # Filter dataframe
        df0 = df0[df0[v].isin(valid_labels)]
            




        # Each row in 'mel_spectrogram' is a 2D numpy array

        labels = df0[v].to_list()
        # Convert to tensors
        mel_array = np.stack(df0['melMean'].values)  # shape: [N, 128]
        # Change tensor shape to: [batch_size, channels=1, length=128]
        mel_tensors = torch.tensor(mel_array, dtype=torch.float32).unsqueeze(1)  # [N, 1, 128]

        # Encode labels (e.g., Bright=1, Medium=0)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        num_classes = len(np.unique(labels))
        label_tensors = torch.tensor(labels, dtype=torch.long)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(mel_tensors, label_tensors, test_size=0.2, stratify=label_tensors, random_state=42)

        # Create DataLoaders
        train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        test_dl = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


        # Model using Conv1d instead of Conv2d
        cnn_model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_model.to(device)

        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            cnn_model.train()
            total_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)

                preds = cnn_model(xb)
                loss = criterion(preds, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        
        
        y_true = []
        y_pred = []

        cnn_model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(device)
                outputs = cnn_model(xb)              # Raw logits
                preds = torch.argmax(outputs, dim=1) # Predicted class

        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        # y_true: actual labels, y_pred: model predictions
        print(v , ': ', str(low),'-',str(high)) 
        print(classification_report(y_true, y_pred, target_names=le.classes_))
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        
        # Sample background and test examples
        background = next(iter(train_dl))[0][:10].to(device)
        test_samples = next(iter(test_dl))[0][:5].to(device)

        # Explain
        explainer = shap.GradientExplainer(cnn_model, background)
        shap_values = explainer.shap_values(test_samples)



        sample_idx = 0
        sr = 16000
        n_mels = shap_values.shape[2]
        fmin = 0
        fmax = sr / 2

        # Convert mel bin indices to real frequencies
        mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

        # Pick 10 frequencies to show as ticks
        yticks = np.linspace(0, n_mels - 1, 10).astype(int)
        ytick_labels = [f"{mel_frequencies[i]/1000:.2f} kHz" for i in yticks]

        # Number of output classes
        num_classes = shap_values.shape[-1]

        # Plot SHAP heatmap per class
        plt.figure(figsize=(10, 3))
        for class_idx in range(num_classes):
            heatmap = shap_values[class_idx][sample_idx]  # shape: (128,) or (128, 2)

            # If shape is (128, 2), average across components
            if heatmap.ndim == 2:
                heatmap = heatmap.mean(axis=1)

            
            plt.plot(mel_frequencies, heatmap, label=le.classes_[class_idx])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("SHAP Value")
            plt.title(f"SHAP Explanation: " + v.replace('.','_') + ', ' + str(low) + '-' + str(high))
            plt.grid(True)
            plt.tight_layout()
        plt.legend()
        # plt.xlim(600, 1400)
        # plt.show()
        plt.savefig(v.replace('.','_') + str(low) + '-' + str(high) + '.png')






#Mask for PEVOC
df1 = df1[(df1['Year'] >= 2008) & (df1['Year'] <= 2018)]
df2 = df2[(df2['Year'] >= 2008) & (df2['Year'] <= 2018)]
df6 = df6[(df6['Year'] >= 2008) & (df6['Year'] <= 2018)]

# df1 = df1[(df1['Year'] < 2008)]# & (df1['Year'] <= 2018)]
# df2 = df2[(df2['Year'] < 2008)]# & (df2['Year'] <= 2018)]
# df6 = df6[(df6['Year'] < 2008)]# & (df6['Year'] <= 2018)]
# df1 = df1[(df1['Year'] > 2018)]
# df2 = df2[(df2['Year'] > 2018)]
# df6 = df6[(df6['Year'] > 2018)]

def fachSplit(stimmfach, lage):
    if ((stimmfach == 'Sop/Mezzo/Alt') & (lage == 'hoch')):
        return 'Soprano'
    if ((stimmfach == 'Sop/Mezzo/Alt') & (lage == 'mittel')):
        # return 'Mezzo-Soprano'
        return 'Mezzo-Soprano/Contralto'
    if ((stimmfach == 'Sop/Mezzo/Alt') & (lage == 'tief')):
        # return 'Alto' 
    if ((stimmfach == 'Ten/Bar/Bass') & (lage == 'hoch')):
        return 'Mezzo-Soprano/Contralto'        
        return 'Tenor'
    if ((stimmfach == 'Ten/Bar/Bass') & (lage == 'mittel')):
        # return 'Baritone'
        return 'Baritone/Bass'
    if ((stimmfach == 'Ten/Bar/Bass') & (lage == 'tief')):
        # return 'Bass'
        return 'Baritone/Bass'
    else:
        return np.nan
        
df1['highmedlow'] = df1['stimme.lage.beginn']
df2['highmedlow'] = df2['stimme.lage.beginn']
df6['highmedlow'] = df6['stimme.lage.beginn']
df1['Voice Type'] = df1.apply(lambda x: fachSplit(x.Stimmfach, x.highmedlow), axis=1)
df2['Voice Type'] = df2.apply(lambda x: fachSplit(x.Stimmfach, x.highmedlow), axis=1)
df6['Voice Type'] = df6.apply(lambda x: fachSplit(x.Stimmfach, x.highmedlow), axis=1)

dfList = [df1,df2,df6]
df_f = [df1,df2,df6]
strList = ['1','2','6']
i=0

for d in dfList:
    d['Voice Type'] = d.apply(lambda x: fachSplit(x.Stimmfach, x.highmedlow), axis=1)
    d['yearDiff'] = d['beginDiff']/365
    testDF = d[d['yearDiff'] < 4]
    testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
    mask2 = testDF2['Jahr'] > 1
    testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id')
    # testDF2_f[['id', 'yearDiff', 'Stimmfach', 'H1H2LTAS', 'alphaRatio', 'CPPs', 'SNR']].to_csv('SNR' + strList[i] + '.csv')
    testDF2_f[['id', 'yearDiff', 'Stimmfach', 'H1H2LTAS', 'alphaRatio', 'CPPs', 'SNR', 'Jahr','alter', 'geschlecht', 'Voice Type','dSPL']].to_csv('Klang' + strList[i] + '_JASA_Final.csv')
    df_f[i] = testDF2_f
    i += 1

dfList = [df1,df2,df6]
validity = ['timbre.sing']#, 'qualitaet.sing', 'zulassung.hno','resonanz.sing', 'nasalitaet.sing']
measures = ['alphaRatio','H1H2LTAS']
strList = ['1','2','6']
nameDict = {'Sop/Mezzo/Alt':'Treble', 'Ten/Bar/Bass':'Non-Treble','hell':'Bright','mittel':'Medium'}
k=0
##Validity Checking
for d in dfList:
    d['yearDiff'] = d['beginDiff']/365
    testDF = d[d['yearDiff'] < 1]
    # print(strList[k] + 'F')
    # print(testDF[testDF['Stimmfach'] == 'Sop/Mezzo/Alt'].groupby(['timbre.sing','resonanz.sing'])['H1H2LTAS'].describe())
    # print(strList[k] + 'M')
    # print(testDF[testDF['Stimmfach'] == 'Ten/Bar/Bass'].groupby(['timbre.sing','nasalitaet.sing'])['alphaRatio'].describe())
    for i in validity:
        for j in measures:
            plt.close('all')
            testDF_hell = testDF[((testDF[i] != 'dunkel') & (testDF[i].notna()))].rename(columns={'Stimmfach':'Voice Group'})
            testDF_hell['Voice Group'] = testDF_hell['Voice Group'].apply(lambda x: nameDict[x])
            testDF_hell['timbre.sing'] = testDF_hell['timbre.sing'].apply(lambda x: nameDict[x])
            # sns.stripplot(data=testDF_hell, x=i, y=j, hue="Stimmfach")
            sns.violinplot(data=testDF_hell, x=i, y=j, hue="Voice Group", order=['Bright','Medium'])
            text = j+i+strList[k]
            plt.xlabel("Timbre Rating Beginning of Studies")
            plt.ylabel("Alpha Ratio (dB)")
            # plt.show()
            plt.savefig(text.replace('.','_')+'violinplot.pdf')
    k += 1
            

validity = ['toneinsatz']
measures = ['CPPs']#, 'GNE']
strList = ['1','2','6']
k=0
nameDict2 = {'behaucht':'Breathy', 'hart':'Hard', 'weich':'Balanced'}
##Validity Checking
for d in dfList:
    d['yearDiff'] = d['beginDiff']/365
    testDF = d[d['yearDiff'] < 1]
    testDF = testDF[((testDF['toneinsatz'].notna()) & (testDF['geschlecht'] == 'weibl.') & (testDF['toneinsatz'] != 'hart'))]
    testDF['toneinsatz'] = testDF['toneinsatz'].apply(lambda x: nameDict2[x])
    for i in validity:
        for j in measures:
            plt.close('all')
            sns.stripplot(data=testDF, x=i, y=j, color='black')#, hue="Stimmfach")
            sns.violinplot(data=testDF, x=i, y=j, inner='quart')#, hue="Stimmfach")
            text = j+i+strList[k]
            plt.xlabel('Onset Rating Beginning of Studies')
            plt.ylabel('CPPs (dB)')
            plt.savefig(text.replace('.','_')+'stripplot.pdf')
    k += 1
          
###Heatmaps
# Load the example flights dataset and convert to long-form
# timbreNasal = sns.load_dataset(df6[df6['Stimmfach'] == 'Ten/Bar/Bass'])
nonTreble = df6[df6['Stimmfach'] == 'Ten/Bar/Bass']
timNasHeat = (
    nonTreble
    .pivot(index="timbre.sing", columns="nasalitaet.sing", values="alphaRatio")
)

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(timNasHeat, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.show()        
        
### Resonanz/Timber KDE F
validity = ['resonanz.sing', 'nasalitaet.sing', 'timbre.sing', 'qualitaet.sing', 'zulassung.hno']
stimmfach = ['Sop/Mezzo/Alt']#,'Ten/Bar/Bass']
import seaborn as sns
for i in validity:
    for j in stimmfach:
        plt.close('all')
        #figure()
        mask = ((df2['beginDiff'] <365) &(df2['Stimmfach'] == j))
        pairs = df2.loc[mask,['H1H2LTAS','CPPs',i]]
        # pairs = pairs.rename(columns={'vibrato.stabilitaet':'Initial Vibrato Rating'})
        # transDict = {'stabil':'stable', 'labil':'unstable', 'ohne':'non-vibrato'}
        mask = pairs[i].notna()
        # pairs.loc[mask, 'Initial Vibrato Rating'] = pairs.loc[mask, 'Initial Vibrato Rating'].apply(lambda x: transDict[x])
        #maskThresh = pairs['vibPerc'] > 90
        #pairs.loc[maskThresh, 'vibPercSize'] = '>90'
        #maskSub = pairs['vibPerc'] < 90
        #pairs.loc[mask, 'vibPercSize'] = '<90' 
        # colors = {'stable':'darkblue', 'unstable':'blue', 'non-vibrato':'skyblue'}
        # order = ['stable', 'unstable', 'non-vibrato']
        #sizesDict = {'>90':1, '<90':0.25}
        sns.scatterplot(x='H1H2LTAS', y='CPPs' ,data=pairs,
                        palette='colorblind',
                        #palette=colors,
                        hue=i, #hue_order=order, 
                        # size='vibPerc'
                        )#, sizes=(0.25,1))

        g = sns.kdeplot(data=pairs, x="H1H2LTAS", y="CPPs", 
                        hue=i, 
                        #hue_order=order, 
                        levels=1, 
                        palette='colorblind')
                        #palette=colors)
        string = j.replace('/','-') + i.replace('.','_') + 'KDE.pdf'
        plt.savefig(string)
        
)



# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))



g.set(xlim=(2.5,8.5),
      ylim=(0,45),
      title='Initial Vibrato Rate and Extent',
      xlabel='Vibrato Frequency (Hz)',
      ylabel='Vibrato Extent (Cents)') 



#Combined DF where all data were recorded on same date:
mergeDF = pd.merge(pd.merge(df1,df2, on=['id','date']),df6, on=['id','date'])
#Let's get all recordings under four years
fourDF = mergeDF[mergeDF['beginDiff'] < 365*4]
#We have some null values from corrupted audio files:
corruptDrop = fourDF[fourDF['CPPs'].notna()]
#Find all recordings that have more than one recording in this time frame:
combineDF = corruptDrop.groupby('id')['Jahr'].count().reset_index()
maskMultiple = combineDF['Jahr'] > 1
df1 = pd.merge(df1, combineDF[maskMultiple]['id'], on='id').reset_index(drop=True)
df2 = pd.merge(df2, combineDF[maskMultiple]['id'], on='id').reset_index(drop=True)
df6 = pd.merge(df6, combineDF[maskMultiple]['id'], on='id').reset_index(drop=True)


#Plot changes in H1H2LTAS:
plt.close('all')
#Let's get all recordings under four years
testDF = df1[df1['beginDiff'] < 365*4]
#We have some null values from corrupted audio files:
testDF = testDF[testDF['CPPs'].notna()]
#Find all recordings that have more than one recording in this time frame:
testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
mask2 = testDF2['Jahr'] > 1
#Remerge with testDF
testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id').reset_index(drop=True)
###What if we just use CPPsvalues after say 2007
mask07 = testDF2_f[testDF2_f['date'] > datetime(2007,1,1)]
maskM = testDF2_f['pitchMed'] < 325
g = sns.lmplot(x='beginDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f)
plt.title('H1H2LTAS im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
g.tight_layout()
# plt.savefig('H1H2LTAS.png')
model = smf.mixedlm("H1H2LTAS ~ beginDiff*C(Stimmfach)",
                    testDF2_f,
                    groups= "id"
                    ).fit()
model.summary()

sns.lmplot(x='beginDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[~maskM])
plt.title('H1H2LTAS im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('H1H2LTAS.png')
model = smf.mixedlm("H1H2LTAS ~ beginDiff*C(Stimmfach)",
                    testDF2_f[~maskM],
                    groups= "id"
                    ).fit()
model.summary()


sns.lmplot(x='beginDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskM])
plt.title('H1H2LTAS im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('H1H2LTAS.png')
model = smf.mixedlm("H1H2LTAS ~ beginDiff*C(Stimmfach)",
                    testDF2_f[maskM],
                    groups= "id"
                    ).fit()
model.summary()



plt.close('all')
sns.lmplot(x='beginDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f)
plt.title('alphaRatio im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('alphaRatio.png')
model = smf.mixedlm("alphaRatio ~ beginDiff*C(Stimmfach)",
                    testDF2_f,
                    groups= "id"
                    ).fit()
model.summary()

plt.close('all')
sns.lmplot(x='beginDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[maskM])
plt.title('alphaRatio im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('alphaRatio.png')
model = smf.mixedlm("alphaRatio ~ beginDiff*C(Stimmfach)",
                    testDF2_f[maskM],
                    groups= "id"
                    ).fit()
model.summary()

plt.close('all')
sns.lmplot(x='beginDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[~maskM])
plt.title('alphaRatio im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('alphaRatio.png')
model = smf.mixedlm("alphaRatio ~ beginDiff*C(Stimmfach)",
                    testDF2_f[~maskM],
                    groups= "id"
                    ).fit()
model.summary()

plt.close('all')
sns.lmplot(x='beginDiff',y='CPPs', hue='Stimmfach',data=testDF2_f)
plt.title('CPPs im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('CPP.png')
model = smf.mixedlm("CPPs ~ beginDiff*C(Stimmfach)",
                    testDF2_f,
                    groups= "id"
                    ).fit()
model.summary()

plt.close('all')
sns.lmplot(x='beginDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[maskM])
plt.title('CPPs im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('CPP.png')
model = smf.mixedlm("CPPs ~ beginDiff*C(Stimmfach)",
                    testDF2_f[maskM],
                    groups= "id"
                    ).fit()
model.summary()

plt.close('all')
sns.lmplot(x='beginDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[~maskM])
plt.title('CPPs im Laufe des Studiums')
plt.xlabel('Jahre nach Studienbeginn')
# plt.savefig('CPP.png')
model = smf.mixedlm("CPPs ~ beginDiff*C(Stimmfach)",
                    testDF2_f[~maskM],
                    groups= "id"
                    ).fit()
model.summary()
#Only include recordings within the four year period
#Rescale the recordings to the earliest available recording time. 
#Right now it is 1.9. of the year.F That's too early and is introducing bias.
#You somehow need to model the uncertainty for the 

def normLTAS(LTAS):    
    idx = (LTAS[0]<5000)
    mask = np.where(idx)
    normFactor = sum(LTAS[1][mask])
    normLTAS = LTAS[1]/normFactor
    return normLTAS
    
def normLTAS2(LTAS):    

    normFactor = sum(LTAS)
    normLTAS = LTAS/normFactor
    return normLTAS
    


#Let's check the different LTAS areas for the normed LTAS
tLTAS = df.LTAS_400.iloc[0][0]
# df['normLTAS'] = df['LTAS_400'].apply(lambda x: normLTAS(x))

df2['normLTAS'] = df2['LTAS_400'].apply(lambda x: normLTAS2(x))
maskM = df2['pitchMed'] < 400
df1['normLTAS'] = df1['LTAS_400'].apply(lambda x: normLTAS2(x))
df6['normLTAS'] = df6['LTAS_400'].apply(lambda x: normLTAS2(x))
df_LTAS = pd.DataFrame({})
def pullLTASval(LTAS, frame):
    try:
        value = LTAS[frame]
    except IndexError:
        print(LTAS)
        return np.nan
    return value
for i in range(tLTAS.shape[0]):
    test = df2['normLTAS'].apply(lambda x: pullLTASval(x,i))
    df_LTAS['f'+str(400*i)+'_'+str(400*(i+1))] = test
    
df_LTAS['VoiceType'] = 'Sop/Mezzo/Alt'
df_LTAS.loc[maskM, 'VoiceType'] = 'Ten/Bar/Bass'
df_LTAS['id'] = df2['id']
df_LTAS['beginDiff'] = df2['beginDiff']
df_LTAS = df_LTAS[df_LTAS['beginDiff'] < 365*4]

testDF2 = df_LTAS.groupby('id')['f0_400'].count().reset_index()
mask2 = testDF2['f0_400'] > 1
df_LTAS_f = pd.merge(df_LTAS, testDF2[mask2]['id'], on='id')

for i in df_LTAS_f.drop(['VoiceType', 'id', 'beginDiff'],axis=1).columns[1:]:
    print(i)
    plt.close('all')
    sns.lmplot(x='beginDiff',y=i, hue='VoiceType',data=df_LTAS_f)
    model = smf.mixedlm(i + " ~ beginDiff*C(VoiceType)",
                        df_LTAS_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    plt.title('Sustained Tone, LTAS Window' + i[1:] + ' Hz')
    if ((model.pvalues['beginDiff'] < 0.05) | (model.pvalues['beginDiff:C(VoiceType)[T.Ten/Bar/Bass]'] < 0.05)):
        plt.savefig(i + 'sustained.png')

#Did female voices change 400-800 Hz?
maskW = df_LTAS_f['VoiceType'] == 'Sop/Mezzo/Alt'
sns.lmplot(x='beginDiff',y='f400_800', hue='VoiceType',data=df_LTAS_f[maskW])
model = smf.mixedlm("f400_800 ~ beginDiff*C(VoiceType)",
                    df_LTAS_f[maskW],
                    groups= "id"
                    ).fit()
print(model.summary())
###Not significantly with this frequency window.
#Males?
# maskW = df_LTAS_f['VoiceType'] == 'Sop/Mezzo/Alt'
sns.lmplot(x='beginDiff',y='f0_400', hue='VoiceType',data=df_LTAS_f[~maskW])
model = smf.mixedlm("f0_400 ~ beginDiff*C(VoiceType)",
                    df_LTAS_f[~maskW],
                    groups= "id"
                    ).fit()
print(model.summary())

df_LTAS_av = pd.DataFrame({})
for i in range(tLTAS.shape[0]):
    test = df1['normLTAS'].apply(lambda x: pullLTASval(x,i))
    df_LTAS_av['f'+str(400*i)+'_'+str(400*(i+1))] = test

maskM = df1['pitchMed'] < 325
df_LTAS_av['VoiceType'] = 'Sop/Mezzo/Alt'
df_LTAS_av.loc[maskM, 'VoiceType'] = 'Ten/Bar/Bass'
df_LTAS_av['id'] = df1['id']
df_LTAS_av['beginDiff'] = df1['beginDiff']
df_LTAS_av = df_LTAS_av[df_LTAS_av['beginDiff'] < 365*5]

testDF2 = df_LTAS_av.groupby('id')['f0_400'].count().reset_index()
mask2 = testDF2['f0_400'] > 1
df_LTAS_av_f = pd.merge(df_LTAS_av, testDF2[mask2]['id'], on='id')

for i in df_LTAS_av_f.drop(['VoiceType', 'id', 'beginDiff'],axis=1).columns[1:]:
    print(i)
    plt.close('all')
    sns.lmplot(x='beginDiff',y=i, hue='VoiceType',data=df_LTAS_av_f)
    model = smf.mixedlm(i + " ~ beginDiff*C(VoiceType)",
                        df_LTAS_av_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    plt.title('Avezzo a vivere, LTAS Window: ' + i[1:] + ' Hz')
    if ((model.pvalues['beginDiff'] < 0.05) | (model.pvalues['beginDiff:C(VoiceType)[T.Ten/Bar/Bass]'] < 0.05)):
        plt.savefig(i + 'avezzo.png')

maskM = df_LTAS_f['VoiceType'] == 'Sop/Mezzo/Alt'
model = smf.mixedlm('f400_800' + " ~ beginDiff",#*C(VoiceType)",
                    df_LTAS_f[maskM],
                    groups= "id"
                    ).fit()
print(model.summary())


###30 Hz windows
tLTAS_30 = df.LTAS_30.iloc[0][0]
df['normLTAS_30'] = df['LTAS_30'].apply(lambda x: normLTAS(x))
df2['normLTAS_30'] = df2['LTAS_30'].apply(lambda x: normLTAS(x))
maskM = df2['pitchMed'] < 400
# df1['normLTAS'] = df1['LTAS_400'].apply(lambda x: normLTAS(x))
df_LTAS = pd.DataFrame({})
def pullLTASval(LTAS, frame):
    try:
        value = LTAS[frame]
    except IndexError:
        print(LTAS)
        return np.nan
    return value
for i in range(tLTAS_30.shape[0]):
    test = df2['normLTAS_30'].apply(lambda x: pullLTASval(x,i))
    df_LTAS['f'+str(30*i)+'_'+str(30*(i+1))] = test
    
df_LTAS['VoiceType'] = 'Sop/Mezzo/Alt'
df_LTAS.loc[maskM, 'VoiceType'] = 'Ten/Bar/Bass'
df_LTAS['id'] = df2['id']
df_LTAS['beginDiff'] = df2['beginDiff']
df_LTAS = df_LTAS[df_LTAS['beginDiff'] < 365*5]

testDF2 = df_LTAS.groupby('id')['f0_30'].count().reset_index()
mask2 = testDF2['f0_30'] > 1
df_LTAS_f = pd.merge(df_LTAS, testDF2[mask2]['id'], on='id')

maskF = df_LTAS_f['VoiceType'] == 'Sop/Mezzo/Alt'
for i in df_LTAS_f.drop(['VoiceType', 'id', 'beginDiff'],axis=1).columns[1:]:
    print(i)
    # plt.close('all')
    model = smf.mixedlm(i + " ~ beginDiff",#*C(VoiceType)",
                        df_LTAS_f[maskF],
                        groups= "id"
                        ).fit()
    
    plt.title('Sustained Tone, LTAS Window' + i[1:] + ' Hz')
    if ((model.pvalues['beginDiff'] < 0.05/(5000/30))):# | (model.pvalues['beginDiff:C(VoiceType)[T.Ten/Bar/Bass]'] < 0.05)):
        sns.lmplot(x='beginDiff',y=i, hue='VoiceType',data=df_LTAS_f[maskF])
        print(model.summary())
        # plt.savefig(i + 'sustained30.png')


df_LTAS_av = pd.DataFrame({})
for i in range(tLTAS_30.shape[0]):
    test = df1['normLTAS'].apply(lambda x: pullLTASval(x,i))
    df_LTAS_av['f'+str(400*i)+'_'+str(400*(i+1))] = test

maskM = df1['pitchMed'] < 325
df_LTAS_av['VoiceType'] = 'Sop/Mezzo/Alt'
df_LTAS_av.loc[maskM, 'VoiceType'] = 'Ten/Bar/Bass'
df_LTAS_av['id'] = df1['id']
df_LTAS_av['beginDiff'] = df1['beginDiff']
df_LTAS_av = df_LTAS_av[df_LTAS_av['beginDiff'] < 365*5]

testDF2 = df_LTAS_av.groupby('id')['f0_30'].count().reset_index()
mask2 = testDF2['f0_30'] > 1
df_LTAS_av_f = pd.merge(df_LTAS_av, testDF2[mask2]['id'], on='id')

for i in df_LTAS_av_f.drop(['VoiceType', 'id', 'beginDiff'],axis=1).columns[1:]:
    print(i)
    plt.close('all')
    sns.lmplot(x='beginDiff',y=i, hue='VoiceType',data=df_LTAS_av_f)
    model = smf.mixedlm(i + " ~ beginDiff*C(VoiceType)",
                        df_LTAS_av_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    plt.title('Avezzo a vivere, LTAS Window: ' + i[1:] + ' Hz')
    if ((model.pvalues['beginDiff'] < 0.05) | (model.pvalues['beginDiff:C(VoiceType)[T.Ten/Bar/Bass]'] < 0.05)):
        # plt.savefig(i + 'avezzo.png')


###Find Audio Comps
maskComp = (compDF['trialNum'] == 'e') & (compDF['pitchMed'] > 400)

maskComp2 = (compDF['trialNum'] == '6') & (compDF['pitchMed'] < 325)

###How many women, how many men?
df_LTAS_f.groupby('id')[['beginDiff','VoiceType']].first().groupby('VoiceType').count()
# VoiceType
# Sop/Mezzo/Alt         88
# Ten/Bar/Bass          52

#How many total?
df_LTAS_f.groupby('id')[['beginDiff','VoiceType']].count().count()
#140


#Plot changes in H1H2LTAS:
dfList = [df1,df2,df6]
for d in dfList:
    d['yearDiff'] = d['beginDiff']/365
df6['alphaRatio'] = 1/df6['alphaRatio']
strList = ['Vowel', 'Triad', 'Avezzo']
for i in range(3):
    df_i = dfList[i]
    df_i = df_i[df_i['CPPs'].notna()]

    
    #Plot changes in H1H2LTAS:
    plt.close('all')
    testDF = df_i[df_i['yearDiff'] < 4]
    testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
    mask2 = testDF2['Jahr'] > 1
    testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id')
    if (i == 0) | (i == 2):
        maskM = testDF2_f['pitchMed'] < 325
        maskF = ~maskM
    else:
        maskM = testDF2_f['pitchMed'] < 400
        maskF = ~maskM
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f)
    plt.title('H1H2LTAS ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    plt.savefig('H1H2LTAS_' + strList[i] + '.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    
    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('H1H2LTAS ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    plt.savefig('H1H2LTAS_' + strList[i] + 'M.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskF])
    plt.title('H1H2LTAS ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    plt.savefig('H1H2LTAS_' + strList[i] + 'F.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskF],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f)
    plt.title('alphaRatio ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    g.tight_layout()
    plt.savefig('alphaRatio'+strList[i]+'.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('alphaRatio ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    g.tight_layout()
    plt.savefig('alphaRatio'+strList[i]+'M.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('alphaRatio ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    g.tight_layout()
    plt.savefig('alphaRatio'+strList[i]+'F.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f)
    plt.title('CPPs ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    g.tight_layout()
    plt.savefig('CPPs' + strList[i] + '.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('CPPs ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    g.tight_layout()
    plt.savefig('CPPs' + strList[i] + 'M.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('CPPs ' + strList[i] + ' im Laufe des Studiums')
    plt.xlabel('Jahre nach Studienbeginn')
    g.tight_layout()
    plt.savefig('CPPs' + strList[i] + 'F.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())







###
plt.close('all')
df = pd.merge(pd.merge(df1,df2, on=['id','date']),df6, on=['id','date'])
#Let's get all recordings under four years
testDF = df[df['beginDiff'] < 365*4]
#We have some null values from corrupted audio files:
testDF = testDF[testDF['CPPs'].notna()]
#Find all recordings that have more than one recording in this time frame:
testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
mask2 = testDF2['Jahr'] > 1
#Remerge with testDF
testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id').reset_index(drop=True)
testDF2_f.groupby('id').count().count()
#187 unique students
testDF2_f.groupby('id')[['Jahr','geschlecht']].first().groupby('geschlecht').count()
# männl.        72
# weibl.       114

###Find largest differences
def deltaX(df, variable, id):
    minDate = df[(df['id'] == id)]['date'].min()
    maxDate = df[df['id'] == id]['date'].max()
    delta = df[(df['id'] == id) & (df['date'] == maxDate)][variable] -  df[(df['id'] == id) & (df['date'] == minDate)][variable]
    return delta

"""
Let's write the abstract right here.
From 2002 to the present, students at the Hochschule für Musik Carl Maria von Weber Dresden have recorded a series of exercises yearly.
Among these exercises is a vowel glide exercise /aeiou/ in medium range, an ascending triad on an /a/ vowel with a sustained fifth, and an exercise from Vaccai's Metodo pratico di canto ("Avezzo a vivere").
The goal of this retrospective longitudinal study is to examine the spectral development of male and female voices, specifically 
    high frequency to low frequency energy ratio (alpha ratio)
    energy in the sung octave to the energy one octave above the sung octave (H1H2LTAS)
    breathiness as measured by CPP
These measures will be compared for sustained sung tones in medium and medium high tessitura as well as for the repertoire sample.
Of the 240 available students in the acoustic database, 187 (114 F, 72 M) had each of the three chosen samples recorded on multiple dates within a four year span.
A linear mixed model was performed with 
"""


#Plot changes in H1H2LTAS:
dfList = [df1,df2,df6]
for d in dfList:
    d['yearDiff'] = d['beginDiff']/365
# df6['alphaRatio'] = 1/df6['alphaRatio']
strList = ['Vowel11', 'Triad11', 'Avezzo11']
for i in range(3):
    df_i = dfList[i]
    # df_i = df_i[df_i['CPPs'].notna()]

    
    #Plot changes in H1H2LTAS:
    plt.close('all')
    testDF = df_i[df_i['yearDiff'] < 4]
    testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
    mask2 = testDF2['Jahr'] > 1
    testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id')
    if (i == 0) | (i == 2):
        maskM = testDF2_f['pitchMed'] < 325
        maskF = ~maskM
    else:
        maskM = testDF2_f['pitchMed'] < 400
        maskF = ~maskM
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f)
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefig('H1H2LTAS_' + strList[i] + '.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    
    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefig('H1H2LTAS_' + strList[i] + 'M.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskF])
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefig('H1H2LTAS_' + strList[i] + 'F.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskF],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f)
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefig('alphaRatio'+strList[i]+'.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefig('alphaRatio'+strList[i]+'M.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefig('alphaRatio'+strList[i]+'F.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f)
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefig('CPPs' + strList[i] + '.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefig('CPPs' + strList[i] + 'M.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefig('CPPs' + strList[i] + 'F.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())


df_i = df1
i = 0
df_i = df_i[df_i['ampSD'].notna()]
testDF = df_i[df_i['yearDiff'] < 4]
testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
mask2 = testDF2['Jahr'] > 1
testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id')
if (i == 0) | (i == 2):
    maskM = testDF2_f['pitchMed'] < 325
    maskF = ~maskM
else:
    maskM = testDF2_f['pitchMed'] < 400
    maskF = ~maskM

plt.close('all')
g = sns.lmplot(x='yearDiff',y='ampSD', hue='Stimmfach',data=testDF2_f)
plt.title('ampSD ' + strList[i] + ' During Study')
plt.xlabel('Year of Study')
g.tight_layout()
plt.savefig('ampSD' + strList[i] + '.png')
model = smf.mixedlm("ampSD ~ yearDiff*C(Stimmfach)",
                    testDF2_f,
                    groups= "id"
                    ).fit()
print(model.summary())

plt.close('all')
g = sns.lmplot(x='yearDiff',y='hammSD', hue='Stimmfach',data=testDF2_f)
plt.title('hammSD ' + strList[i] + ' During Study')
plt.xlabel('Year of Study')
g.tight_layout()
plt.savefig('hammSD' + strList[i] + '.png')
model = smf.mixedlm("hammSD ~ yearDiff*C(Stimmfach)",
                    testDF2_f,
                    groups= "id"
                    ).fit()
print(model.summary())

plt.close('all')
g = sns.lmplot(x='yearDiff',y='slopeSD', hue='Stimmfach',data=testDF2_f)
plt.title('slopeSD ' + strList[i] + ' During Study')
plt.xlabel('Year of Study')
g.tight_layout()
plt.savefig('slopeSD' + strList[i] + '.png')
model = smf.mixedlm("slopeSD ~ yearDiff*C(Stimmfach)",
                    testDF2_f,
                    groups= "id"
                    ).fit()
print(model.summary())

###Year Grouping Constant
dfList = [df1,df2,df6]
for d in dfList:
    cMask0 = (d['Year'] < 2008)
    cMask1 = (d['Year'] > 2007) & (d['Year'] < 2019)
    cMask2 = (d['Year'] > 2018)
    d.loc[cMask0, 'yearGroup'] = 'A'
    d.loc[cMask1, 'yearGroup'] = 'B'
    d.loc[cMask2, 'yearGroup'] = 'C'
    d['yearDiff'] = d['beginDiff']/365
# df6['alphaRatio'] = 1/df6['alphaRatio']
strList = ['Vowel', 'Triad', 'Avezzo']

for i in range(3):
    df_i = dfList[i]
    df_i = df_i[df_i['CPPs'].notna()]

    
    #Plot changes in H1H2LTAS:
    plt.close('all')
    testDF = df_i[df_i['yearDiff'] < 4]
    testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
    mask2 = testDF2['Jahr'] > 1
    testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id')
    if (i == 0) | (i == 2):
        maskM = testDF2_f['pitchMed'] < 325
        maskF = ~maskM
    else:
        maskM = testDF2_f['pitchMed'] < 400
        maskF = ~maskM
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f)
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefigfig('H1H2LTAS_' + strList[i] + '.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    
    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefigfig('H1H2LTAS_' + strList[i] + 'M.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskF])
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefigfig('H1H2LTAS_' + strList[i] + 'F.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f[maskF],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f)
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('alphaRatio'+strList[i]+'.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('alphaRatio'+strList[i]+'M.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('alphaRatio'+strList[i]+'F.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f)
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('CPPs' + strList[i] + '.png')
    model = smf.mixedlm("CPPs ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('CPPs' + strList[i] + 'M.png')
    model = smf.mixedlm("CPPs ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('CPPs' + strList[i] + 'F.png')
    model = smf.mixedlm("CPPs ~ yearDiff + C(Stimmfach) + C(yearGroup)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())


###Outlier Analysis
dfList = [df1,df2,df6]
varList = ['alphaRatio','H1H2LTAS', 'CPPs'] 
for i in range(len(dfList)):
    print(str(i))
    print(' ')
    for j in varList:
        print(j)
        print('Max: ' + str(dfList[i].sort_values(j, ascending=False).iloc[0][['id', 'date']]))
        print('Min: ' + str(dfList[i].sort_values(j).iloc[0][['id', 'date']]))#ascending=False)))
        
        
        
#Year as factor
###Year constant
dfList = [df1,df2,df6]
for d in dfList:
    # cMask0 = d['Year'] == 2002
    cMask1 = (d['Year'] > 2002) & (d['Year'] < 2013)
    cMask2 = (d['Year'] > 2012)
    # d.loc[cMask0, 'yearGroup'] = 'A'
    d.loc[cMask1, 'yearGroup'] = 'B'
    d.loc[cMask2, 'yearGroup'] = 'C'
    d['yearDiff'] = d['beginDiff']/365
# df6['alphaRatio'] = 1/df6['alphaRatio']
strList = ['Vowel', 'Triad', 'Avezzo']

for i in range(3):
    df_i = dfList[i]
    df_i = df_i[df_i['CPPs'].notna()]

    
    #Plot changes in H1H2LTAS:
    plt.close('all')
    testDF = df_i[df_i['yearDiff'] < 4]
    testDF2 = testDF.groupby('id')['Jahr'].count().reset_index()
    mask2 = testDF2['Jahr'] > 1
    testDF2_f = pd.merge(testDF, testDF2[mask2]['id'], on='id')
    if (i == 0) | (i == 2):
        maskM = testDF2_f['pitchMed'] < 325
        maskF = ~maskM
    else:
        maskM = testDF2_f['pitchMed'] < 400
        maskF = ~maskM
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f)
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefigfig('H1H2LTAS_' + strList[i] + '.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())
    
    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefigfig('H1H2LTAS_' + strList[i] + 'M.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='H1H2LTAS', hue='Stimmfach',data=testDF2_f[maskF])
    plt.title('H1H2LTAS ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    #plt.xlabel('Studiumjahr')
    plt.ylim(-1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].max(), 1.2*(testDF2_f['H1H2LTAS'].max() - testDF2_f['H1H2LTAS'].min()) + testDF2_f['H1H2LTAS'].min())
    g.tight_layout()
    #plt.legend(loc='upper right')
    #plt.savefigfig('H1H2LTAS_' + strList[i] + 'F.png')
    model = smf.mixedlm("H1H2LTAS ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f[maskF],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f)
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('alphaRatio'+strList[i]+'.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('alphaRatio'+strList[i]+'M.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='alphaRatio', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('alphaRatio ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('alphaRatio'+strList[i]+'F.png')
    model = smf.mixedlm("alphaRatio ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f)
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('CPPs' + strList[i] + '.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f,
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[maskM])
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('CPPs' + strList[i] + 'M.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f[maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())

    # plt.close('all')
    g = sns.lmplot(x='yearDiff',y='CPPs', hue='Stimmfach',data=testDF2_f[~maskM])
    plt.title('CPPs ' + strList[i] + ' During Study')
    plt.xlabel('Year of Study')
    g.tight_layout()
    #plt.savefigfig('CPPs' + strList[i] + 'F.png')
    model = smf.mixedlm("CPPs ~ yearDiff*C(Stimmfach) + C(Year)",
                        testDF2_f[~maskM],
                        groups= "id"
                        ).fit()
    print(model.summary())
    
    
    
###Checking time dependence
strList = ['Vowel', 'Triad', 'Avezzo']
for i in range(3):
    df_i = dfList[i]
    df_i = df_i[df_i['CPPs'].notna()]
    plt.close()
    sns.scatterplot(x='date', y='CPPs', data=df_i,hue='Stimmfach')
    #plt.legend('', frameon=False)
    plt.title('Time Dependency CPPs' + strList[i])
    plt.savefig('YearDepCPP' + strList[i] + '.png')
    plt.close()
    sns.scatterplot(x='date', y='H1H2LTAS', data=df_i,hue='Stimmfach')
    #plt.legend('', frameon=False)
    plt.title('Time Dependency H1H2LTAS ' + strList[i])
    plt.savefig('YearDepH1H2LTAS' + strList[i] + '.png')
    plt.close()
    sns.scatterplot(x='date', y='alphaRatio', data=df_i,hue='Stimmfach')
    #plt.legend('', frameon=False)
    plt.title('Time Dependency alphaRatio ' + strList[i])
    plt.savefig('YearDepalphaRatio' + strList[i] + '.png')   
    
    
###Checking what recordings are missing:
predf = df1.groupby('id').first().reset_index()
D1 = pd.read_csv('Klang1.csv')
D1 = D1.drop(columns='Unnamed: 0')
postdf = D1.groupby('id').first().reset_index()

res = predf['id'][~predf['id'].isin(list(postdf['id']))].values

with open('SNRdraft.pkl', 'rb') as f:
    snr = pd.read_pickle(f)

snr['id'] = snr['id'].astype(int)
df = pd.merge(df, snr, on=['id', 'trialNum','date'])


plt.close('all')
fig, ax = plt.subplots(3,1,figsize=(10,30))
ax[0].plot(df1['Year'].astype(int), df1['zSNR'], 'o')
ax[0].plot(df1['Year'].astype(int), df1['zCPP'], 'o')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Z Score')
ax[0].set_title('Mean Centered SNR (Blue) and CPPs(Orange) Low Pitch')

ax[1].plot(df2['Year'].astype(int), df2['zSNR'], 'o')
ax[1].plot(df2['Year'].astype(int), df2['zCPP'], 'o')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Z Score')
ax[1].set_title('Mean Centered SNR (Blue) and CPPs(Orange) High Pitch')

ax[2].plot(df6['Year'].astype(int), df6['zSNR'], 'o')
ax[2].plot(df6['Year'].astype(int), df6['zCPP'], 'o')
ax[2].set_xlabel('Year')
ax[2].set_ylabel('Z Score')
ax[2].set_title('Mean Centered SNR (Blue) and CPPs(Orange) Repertoire Sample')

plt.savefig('CPPvSNRcomp.png')


###Find CPPs extremes
mask1 = (df1['yearFloor'] == 0) & (df1['Stimmfach'] == 'Sop/Mezzo/Alt') #& (df1['CPPs'] < 14)
mask2 = (df1['yearFloor'] == 4) & (df1['Stimmfach'] == 'Sop/Mezzo/Alt')
df1.iloc[df1[mask1]['CPPs'].argmin()]

diffDF = pd.merge(df1[mask1][['id', 'CPPs']], df1[mask2][['id', 'CPPs']], on='id')
diffDF['diff'] = diffDF['CPPs_y']-diffDF['CPPs_x']
diffDF.sort_values('diff',ascending=False)

###Find CPPs extremes
mask1 = (df6['yearFloor'] == 0) & (df6['Stimmfach'] == 'Sop/Mezzo/Alt') #& (df6['CPPs'] < 14)
mask2 = (df6['yearFloor'] == 4) & (df6['Stimmfach'] == 'Sop/Mezzo/Alt')
df6.iloc[df6[mask1]['CPPs'].argmin()]

diffDF = pd.merge(df6[mask1][['id', 'CPPs']], df6[mask2][['id', 'CPPs']], on='id')
diffDF['diff'] = diffDF['CPPs_y']-diffDF['CPPs_x']
diffDF.sort_values('diff',ascending=False)


###Paper questions:
#How many subjects were there?
df1 = df_f[0]
df2 = df_f[1]
df6 = df_f[2]
df1.groupby('id').first().groupby('geschlecht').count()

np.median(df1.groupby('id')['beginDiff'].max())/365.25
np.median(df2.groupby('id')['beginDiff'].max())/365.25
np.median(df6.groupby('id')['beginDiff'].max())/365.25
#All 3.29 years

#What is the median time from first to last recording?
df1['minDate2'] = df1['id'].apply(lambda x: df1[['id', 'date']].loc[df1['id'] == x].groupby('id').min().iloc[0])
df2['minDate2'] = df2['id'].apply(lambda x: df2[['id', 'date']].loc[df2['id'] == x].groupby('id').min().iloc[0])
df6['minDate2'] = df6['id'].apply(lambda x: df6[['id', 'date']].loc[df6['id'] == x].groupby('id').min().iloc[0])

df1['maxDate'] = df1['id'].apply(lambda x: df1[['id', 'date']].loc[df1['id'] == x].groupby('id').max().iloc[0])
df2['maxDate'] = df2['id'].apply(lambda x: df2[['id', 'date']].loc[df2['id'] == x].groupby('id').max().iloc[0])
df6['maxDate'] = df6['id'].apply(lambda x: df6[['id', 'date']].loc[df6['id'] == x].groupby('id').max().iloc[0])

df1['dayDiff'] = (df1['maxDate'] - df1['minDate2'])
df1['yearDiff'] = df1['dayDiff'].apply(lambda x: x.days/365.25)
df2['dayDiff'] = (df2['maxDate'] - df2['minDate2'])
df2['yearDiff'] = df2['dayDiff'].apply(lambda x: x.days/365.25)
df6['dayDiff'] = (df6['maxDate'] - df6['minDate2'])
df6['yearDiff'] = df6['dayDiff'].apply(lambda x: x.days/365.25)