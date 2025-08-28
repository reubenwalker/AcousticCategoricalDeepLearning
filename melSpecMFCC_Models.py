import pickle
from datetime import datetime
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('mergedResults.pkl')
df5 = df[df['trialNum'] == '5'].copy()
df5['Stimmfach'] = 'Sop/Mezzo/Alt'
df5.loc[df5['pitchMed'] < 325, 'Stimmfach'] = 'Ten/Bar/Bass'
df5.loc[df5['pitchMed'] >= 325, 'Stimmfach'] = 'Sop/Mezzo/Alt'

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

df0 = df5[df5['beginDiff'] < 365].groupby('id').first().reset_index()
df0 = df0[(df0['Jahr'].astype('int') >= 2008) & (df0['Jahr'].astype('int') <= 2018)] #Subset 2008-2018 for recording consistency

timbreColumns = ['timbre','SoloChor','nasal','atemrelation']

# test = variable_level_balance_summary(df0)
test = pd.read_pickle('nullDistributions.pkl')
# df0 = df5[((df5['beginDiff'] > 365*3) & (df5['beginDiff'] < 365*4))].groupby('id').first().reset_index()
use_mlp = True  # set to False to use CNN
final_accuracies = []
count = 0
for j in range(1): 
    if j == 1:
        use_mlp = False
    for i in timbreColumns:#df5.columns:
        k_folds = 5
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
        if test.loc[i, 'n_levels'] > 3:
            continue
        count +=1
        if test.loc[i, 'n_levels'] == 2: 
            if test.loc[i, 'balance'] < 0.4:
                continue
            if test.loc[i, 'n_non_na'] < 100:
                print(i,', n = ',test.loc[i, 'n_non_na'])
                k_folds = 3
        if test.loc[i, 'n_levels'] == 3: 
            if test.loc[i, 'balance'] < 0.6:
                continue
            if test.loc[i, 'n_non_na'] < 100:
                print(i,', n = ',test.loc[i, 'n_non_na'])
                k_folds = 3


        final_accuracies = []
        df0 = df5[df5['beginDiff'] < 365].groupby('id').first().reset_index()
        df0 = df0[(df0['Jahr'].astype('int') >= 2008) & (df0['Jahr'].astype('int') <= 2018)]
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
        for t in range(1): #For null distribution simulations
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
        # if use_mlp:
            # test.loc[i, 'mlp_accuracy'] = mean_acc
            # test.loc[i, 'mlp_std'] = np.std(accuracies)
            # test.loc[i, 'mlp_sensitivity'] = mean_sens
            # test.loc[i, 'mlp_specificity'] = mean_spec
            # test.loc[i, 'mlp_null'] = np.mean(final_accuracies)
            # test.loc[i, 'mlp_null_std'] = np.std(final_accuracies)
        # else:
            # test.loc[i, 'cnn_accuracy'] = mean_acc
            # test.loc[i, 'cnn_std'] = np.std(accuracies)
            # test.loc[i, 'cnn_sensitivity'] = mean_sens
            # test.loc[i, 'cnn_specificity'] = mean_spec
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

