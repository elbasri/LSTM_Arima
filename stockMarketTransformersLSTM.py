import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import math
import re
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load CSV file and split columns correctly
CIH_data = pd.read_csv("data/CIH_DS.csv", delimiter=",", quotechar='"', header=None)
CIH_data = CIH_data[0].str.split(',"', expand=True)
CIH_data = CIH_data.replace('"', '', regex=True)

CIH_data.columns = CIH_data.iloc[0]
CIH_data = CIH_data.drop(0)

# Set the first row as column headers and drop it from the DataFrame
def convert_volume_to_numeric(df, column_name):
    # Supprimer 'K', remplacer ',' par '.', et gérer les valeurs vides
    df[column_name] = df[column_name].str.replace('K', '', regex=False)  # Supprimer 'K'
    df[column_name] = df[column_name].str.replace(',', '.', regex=False)  # Remplacer ',' par '.'
    df[column_name] = df[column_name].replace('', np.nan)  # Remplacer les chaînes vides par NaN
    df[column_name] = df[column_name].astype(float) * 1000  # Convertir en float et multiplier par 1000

       
    # Remplir les NaN avec la dernière valeur non nulle
    df[column_name] = df[column_name].fillna(method='ffill')
    
    return df

# Appliquer la fonction à la colonne 'Vol.'
CIH_data = convert_volume_to_numeric(CIH_data, 'Vol.')

# Vérifier le résultat
CIH_data.head()


CIH_data.isna().sum()

CIH_data.describe().T

CIH_data['Date'] = pd.to_datetime(CIH_data['Date'], dayfirst=True)
CIH_data.sort_values(by='Date', inplace=True)
CIH_data.set_index('Date', inplace=True)

min_date = CIH_data.index.min()
max_date = CIH_data.index.max()
date_range = pd.date_range(start=min_date, end=max_date)
CIH_data = CIH_data.reindex(date_range)
CIH_data = CIH_data.fillna(method='ffill')
CIH_data.reset_index(inplace=True)

CIH_data = CIH_data.replace('",', regex=False)

# Remplacer '",' dans les noms de colonnes par une chaîne vide
CIH_data.columns = CIH_data.columns.str.replace('",', '', regex=False)

# Reset index with a custom column name for the date
CIH_data.reset_index(inplace=True)
CIH_data.rename(columns={'index': 'Date'}, inplace=True)

CIH_data.head()


# Drop the 'level_0' column
CIH_data.drop(columns='level_0', inplace=True)

# Display the updated DataFrame
CIH_data.head()


BCP_data = pd.read_csv("data/BCP_DS.csv", delimiter=",", quotechar='"', header=None)
BCP_data = BCP_data[0].str.split(',"', expand=True)
BCP_data = BCP_data.replace('"', '', regex=True)

BCP_data.columns = BCP_data.iloc[0]
BCP_data = BCP_data.drop(0)


def convert_volume_to_numeric(df, column_name):
    """
    Convertit les valeurs de volume en nombres, en prenant en compte les notations
    avec 'K' (milliers) et 'M' (millions).
    """
    # Remplacer 'K' par rien et 'M' par '*1_000'
    df[column_name] = df[column_name].replace('', np.nan)
    df[column_name] = df[column_name].str.replace('K', '', regex=False)
    df[column_name] = df[column_name].str.replace('M', '*1_000', regex=False)
    
    # Remplacer les virgules par des points
    df[column_name] = df[column_name].str.replace(',', '.', regex=False)
    
    # Appliquer pd.eval uniquement aux lignes non nulles
    df[column_name] = df[column_name].apply(lambda x: pd.eval(x) if pd.notnull(x) else np.nan)
    
    # Convertir en valeurs numériques et multiplier par 1 000
    df[column_name] = df[column_name] * 1_000
       
    # Remplir les NaN avec la dernière valeur non nulle
    df[column_name] = df[column_name].fillna(method='ffill')
    
    return df

# Appliquer la fonction à la colonne 'Vol.' pour BCP_data
BCP_data = convert_volume_to_numeric(BCP_data, 'Vol.')

# Vérifier le résultat
BCP_data.head()


BCP_data['Date'] = pd.to_datetime(BCP_data['Date'], dayfirst=True)
BCP_data.sort_values(by='Date', inplace=True)
BCP_data.set_index('Date', inplace=True)

min_date = BCP_data.index.min()
max_date = BCP_data.index.max()
date_range = pd.date_range(start=min_date, end=max_date)
BCP_data = BCP_data.reindex(date_range)
BCP_data = BCP_data.fillna(method='ffill')
BCP_data.reset_index(inplace=True)

BCP_data = BCP_data.replace('",', regex=False)

# Remplacer '",' dans les noms de colonnes par une chaîne vide
BCP_data.columns = BCP_data.columns.str.replace('",', '', regex=False)

# Reset index with a custom column name for the date
BCP_data.reset_index(inplace=True)
BCP_data.rename(columns={'index': 'Date'}, inplace=True)

# Drop the 'level_0' column
BCP_data.drop(columns='level_0', inplace=True)

BCP_data.head()

CIH_data['Company'] = 'CIH'
BCP_data['Company'] = 'BCP'

Df = pd.concat([CIH_data, BCP_data], ignore_index=True)

print("len CIH_data", len(CIH_data))
print("len BCP_data", len(BCP_data))
print("len Df Totale", len(Df))

def clean_and_convert_to_float(value):

  cleaned_value = value.replace(',', '').replace('.', '')
  return float(cleaned_value)

Df['Dernier'] = Df['Dernier'].apply(clean_and_convert_to_float)
Df['Ouv.'] = Df['Ouv.'].apply(clean_and_convert_to_float)
Df[' Plus Haut'] = Df[' Plus Haut'].apply(clean_and_convert_to_float)
Df['Plus Bas'] = Df['Plus Bas'].apply(clean_and_convert_to_float)
Df['Variation %'] = Df['Variation %'].str.replace('%', '').apply(clean_and_convert_to_float)

Df['Variation %'].dtype

Df.tail()

# Paramètres de la figure
plt.figure(figsize=(15, 10), facecolor='white', edgecolor='white')
plt.subplots_adjust(top=1.25, bottom=0.8, hspace=0.3, wspace=0.2)

# Obtenir les entreprises uniques
companies = Df['Company'].unique()

# Tracer les sous-graphiques pour chaque entreprise
for i, company in enumerate(companies, 1):
    company_df = Df[Df['Company'] == company]

    plt.subplot(2, 2, i)  # Adapter (2, 2, i) selon le nombre d'entreprises pour organiser les sous-graphiques
    plt.plot(company_df['Date'], company_df['Dernier'], color='blue', linewidth=2)
    plt.title(f'Closing Price of {company} Stock', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Close Price', fontsize=10)
    plt.xticks(rotation=45)

    # Configurer l'affichage des dates
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))  # Afficher au plus 12 ticks
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))  # Format des dates (mois année)

    # Masquer les bordures supérieures et droites
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_frame_on(False)

plt.tight_layout()
plt.show()


# Paramètres de la figure
plt.figure(figsize=(15, 10), facecolor='white', edgecolor='white')
plt.subplots_adjust(top=1.25, bottom=0.8, hspace=0.3, wspace=0.2)

# Obtenir les entreprises uniques
companies = Df['Company'].unique()

# Tracer les sous-graphiques pour chaque entreprise
for i, company in enumerate(companies, 1):
    company_df = Df[Df['Company'] == company]

    plt.subplot(2, 2, i)  # Adapter (2, 2, i) selon le nombre d'entreprises pour organiser les sous-graphiques
    plt.plot(company_df['Date'], company_df['Ouv.'], color='blue', linewidth=2)
    plt.title(f'Opening Price of {company} Stock', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Open Price', fontsize=10)
    plt.xticks(rotation=45)

    # Configurer l'affichage des dates
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))  # Afficher au plus 12 ticks
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))  # Format des dates (mois année)

    # Masquer les bordures supérieures et droites
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_frame_on(False)

plt.tight_layout()
plt.show()


# Paramètres de la figure
plt.figure(figsize=(20, 14), facecolor='white', edgecolor='white')
plt.subplots_adjust(top=1.25, bottom=0.8, hspace=0.3, wspace=0.2)

# Obtenir les entreprises uniques
companies = Df['Company'].unique()

# Tracer les sous-graphiques pour chaque entreprise
for i, company in enumerate(companies, 1):
    company_df = Df[Df['Company'] == company]

    # Calculer la différence entre le prix de clôture et d'ouverture
    company_df['Difference'] = company_df['Dernier'] - company_df['Ouv.']

    plt.subplot(2, 2, i)  # Adapter (2, 2, i) selon le nombre d'entreprises pour organiser les sous-graphiques

    # Tracer le prix d'ouverture en vert
    plt.plot(company_df['Date'], company_df['Ouv.'], color='green', linewidth=2, label='Opening Price')

    # Tracer le prix de clôture en bleu
    plt.plot(company_df['Date'], company_df['Dernier'], color='blue', linewidth=2, label='Closing Price')

    # Tracer la différence en rouge
    plt.plot(company_df['Date'], company_df['Difference'], color='red', linewidth=2, label='Difference (Close - Open)')

    plt.title(f'Opening, Closing Price and Difference for {company} Stock', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Price', fontsize=10)
    plt.xticks(rotation=45)

    # Ajouter une légende
    plt.legend()

    # Configurer l'affichage des dates
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))  # Afficher au plus 12 ticks
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))  # Format des dates (mois année)

    # Masquer les bordures supérieures et droites
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_frame_on(False)

plt.tight_layout()
plt.show()


Df['Avg Daily Return'] = (Df['Dernier'] - Df['Ouv.']) / Df['Ouv.']
pivot_df = Df.pivot_table(index='Date', columns='Company', values=['Avg Daily Return', 'Dernier', 'Vol.'])
correlation_returns = pivot_df['Avg Daily Return'].corr()
correlation_closing = pivot_df['Dernier'].corr()

plt.figure(figsize=(12, 10), facecolor='white')

plt.subplot(2, 2, 1)
sns.heatmap(correlation_returns, annot=True, cmap='summer')
plt.title('Correlation of Avg Daily Return', fontsize=10)

plt.subplot(2, 2, 2)
sns.heatmap(correlation_closing, annot=True, cmap='summer')
plt.title('Correlation of Closing Price', fontsize=10)

plt.tight_layout()
plt.show()



company_dfs = []
for company in Df['Company'].unique():
    company_df = Df[Df['Company'] == company].copy()
    company_df['Date'] = pd.to_datetime(company_df['Date'], format='%d/%m/%y')
    company_df.set_index('Date', inplace=True)
    company_df.rename(columns={'Ouv.': 'Open', 'Plus Haut': 'High', 'Plus Bas': 'Low', 'Dernier': 'Close', 'Vol.': 'Volume'}, inplace=True)
    company_dfs.append(company_df)


# Split data into train and test sets
train_data, test_data = company_dfs[0].iloc[3:int(len(company_dfs[0])*0.9)], company_dfs[0].iloc[int(len(company_dfs[0])*0.9):]

# Plotting
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices for CIH')
plt.plot(train_data.index, train_data['Close'], 'green', label='Train data')
plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
plt.legend()
plt.show()


# pour BCP
# Split data into train and test sets
train_data, test_data = company_dfs[1].iloc[3:int(len(company_dfs[1])*0.9)], company_dfs[1].iloc[int(len(company_dfs[1])*0.9):]

# Plotting
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices for BCP')
plt.plot(train_data.index, train_data['Close'], 'green', label='Train data')
plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
plt.legend()
plt.show()

train_data, test_data = company_dfs[0].iloc[3:int(len(company_dfs[0])*0.9)], company_dfs[0].iloc[int(len(company_dfs[0])*0.9):]
train_data_close = train_data['Close']
test_data_close = test_data['Close']

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(np.array(train_data_close).reshape(-1, 1))
test_data_normalized = scaler.transform(np.array(test_data_close).reshape(-1, 1))

# Define LSTM model architecture
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(train_data_normalized.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Define callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train model
history = model.fit(x=train_data_normalized.reshape((train_data_normalized.shape[0], train_data_normalized.shape[1],1)),
y=train_data_normalized,
epochs=50,
batch_size=32,
validation_split=0.1,
callbacks=[early_stopping, model_checkpoint],
verbose=1)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
