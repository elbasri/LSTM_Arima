import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Load CSV file and split columns correctly
df1 = pd.read_csv("data/BCP_DS.csv", delimiter=",", quotechar='"', header=None)
df1 = df1[0].str.split(',"', expand=True)
df1 = df1.replace('"', '', regex=True)

# Set the first row as column headers and drop it from the DataFrame
df1.columns = df1.iloc[0]
df1 = df1.drop(0).reset_index(drop=True)

def convert_volume_to_numeric(volume_str):
    volume_str = volume_str.replace(',', '.').replace(' ', '')
    if pd.isna(volume_str) or volume_str.strip() == '':
        return None 
    if volume_str.endswith('K'):
        return float(volume_str[:-1]) * 1000
    elif volume_str.endswith('M'):
        return float(volume_str[:-1]) * 1000000
    else:
        return float(volume_str)

# Convert columns to appropriate data types
df1['Vol.'] = df1['Vol.'].apply(convert_volume_to_numeric)
df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)
df1.sort_values(by='Date', inplace=True)

# Check for and fill missing data
min_data = df1['Date'].min()
max_data = df1['Date'].max()
date_range = pd.date_range(start=min_data, end=max_data) 
df1.set_index('Date', inplace=True)
df1 = df1.reindex(date_range).ffill().reset_index()

df2 = pd.read_csv("data/CIH_DS.csv", delimiter=",", quotechar='"', header=None)
df2 = df2[0].str.split(',"', expand=True)
df2 = df2.replace('"', '', regex=True)

df2.columns = df2.iloc[0]
df2 = df2.drop(0).reset_index(drop=True)

df2['Vol.'] = df2['Vol.'].apply(convert_volume_to_numeric)
df2['Date'] = pd.to_datetime(df2['Date'], dayfirst=True)
df2.sort_values(by='Date', inplace=True)

# Fill missing data in df2
min_data = df2['Date'].min()
max_data = df2['Date'].max()
date_range = pd.date_range(start=min_data, end=max_data)
df2.set_index('Date', inplace=True)
df2 = df2.reindex(date_range).ffill().reset_index()

# Add company names
df1['Company'] = "BCP"
df2['Company'] = "CIH"

# Ensure Dernier is numeric and drop rows with NaN values
df1['Dernier'] = pd.to_numeric(df1['Dernier'], errors='coerce')
df1.dropna(subset=['Dernier'], inplace=True)
df2['Dernier'] = pd.to_numeric(df2['Dernier'], errors='coerce')
df2.dropna(subset=['Dernier'], inplace=True)

# Combine the dataframes
df_total = pd.concat([df1, df2], ignore_index=True)

# Plotting
plt.figure(figsize=(15, 10), facecolor='white', edgecolor='white')
plt.subplots_adjust(top=1.25, bottom=0.8, hspace=0.2, wspace=0.2)

companies = df_total['Company'].unique()
for i, company in enumerate(companies, 1):
    company_df = df_total[df_total['Company'] == company]
    plt.subplot(2, 2, i)
    plt.plot(company_df['Date'], company_df['Dernier'], color='blue', linewidth=2)
    plt.title(f'Closing Price of {company} Stock', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Close Price', fontsize=10)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b %Y'))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_frame_on(False)

plt.tight_layout()
plt.show()
