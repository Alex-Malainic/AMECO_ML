import pandas as pd
import numpy as np
import utils
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 10)

# Read dataset
df = pd.read_csv('combined_AMECO_data.csv')

# Choose the UK as the analyzed country (most data)
df = df[df['COUNTRY'] == 'United Kingdom']

# Drop unnecessary columns
df.drop(columns = ["SERIES", "REF", "CNTRY", "TRN", "AGG", "UNIT", "CODE", "COUNTRY", "SUB-CHAPTER"], inplace = True)

# Remove column name trailing and leading whitespace
df['TITLE']= df['TITLE'].str.strip()
df['UNIT.1']= df['UNIT.1'].str.strip()


# Define a list of macroeconomic indicators of interest to analyse
# Where possible, indicators are chosen at current prices in EUR
indicators_rename = {
              'Unemployment rate: total :- Member States: definition EUROSTAT':'Unemployment',
              'National consumer price index (All-items)': 'CPI',
              'Gross national saving': 'National_Savings',
              'Final demand at current prices':'Final_Demand',
              'Gross domestic product at current prices per head of population': 'GDP/Capita',
              'Gross Value Added at current prices: total of branches': 'GVA',
              'Exports of goods and services at current prices (National accounts)': 'Exports',
              'Imports of goods and services at current prices (National accounts)': 'Imports',
              'Nominal long-term interest rates': 'Long_IR',
              'Nominal short-term interest rates': 'Short_IR',
              'Yield curve': 'Yield_Curve',
              }
# Get list of indicators
indicators = list(indicators_rename.keys())

# Define a dictionary of units (where applicable)
units = {'Gross national saving':'Mrd ECU/EUR',
         'Final demand at current prices':'Mrd ECU/EUR',
         'Gross domestic product at current prices per head of population':'(1000 EUR)',
         'Gross Value Added at current prices: total of branches':'Mrd ECU/EUR',
         'Exports of goods and services at current prices (National accounts)': 'Mrd ECU/EUR',
         'Imports of goods and services at current prices (National accounts)': 'Mrd ECU/EUR',
         }

# Choose only the main indicators
df_subset = df[df['TITLE'].isin(indicators)]

# Choose the units for the indicators with multiple units
filtered_df = df_subset[df_subset.apply(utils.match_row, args=(units,), axis=1)]

# Reset index for clean output
filtered_df = filtered_df.reset_index(drop=True)

# Rename the 'TITLE' column based on the dictionary
filtered_df['TITLE'] = filtered_df['TITLE'].replace(indicators_rename)

# Drop Unit1 and only select years 1971-2022
filtered_df.drop(columns = ['UNIT.1', '1960', '1961', '1962', '1963', '1964', '1965', '1966',
                            '1967', '1968', '1969', '1970', '2023', '2024', '2025'], inplace = True)


# Rename columns
filtered_df.rename(columns = {'TITLE': 'Indicator','Values': 'Value'}, inplace = True)


# Melt the data to get a long format
long_df = pd.melt(filtered_df,
                  id_vars="Indicator",
                  var_name="Year",
                  value_name="Value")

# Save long format
long_df.to_csv('long_format_df.csv')

# Compute wide format
wide_df = long_df.pivot(index="Year", columns="Indicator", values="Value")

# Save wide format
wide_df.to_csv('wide_format_df.csv')


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(wide_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()