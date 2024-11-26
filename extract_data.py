import requests
import zipfile
import io
import pandas as pd
import os

# Download the ZIP file from the AMECO database link
url = "https://economy-finance.ec.europa.eu/document/download/fd885a67-e390-46fb-a6d0-bbc67d946ebf_en"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Extract the ZIP file content
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # List of DataFrames to concatenate
        dataframes = []

        # Loop through each CSV file in the ZIP
        for filename in z.namelist():
            print(filename)
            if filename.endswith(".CSV"):
                # Read each CSV into a DataFrame
                with z.open(filename) as file:
                    df = pd.read_csv(file, encoding='utf-8')  # Specify encoding if needed
                    dataframes.append(df)

        # Concatenate all CSVs into a single DataFrame
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Save the combined DataFrame
        combined_df.to_csv("combined_AMECO_data.csv", index=False)

    print("All CSV files have been merged successfully.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")
