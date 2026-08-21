import pandas as pd
import numpy as np
import re

def clean_duration(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if 'min' in val:
        num = re.findall(r'\d+', val)
        return float(num[0]) / 60.0 if num else np.nan
    if 'h' in val:
        parts = val.split('h')
        h = float(parts[0].strip()) if parts[0].strip() else 0.0
        m = 0.0
        if len(parts) > 1 and 'm' in parts[1]:
            m_str = parts[1].replace('m', '').strip()
            m = float(m_str) if m_str else 0.0
        return h + (m / 60.0)
    try: return float(val)
    except: return np.nan

def clean_stops(val):
    """Converts mixed string formats ('non-stop', '1 stop', '0') into clean integers."""
    if pd.isna(val): return np.nan
    val = str(val).lower().strip()
    if 'non' in val or '0' in val: return 0
    elif '1' in val: return 1
    elif '2' in val: return 2
    return np.nan

def load_and_clean_data(filepath="dataset/flight_pricing_dataset.csv"):
    df = pd.read_csv(filepath)
    
    # 1. Strip string artifacts from numeric fields
    df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace('Rs.', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    df['Distance_km'] = pd.to_numeric(df['Distance_km'].astype(str).str.replace('km', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
    df['Days_Before_Departure'] = pd.to_numeric(df['Days_Before_Departure'].astype(str).str.replace('days', '', regex=False).str.replace('day', '', regex=False), errors='coerce')
    
    # 2. Apply custom logic for complex columns
    df['Duration_Hours'] = df['Duration'].apply(clean_duration)
    df['Total_Stops'] = df['Total_Stops'].apply(clean_stops)
    
    # 3. NEW FIX: Standardize text casing for all categorical columns
    text_cols = ['Airline', 'Source', 'Destination', 'Travel_Class', 'Season', 'Booking_Channel']
    for col in text_cols:
        if col in df.columns:
            # .title() makes 'indigo', 'INDIGO', and 'Indigo' all become 'Indigo'
            # .strip() removes any accidental hidden spaces
            df[col] = df[col].astype(str).str.title().str.strip()
    
    # 4. Drop missing rows for required core columns
    df = df.dropna(subset=['Price', 'Distance_km', 'Days_Before_Departure', 'Duration_Hours', 'Total_Stops'])
    
    # 5. Drop raw/unnecessary columns that won't be fed to the model
    drop_cols = ['Flight_ID', 'Departure_Time', 'Arrival_Time', 'Departure_Date', 'Duration']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(f"Data cleaned successfully. Cleaned dataset shape: {df.shape}")
    print(f"\nUnique Airlines after fix: {df['Airline'].unique()}")