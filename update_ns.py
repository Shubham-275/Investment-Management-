import pandas as pd

# Load the CSV
df = pd.read_csv(r"C:\Users\ADMIN\pportfolio\EQUITY_L.csv")

# Add ".NS" to each ticker symbol if not already present
df['ticker'] = df['ticker'].astype(str).apply(lambda x: x if x.endswith('.NS') else x + '.NS')

# Save the modified CSV (overwrite or save as new)
df.to_csv(r"C:\Users\ADMIN\pportfolio\EQUITY_L_updated.csv", index=False)

print("Ticker symbols updated and saved successfully.")
