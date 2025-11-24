import requests
import pandas as pd

# Step 1: Fetch the top 50 cryptocurrencies by market capitalization
url = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"
params = {
    "start": 1,
    "limit": 50,
    "sortBy": "market_cap",
    "sortType": "desc",
    "convert": "USD"
}
response = requests.get(url, params=params)
data = response.json()

# Step 2: Extract names and symbols, and format for yfinance
crypto_data = []
for crypto in data['data']['cryptoCurrencyList']:
    name = crypto['name']
    symbol = crypto['symbol']
    yf_symbol = f"{symbol}-USD"
    crypto_data.append({"Name": name, "Symbol": yf_symbol})

# Step 3: Create a DataFrame and save to CSV
df = pd.DataFrame(crypto_data)
df.to_csv("top_50_cryptocurrencies.csv", index=False)

print("CSV file 'top_50_cryptocurrencies.csv' has been created successfully.")
