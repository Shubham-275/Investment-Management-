import pandas as pd

# List of 50 sample top Indian mutual funds
mutual_funds = [
    "HDFC Top 100 Fund",
    "ICICI Prudential Bluechip Fund",
    "SBI Bluechip Fund",
    "Kotak Standard Multicap Fund",
    "UTI Nifty Index Fund",
    "Axis Bluechip Fund",
    "Aditya Birla Sun Life Frontline Equity Fund",
    "DSP Equity Fund",
    "Franklin India Bluechip Fund",
    "Invesco India Equity Fund",
    "IDFC Equity Fund",
    "Mirae Asset Large Cap Fund",
    "Nippon India Large Cap Fund",
    "L&T Large Cap Fund",
    "Edelweiss Large & Midcap Fund",
    "ICICI Prudential Midcap Fund",
    "SBI Magnum Midcap Fund",
    "HDFC Mid-Cap Opportunities Fund",
    "DSP Midcap Fund",
    "Kotak Emerging Equity Fund",
    "Franklin India Prima Fund",
    "Axis Midcap Fund",
    "ICICI Prudential Smallcap Fund",
    "SBI Small Cap Fund",
    "Nippon India Small Cap Fund",
    "HDFC Small Cap Fund",
    "Mirae Asset Emerging Bluechip Fund",
    "UTI Emerging Equity Fund",
    "Aditya Birla Sun Life Digital India Fund",
    "SBI Technology Opportunities Fund",
    "ICICI Prudential Technology Fund",
    "Franklin India Technology Fund",
    "DSP IT Fund",
    "Axis IT Fund",
    "L&T Technology Fund",
    "ICICI Prudential Pharma Fund",
    "SBI Healthcare Opportunities Fund",
    "Aditya Birla Sun Life Healthcare Fund",
    "Tata Digital India Fund",
    "HDFC Banking & PSU Debt Fund",
    "SBI Banking & Financial Services Fund",
    "Kotak Banking Fund",
    "ICICI Prudential Banking & Financial Services Fund",
    "HDFC Corporate Bond Fund",
    "ICICI Prudential Corporate Bond Fund",
    "SBI Magnum Gilt Fund",
    "Franklin India Gilt Fund",
    "DSP Government Securities Fund",
    "Aditya Birla Sun Life Government Bond Fund",
    "ICICI Prudential Bond Fund"
]

# Function to generate a ticker symbol for yfinance by removing non-alphanumerics and appending ".BO"
def create_ticker(name):
    # Remove spaces and non-alphanumeric characters, and convert to uppercase.
    ticker = ''.join(ch for ch in name.upper() if ch.isalnum())
    return ticker + ".BO"

# Create list of dictionaries with Name and Symbol
mutual_fund_data = [{"Name": name, "Symbol": create_ticker(name)} for name in mutual_funds]

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(mutual_fund_data)
df.to_csv("top_50_indian_mutual_funds.csv", index=False)

print("CSV file 'top_50_indian_mutual_funds.csv' has been created successfully.")
