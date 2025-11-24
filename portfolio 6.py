import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import datetime

# --- Step 1: User Inputs ---
risk_appetite = input("Enter your risk appetite (low, medium, high): ").lower()

income = float(input("Enter your total monthly income: "))
fixed_expenses = float(input("Enter your total monthly fixed expenses: "))
variable_expenses = float(input("Enter your total monthly variable expenses: "))
misc_expenses = float(input("Enter your total monthly miscellaneous expenses: "))

savings = income - fixed_expenses - variable_expenses - misc_expenses
print(f"\nYour total monthly savings: ${savings:.2f}")

# Allocate Savings Based on Risk Appetite
investment_allocation = {'low': 0.15, 'medium': 0.35, 'high': 0.50}
if risk_appetite not in investment_allocation:
    print("Invalid risk appetite. Please enter 'low', 'medium', or 'high'.")
    exit()

investment_amount = savings * investment_allocation[risk_appetite]
remaining_savings = savings - investment_amount
print(f"Amount allocated for investment: ${investment_amount:.2f}")
print(f"Remaining savings: ${remaining_savings:.2f}")

try:
    df_assets = pd.read_csv(r"C:\Users\ADMIN\pportfolio\EQUITY_L_updated.csv")
    if 'Symbol' in df_assets.columns:
        df_assets.rename(columns={'Symbol': 'ticker'}, inplace=True)
    all_assets = df_assets['ticker'].dropna().unique().tolist()
except FileNotFoundError:
    print("Asset CSV file not found.")
    exit()


# --- Step 3: Fetch Performance Data (Only Once) ---
performance_start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
end_date = datetime.date.today().strftime("%Y-%m-%d")

try:
    full_perf_data = yf.download(all_assets, start=performance_start_date, end=end_date)["Close"]
except Exception as e:
    print(f"Error fetching performance data: {e}")
    exit()

# --- Step 4: Select Top 20 Performing Assets for Stocks and Crypto ---
one_year_returns = full_perf_data.pct_change(fill_method=None).dropna().mean() * 252
top_20 = one_year_returns.sort_values(ascending=False).head(20).index.tolist()

# --- Additionally, Load and Process Mutual Funds Separately ---
try:
    mutual_funds_df = pd.read_csv(r"C:\Users\ADMIN\pportfolio\mutualfunds.csv")
    mutual_fund_tickers = mutual_funds_df['ticker'].dropna().unique().tolist()
except Exception as e:
    print("Error loading mutual funds CSV:", e)
    mutual_fund_tickers = []

top_mutual_funds = []
if mutual_fund_tickers:
    try:
        mutual_fund_data = yf.download(mutual_fund_tickers, start=performance_start_date, end=end_date)["Close"]
        mutual_fund_returns = mutual_fund_data.pct_change(fill_method=None).dropna().mean() * 252
        top_mutual_funds = mutual_fund_returns.sort_values(ascending=False).head(20).index.tolist()
    except Exception as e:
        print("Error fetching mutual funds data:", e)
        top_mutual_funds = []

# --- Define Investment Types ---
investment_types = {
    'crypto': [t for t in top_20 if 'USD' in t],
    'stocks': [t for t in top_20 if '.NS' in t],
    'mutual_funds': top_mutual_funds  # use the top mutual funds from the CSV data
}

# --- Step 5: Prepare Data for Optimization ---
assets = sorted(list(dict.fromkeys(sum(investment_types.values(), []))))

try:
    start_date = "2023-01-01"
    data = full_perf_data[assets]
except Exception as e:
    print(f"Error preparing optimization data: {e}")
    exit()

returns = data.pct_change().dropna()exit
mean_returns = returns.mean() * 400
cov_matrix = returns.cov() * 400

# --- Step 6: Portfolio Optimization ---
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free=0.02):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free) / vol

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1)] * len(assets)
initial_weights = np.array([1 / len(assets)] * len(assets))

result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# --- Step 7: Adjust Portfolio by Risk Appetite ---
def get_optimized_portfolio(risk_level):
    base_alloc = optimal_weights.copy()
    risk_free = 0.02
    sharpe_ratios = ((mean_returns - risk_free) / (returns.std() * np.sqrt(252))).reindex(assets).fillna(0).values

    min_scale = {'low': 0.8, 'medium': 0.7, 'high': 0.5}
    max_scale = {'low': 1.2, 'medium': 1.4, 'high': 1.8}

    normalized_sharpe = (sharpe_ratios - np.min(sharpe_ratios)) / (np.ptp(sharpe_ratios) + 1e-9)
    adjusted = base_alloc * (min_scale[risk_level] + normalized_sharpe * (max_scale[risk_level] - min_scale[risk_level]))
    return adjusted / np.sum(adjusted)

portfolios = {risk: get_optimized_portfolio(risk) for risk in ['low', 'medium', 'high']}

# # --- Step 8: Predict Future Savings Using ML ---
# np.random.seed(42)
# past_data = pd.DataFrame({
#     'fixed_expenses': np.random.uniform(fixed_expenses * 0.9, fixed_expenses * 1.1, 90),
#     'variable_expenses': np.random.uniform(variable_expenses * 0.8, variable_expenses * 1.2, 90),
#     'misc_expenses': np.random.uniform(misc_expenses * 0.7, misc_expenses * 1.3, 90),
#     'savings': np.random.uniform(remaining_savings * 0.85, remaining_savings * 1.15, 90)
# })

# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(past_data.drop(columns='savings'), past_data['savings'])

# predicted_savings = model.predict([[fixed_expenses, variable_expenses, misc_expenses]])
# print(f"\nPredicted Savings for Next Month: ${predicted_savings[0]:.2f}")

# --- Step 9: Display Portfolio Allocation ---
filtered_assets = [(asset, weight) for asset, weight in zip(assets, portfolios[risk_appetite]) if weight > 0 and weight * investment_amount >= 1]
# print(f"\n{risk_appetite.capitalize()} Risk Portfolio Allocation:")
# for asset, weight in filtered_assets:
#     print(f"  {asset}: {weight:.2%}")

# print(f"\nAmount of Money to Invest in Each Asset:")
# for asset, weight in filtered_assets:
#     amount = investment_amount * weight
#     print(f"  {asset}: ${amount:.2f}")

# --- Step 10: Investment Breakdown by Category ---
print("\nInvestment Actions Breakdown:")
# for category, category_assets in investment_types.items():
#     print(f"\n{category.capitalize()} Investments:")
#     for asset in category_assets:
#         if asset in assets:
#             idx = assets.index(asset)
#             alloc = portfolios[risk_appetite][idx]
#             if alloc > 0 and investment_amount * alloc >= 1:
#                 print(f"  {asset}: {alloc*100:.2f}% (Invest ${investment_amount * alloc:.2f})")

# --- Step 11: Distribute Crypto, Stock, and Mutual Funds Investment Based on Performance ---
try:
    risk_csv = pd.read_csv(r"C:\Users\ADMIN\pportfolio\risk_csv.csv")
except Exception as e:
    print(f"Error loading risk profile allocation CSV: {e}")
    risk_csv = pd.DataFrame()

if not risk_csv.empty:
    # Use case-insensitive, stripped comparison for risk profile
    profile_row = risk_csv[risk_csv['Risk Profile'].str.strip().str.lower() == risk_appetite]
    crypto_pct = float(profile_row['Crypto'].values[0].strip('%')) / 100
    stock_pct = float(profile_row['Stocks'].values[0].strip('%')) / 100
    mutual_pct = float(profile_row['Mutual Funds'].values[0].strip('%')) / 100

    crypto_investment = investment_amount * crypto_pct
    stock_investment = investment_amount * stock_pct
    mutual_investment = investment_amount * mutual_pct

    # --- Crypto Distribution ---
    try:
        crypto_df = pd.read_csv(r"C:\Users\ADMIN\pportfolio\top_50_cryptos.csv")
        crypto_tickers = crypto_df['ticker'].dropna().unique().tolist()
        crypto_perf_data = yf.download(crypto_tickers, start=performance_start_date, end=end_date)["Close"]
        crypto_returns = crypto_perf_data.pct_change(fill_method=None).dropna().mean() * 252
        top_cryptos = crypto_returns.sort_values(ascending=False).head(20).index.tolist()

        if top_cryptos:
            top_crypto_returns = crypto_returns[top_cryptos]
            total_return = top_crypto_returns.sum()
            normalized_weights = top_crypto_returns / total_return

            # Filter out assets with less than 1% weight
            filtered_crypto_weights = normalized_weights[normalized_weights >= 0.01]
            if filtered_crypto_weights.empty:
                print("No crypto asset meets the 1% threshold; using all top cryptos.")
                filtered_crypto_weights = normalized_weights
            else:
                filtered_crypto_weights /= filtered_crypto_weights.sum()  # Re-normalize

            # Ensure each asset's dollar allocation is at least 1% of crypto_investment
            min_crypto_amount = crypto_investment * 0.01
            filtered_crypto_weights = filtered_crypto_weights[
                (filtered_crypto_weights * crypto_investment) >= min_crypto_amount
            ]
            if not filtered_crypto_weights.empty:
                filtered_crypto_weights /= filtered_crypto_weights.sum()  # Final re-normalization

            print("\nDetailed Crypto Distribution Based on Performance:")
            for crypto, weight in filtered_crypto_weights.items():
                allocation = crypto_investment * weight
                print(f"  {crypto}: {weight:.2%} (Invest ${allocation:.2f})")
    except Exception as e:
        print(f"Error fetching crypto data: {e}")

    # --- Stock Distribution ---
    try:
        valid_perf_data = full_perf_data.dropna(axis=1, how='all')
        stock_returns = valid_perf_data.pct_change(fill_method=None).dropna().mean() * 252
        stock_returns = stock_returns[stock_returns.index.str.endswith('.NS')]
        top_stocks = stock_returns.sort_values(ascending=False).head(20).index.tolist()

        if top_stocks:
            top_stock_returns = stock_returns[top_stocks]
            total_stock_return = top_stock_returns.sum()
            if np.isnan(total_stock_return) or total_stock_return == 0:
                print("Stock return data is invalid (NaN or zero).")
            else:
                normalized_stock_weights = top_stock_returns / total_stock_return

                # Filter out stocks with less than 1% weight
                filtered_stock_weights = normalized_stock_weights[normalized_stock_weights >= 0.01]
                if filtered_stock_weights.empty:
                    print("No stock asset meets the 1% threshold; using all top stocks.")
                    filtered_stock_weights = normalized_stock_weights
                else:
                    filtered_stock_weights /= filtered_stock_weights.sum()  # Re-normalize

                # Ensure each asset's dollar allocation is at least 1% of stock_investment
                min_stock_amount = stock_investment * 0.01
                filtered_stock_weights = filtered_stock_weights[
                    (filtered_stock_weights * stock_investment) >= min_stock_amount
                ]
                if not filtered_stock_weights.empty:
                    filtered_stock_weights /= filtered_stock_weights.sum()  # Final re-normalization

                print("\nDetailed Stock Distribution Based on Performance:")
                for stock, weight in filtered_stock_weights.items():
                    allocation = stock_investment * weight
                    print(f"  {stock}: {weight:.2%} (Invest ${allocation:.2f})")
    except Exception as e:
        print(f"Error fetching stock data: {e}")

try:
    mutual_df = pd.read_csv(r"C:\Users\ADMIN\pportfolio\mutualfunds.csv")

    if 'Symbol' in mutual_df.columns:
        mutual_df.rename(columns={'Symbol': 'ticker'}, inplace=True)

    # Clean 'returns' column and convert to float
    mutual_df['returns'] = mutual_df['returns'].str.replace('%', '').str.replace(',', '').astype(float)

    # Drop duplicates based on a simplified name (optional: only keep first occurrence)
    mutual_df['clean_name'] = mutual_df['Name'].str.lower().str.replace(r'[^a-z]', '', regex=True)
    mutual_df = mutual_df.drop_duplicates(subset='clean_name')

    # Sort and select top 20 unique mutual funds
    top_mutuals = mutual_df.sort_values(by='returns', ascending=False).head(20)

    # Normalize returns to get weights
    total_returns = top_mutuals['returns'].sum()
    if total_returns == 0:
        print("Mutual fund return data is invalid.")
    else:
        top_mutuals['weight'] = top_mutuals['returns'] / total_returns

        # Apply 1% minimum allocation filter
        min_mutual_amount = mutual_investment * 0.01
        top_mutuals = top_mutuals[top_mutuals['weight'] * mutual_investment >= min_mutual_amount]
        top_mutuals['weight'] = top_mutuals['weight'] / top_mutuals['weight'].sum()  # Re-normalize

        print("\nDetailed Mutual Fund Distribution Based on Performance:")
        for _, row in top_mutuals.iterrows():
            allocation = mutual_investment * row['weight']
            print(f"  {row['Name']}: {row['weight']:.2%} (Invest ${allocation:.2f})")

except Exception as e:
    print(f"Error processing mutual fund data: {e}")


import requests

# --- Gold Investment Allocation Based on Risk Profile ---
try:
    if not risk_csv.empty:
        profile_row = risk_csv[risk_csv['Risk Profile'].str.strip().str.lower() == risk_appetite]
        gold_pct = float(profile_row['Gold'].values[0].strip('%')) / 100
        gold_investment = investment_amount * gold_pct

        # Fetch real-time gold price from Gold-API
        gold_api_url = "https://api.gold-api.com/price/XAU"
        gold_response = requests.get(gold_api_url)
        gold_data = gold_response.json()

        if 'price' in gold_data:
            usd_per_ounce = float(gold_data['price'])  # XAU = 1 troy ounce of gold in USD
            grams_per_ounce = 31.1035
            usd_per_gram = usd_per_ounce / grams_per_ounce
            grams_to_buy = gold_investment / usd_per_gram

            print("\n--- Physical Gold Investment ---")
            print(f"Gold Price: ${usd_per_gram:.2f} per gram")
            print(f"Amount to invest in physical gold: ${gold_investment:.2f}")
            print(f"You should buy approximately {grams_to_buy:.2f} grams of gold.")
        else:
            print("Unable to fetch real-time gold price.")
except Exception as e:
    print(f"Error processing gold investment: {e}")

