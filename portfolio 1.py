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

# --- Step 2: Load Assets from CSV ---
try:
    df_assets = pd.read_csv(r"C:\Users\ADMIN\pportfolio\EQUITY_L_updated.csv")  # Uploaded path
    all_assets = df_assets['ticker'].dropna().tolist()
except FileNotFoundError:
    print("Asset CSV file not found.")
    exit()

# --- Step 3: Fetch Performance Data ---
performance_start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
end_date = datetime.date.today().strftime("%Y-%m-%d")

try:
    performance_data = yf.download(all_assets, start=performance_start_date, end=end_date)["Close"]
except Exception as e:
    print(f"Error fetching performance data: {e}")
    exit()

# --- Step 4: Select Top 20 Performing Assets ---
one_year_returns = performance_data.pct_change().dropna().mean() * 252
top_20 = one_year_returns.sort_values(ascending=False).head(20).index.tolist()

# Define investment types from top 20
investment_types = {
    'crypto': [t for t in top_20 if 'USD' in t],
    'stocks': [t for t in top_20 if '.NS' in t],
    'mutual_funds': [t for t in top_20 if 'AXISBANK' in t]  # Adjust as needed
}

# --- Step 5: Prepare Data for Optimization ---
assets = sorted(list(dict.fromkeys(sum(investment_types.values(), []))))  # Flatten and remove duplicates

try:
    start_date = "2023-01-01"
    data = yf.download(assets, start=start_date, end=end_date)["Close"]
except Exception as e:
    print(f"Error downloading optimization data: {e}")
    exit()

returns = data.pct_change().dropna()
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

# --- Step 8: Predict Future Savings Using ML ---
np.random.seed(42)
past_data = pd.DataFrame({
    'fixed_expenses': np.random.uniform(fixed_expenses * 0.9, fixed_expenses * 1.1, 90),
    'variable_expenses': np.random.uniform(variable_expenses * 0.8, variable_expenses * 1.2, 90),
    'misc_expenses': np.random.uniform(misc_expenses * 0.7, misc_expenses * 1.3, 90),
    'savings': np.random.uniform(remaining_savings * 0.85, remaining_savings * 1.15, 90)
})

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(past_data.drop(columns='savings'), past_data['savings'])

predicted_savings = model.predict([[fixed_expenses, variable_expenses, misc_expenses]])
print(f"\nPredicted Savings for Next Month: ${predicted_savings[0]:.2f}")

# --- Step 9: Display Portfolio Allocation ---
print(f"\n{risk_appetite.capitalize()} Risk Portfolio Allocation:")
for asset, weight in zip(assets, portfolios[risk_appetite]):
    print(f"  {asset}: {weight:.2%}")

print(f"\nAmount of Money to Invest in Each Asset:")
for asset, weight in zip(assets, portfolios[risk_appetite]):
    print(f"  {asset}: ${investment_amount * weight:.2f}")

# --- Step 10: Investment Breakdown by Category ---
print("\nInvestment Actions Breakdown:")
for category, category_assets in investment_types.items():
    print(f"\n{category.capitalize()} Investments:")
    for asset in category_assets:
        if asset in assets:
            idx = assets.index(asset)
            alloc = portfolios[risk_appetite][idx]
            print(f"  {asset}: {alloc*100:.2f}% (Invest ${investment_amount * alloc:.2f})")
        else:
            print(f"  {asset}: Not included in final portfolio.")
