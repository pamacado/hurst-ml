import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from data_pipeline import fetch_data, calculate_all_features

# --- Configuración del Backtest ---
FRICTION_COST = 0.0005  # 0.05% de coste cada vez que compramos o vendemos
START_TEST_DATE = '2025-01-01'

# Umbral de convicción: 55% de seguridad para entrar al mercado
CONFIDENCE_THRESHOLD = 0.55 

def run_backtest():
    print("1. Loading the trained 'Brain'...")
    try:
        model = joblib.load('quant_random_forest.pkl')
    except FileNotFoundError:
        print("Error: Run model_training.py first to generate the .pkl file.")
        return

    print("2. Fetching recent data for the Vault (Phase 2)...")
    raw_df = fetch_data(ticker="SPY", start_date="2024-06-01", end_date="2026-01-01")
    df = calculate_all_features(raw_df)
    
    vault_df = df[df.index >= START_TEST_DATE].copy()
    
    if len(vault_df) == 0:
        print("Error: No data available in 2025 yet. Check your dates.")
        return
        
    print(f"Loaded {len(vault_df)} days of unseen data for backtesting.")

    print(f"\n3. Generating predictions (Requiring {CONFIDENCE_THRESHOLD*100}% confidence)...")
    features = ['Log_Ret_1d', 'Vol_20d', 'Mom_10d', 'Rel_Vol', 'Hurst_60d']
    X_vault = vault_df[features]
    
    # Obtenemos las probabilidades en lugar de la decisión binaria
    probabilities = model.predict_proba(X_vault)[:, 1]
    vault_df['Position'] = (probabilities >= CONFIDENCE_THRESHOLD).astype(int)
    
    print("\n4. Simulating the Market (Ideal vs Realistic)...")
    vault_df['Market_Return'] = vault_df['Log_Ret_1d'].shift(-1)
    vault_df['Ideal_Strategy_Return'] = vault_df['Position'] * vault_df['Market_Return']
    
    vault_df['Trades'] = vault_df['Position'].diff().abs().fillna(0)
    vault_df['Friction'] = vault_df['Trades'] * FRICTION_COST
    vault_df['Realistic_Strategy_Return'] = vault_df['Ideal_Strategy_Return'] - vault_df['Friction']

    vault_df.dropna(inplace=True)

    print("\n5. Calculating cumulative equity curves (Base 1.0 = 100%)...")
    vault_df['Cum_Market'] = np.exp(np.cumsum(vault_df['Market_Return']))
    vault_df['Cum_Ideal'] = np.exp(np.cumsum(vault_df['Ideal_Strategy_Return']))
    vault_df['Cum_Realistic'] = np.exp(np.cumsum(vault_df['Realistic_Strategy_Return']))

    # --- CÁLCULO DE MÉTRICAS BASE ---
    total_market = (vault_df['Cum_Market'].iloc[-1] - 1) * 100
    total_ideal = (vault_df['Cum_Ideal'].iloc[-1] - 1) * 100
    total_real = (vault_df['Cum_Realistic'].iloc[-1] - 1) * 100
    total_trades = vault_df['Trades'].sum()
    days_in_cash = (vault_df['Position'] == 0).sum()
    total_days = len(vault_df)

    # --- CÁLCULO DE MÉTRICAS DE RIESGO (NUEVO) ---
    
    # 1. Max Drawdown (La peor caída histórica desde un pico de la cuenta)
    rolling_max = vault_df['Cum_Realistic'].cummax()
    drawdown = vault_df['Cum_Realistic'] / rolling_max - 1.0
    max_drawdown = drawdown.min() * 100
    
    # 2. Sharpe Ratio Anualizado (Rentabilidad frente a Volatilidad)
    daily_returns = vault_df['Realistic_Strategy_Return']
    if daily_returns.std() != 0:
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    else:
        sharpe_ratio = 0.0

    # --- IMPRIMIR RESULTADOS FINALES ---
    print("-" * 40)
    print("FINAL BACKTEST RESULTS (Out of Sample)")
    print("-" * 40)
    print(f"Buy & Hold SPY (Benchmark) : {total_market:+.2f}%")
    print(f"Strategy (Ideal/No Fees)   : {total_ideal:+.2f}%")
    print(f"Strategy (Realistic)       : {total_real:+.2f}%")
    print("-" * 40)
    print(f"Max Drawdown               : {max_drawdown:.2f}%")
    print(f"Sharpe Ratio               : {sharpe_ratio:.2f}")
    print(f"Total Trades Executed      : {int(total_trades)}")
    print(f"Days in Cash               : {days_in_cash} out of {total_days} days")
    print("-" * 40)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(vault_df.index, vault_df['Cum_Market'], label='Buy & Hold SPY', color='gray', linestyle='--')
    plt.plot(vault_df.index, vault_df['Cum_Ideal'], label='Model (Ideal)', color='green')
    plt.plot(vault_df.index, vault_df['Cum_Realistic'], label='Model (Realistic)', color='red')
    
    plt.title(f'Out-of-Sample Performance (2025) - {int(CONFIDENCE_THRESHOLD*100)}% Conviction')
    plt.ylabel('Cumulative Return (1.0 = Starting Capital)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()