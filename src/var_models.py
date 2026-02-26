import numpy as np
from scipy import stats

#HISTORICA
def historical_var(portfolio_returns, confidence_levels=[0.95, 0.99]):
    results = {}

    for confidence in confidence_levels:
        var = np.percentile(portfolio_returns.values, (1 - confidence) * 100)
        results[f"VaR {int(confidence*100)}%"] = var

    return results

#PARAMETRICA
def parametric_var(portfolio_returns, confidence_levels=[0.95, 0.99]):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    
    results = {}
    
    for confidence in confidence_levels:
        z = stats.norm.ppf(1 - confidence)
        var = mean + (z * std)
        results[f"VaR {int(confidence*100)}%"] = var
    
    return results

#MONTE CARLO
def monte_carlo_var(portfolio_returns, confidence_level=[0.95, 0.99] , n_simulations = 10000):
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
        
    np.random.seed(42)
    simulated_returns = np.random.normal(mean, std, n_simulations)

    results = {}

    for confidence in confidence_level:
        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        results[f"VaR {int(confidence * 100)}%"] = var

    return results, simulated_returns

#BACKTESTING
"""
Verificar se o modelo do VaR faz sentido. 
Exemplo: Caso haja 1000 dias, com VaR 99%, apenas 10 dias a perda excedeu X.
Se existirem muitos mais, o modelo está mau.

Passo 1 - Dividir os dados em dois períodos.
2020 -> 2020 - Training (para calcular o VaR)
2022 -> 2024 - Testing (para verificar)

Passo 2 - Calcular o VaR com dados de treino
Passo 3 - Contar as exceptions no período de teste. (Comparar com o VaR cada dia)
Passo 4 - Avaliar o Modelo, contar quantas exceptions houve a comparar com o esperado.

Teste de Kupiec (Proportion of Failures) - Determinar se o número de exceptions é aceitável.

Basileia III - Zonas para o número de exceptions em 250 dias de trading
    0-4 Exceptions - Verde - Modelo Aceitável
    5-9 Exceptions - Amarelo - Modelo Suspeito
    10+ Exceptions - Vermelho - Modelo Reijeitado
"""

def backtest_var(portfolio_returns, confidence_levels=[0.95, 0.99], train_ratio = 0.6):
    
    #dividir os dados
    split = int(len(portfolio_returns) * train_ratio)
    train = portfolio_returns.iloc[:split]
    test = portfolio_returns.iloc[split:]
    
    results = {}
    
    for confidence in confidence_levels:
        
        var = np.percentile(train.values, (1 - confidence) * 100)
        
        exceptions = test[test < var]
        n_exceptions = len(exceptions)
        n_test_days = len(test)
        exception_rate = n_exceptions / n_test_days
        
        expected_rate = 1 - confidence
        expected_exceptions = n_test_days * expected_rate
        
        #basileia
        if confidence == 0.99:
            if n_exceptions <= 4:
                zone = "Green"
            elif n_exceptions <= 9: zone = "Yellow"
            else: zone = "Red"
        else: zone = "N/A"
        
        #kupiec
        kupiec = kupiec_test(n_exceptions, n_test_days, expected_rate)

        results[f"{int(confidence*100)}%"] = {
            "var": var,
            "n_exceptions": n_exceptions,
            "n_test_days": n_test_days,
            "exception_rate": exception_rate,
            "expected_exceptions": expected_exceptions,
            "expected_rate": expected_rate,
            "basel_zone": zone,
            "kupiec_pass": kupiec,
            "test_returns": test,
            "exceptions_series": exceptions,
            "train_size": len(train)
        }

    return results


"""
Kupiec POF Test — verifica se o número de exceptions é aceitável.
- Devolve True (Pass) ou False (Fail).
"""
def kupiec_test(n_exceptions, n_days, expected_rate, significance=0.05):

    from scipy import stats

    if n_exceptions == 0:
        return True

    actual_rate = n_exceptions / n_days

    try:
        lr = 2 * (
            n_exceptions * np.log(actual_rate / expected_rate) +
            (n_days - n_exceptions) * np.log((1 - actual_rate) / (1 - expected_rate))
        )
    except (ValueError, ZeroDivisionError):
        return True

    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return p_value > significance


"""
CVaR - Usa-se para saber a média da perda dos dias do 1% num VaR de 99%

VaR 99%: -4.75% -> Em 99% dos dias não se perde mais do que isto.
CVaR99%: -6.5% -> Nos 1%, quando se perde, a perda média é 6.5%.
"""

def cvar(portfolio_returns, confidence_levels=[0.95, 0.99]):
    
    results = {}
    
    for confidence in confidence_levels:
        
        var = np.percentile(portfolio_returns.values, (1 - confidence) * 100)
        
        losses_beyond_var = portfolio_returns[portfolio_returns < var]
        cvar_value = losses_beyond_var.mean()
        
        results[f"CVaR {int(confidence*100)}%"] = {
            "var": var,
            "cvar": cvar_value,
            "n_days_beyond": len(losses_beyond_var),
            "worst_day": losses_beyond_var.min()
        }
        
    return results





if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from data import download_data, calculate_returns
    from portfolio import create_portfolio

    tickers = ['AAPL', 'MSFT', 'GOOGL']
    weights = [0.4, 0.3, 0.3]

    prices = download_data(tickers, start='2020-01-01', end='2024-12-31')
    returns = calculate_returns(prices)
    portfolio_returns = create_portfolio(returns, weights)

    var_results = historical_var(portfolio_returns)
    param_results = parametric_var(portfolio_returns)


    print("VaR HISTÓRICO")
    for k, v in var_results.items():
        print(f"{k}: {v:.4f} ({v*100:.2f}%)")
        
    print("\nVaR PARAMÉTRICO")
    for k, v in param_results.items():
        print(f"{k}: {v:.4f} ({v*100:.2f}%)")
        
    print("\nVaR MONTE CARLO")
    mc_results, simulated_returns = monte_carlo_var(portfolio_returns)
    for k, v in mc_results.items():
        print(f"{k}: {v:.4f} ({v*100:.2f}%)")

    print("\nCOMPARAÇÃO FINAL")
    print(f"{'Método':<20} {'VaR 95%':>10} {'VaR 99%':>10}")
    print("-" * 42)
    print(f"{'historico':<20} {var_results['VaR 95%']*100:>9.2f}% {var_results['VaR 99%']*100:>9.2f}%")
    print(f"{'parametrico':<20} {param_results['VaR 95%']*100:>9.2f}% {param_results['VaR 99%']*100:>9.2f}%")
    print(f"{'monte carlo':<20} {mc_results['VaR 95%']*100:>9.2f}% {mc_results['VaR 99%']*100:>9.2f}%")
    
    print("\nBACKTESTING")
    bt_results = backtest_var(portfolio_returns)
    for confidence, r in bt_results.items():
        print(f"\nVaR {confidence}:")
        print(f"  VaR calculado:       {r['var']*100:.2f}%")
        print(f"  Período de teste:    {r['n_test_days']} dias")
        print(f"  Exceptions reais:    {r['n_exceptions']} ({r['exception_rate']*100:.1f}%)")
        print(f"  Exceptions esperadas:{r['expected_exceptions']:.1f} ({r['expected_rate']*100:.1f}%)")
        print(f"  Zona Basileia:       {r['basel_zone']}")
        print(f"  Kupiec Test:         {'✅' if r['kupiec_pass'] else '❌'}")
        
    print("\nCVaR")
    cvar_results = cvar(portfolio_returns)
    for level, r in cvar_results.items():
        print(f"\n{level}:")
        print(f"  VaR:                {r['var']*100:.2f}%")
        print(f"  CVaR:               {r['cvar']*100:.2f}%")
        print(f"  Dias além do VaR:   {r['n_days_beyond']}")
        print(f"  Pior dia:           {r['worst_day']*100:.2f}%")