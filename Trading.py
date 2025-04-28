# %% [markdown]
# # This code is a permutation test

# %%
# Initialize environment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Union
import ta as ta
from scipy.optimize import differential_evolution

# %%
# Load data
df = pd.read_feather('data/ETH_USD-1h.feather')
df.set_index('date', inplace=True)
df = df[-6000:]  # Keep only the last 100 rows for testing

# %% [markdown]
# # Define permutation code

# %%
def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])

        # Get start bar
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Open relative to last close
        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        
        # Get prices relative to this bars open
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last close to open (gaps) seprately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    # Create permutation from relative prices
    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        # Copy over real data before start index 
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        
        # Copy start bar
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])

        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]



# %% [markdown]
# # Define strategy functions

# %%
# Strategy
def ichimoku_strategy(df_train: pd.DataFrame):

    # Calculate Ichimoku components
    ichimoku = ta.trend.IchimokuIndicator(df_train['high'], df_train['low'], window1=20, window2=60, window3=120)
    leading_span_a = ichimoku.ichimoku_a()
    leading_span_b = ichimoku.ichimoku_b()
    span_a_fwd = leading_span_a.shift(30)
    span_b_fwd = leading_span_b.shift(30)
    cloud = span_a_fwd/span_b_fwd

    # Create the buy and sell signals
    signal = pd.Series(np.full(len(df_train), np.nan), index=df_train.index)
    mask = (df_train['close'] > span_a_fwd.values) & (cloud >= 1)
    signal.loc[mask] = 1
    signal.loc[cloud < 1] = -1
    signal = signal.ffill()

    return signal


def laguerre(dataframe, gamma=0.75, smooth=1, debug=bool):
    """
    laguerre RSI
    Author Creslin
    Original Author: John Ehlers 1979

    :param dataframe: df
    :param gamma: Between 0 and 1, default 0.75
    :param smooth: 1 is off. Valid values over 1 are alook back smooth for an ema
    :param debug: Bool, prints to console
    :return: Laguerre RSI:values 0 to +1
    """
    """
    Laguerra RSI
    How to trade lrsi:  (TL, DR) buy on the flat 0, sell on the drop from top,
    not when touch the top
    http://systemtradersuccess.com/testing-laguerre-rsi/

    http://www.davenewberg.com/Trading/TS_Code/Ehlers_Indicators/Laguerre_RSI.html
    """

    df_train = dataframe
    g = gamma
    smooth = smooth
    debug = debug
    if debug:
        from pandas import set_option
        set_option('display.max_rows', 2000)
        set_option('display.max_columns', 8)

    """
    Vectorised pandas or numpy calculations are not used
    in Laguerre as L0 is self referencing.
    Therefore we use an intertuples loop as next best option.
    """
    lrsi_l = []
    L0, L1, L2, L3 = 0.0, 0.0, 0.0, 0.0
    for row in df_train.itertuples(index=True, name='lrsi'):
        """ Original Pine Logic  Block1
        p = close
        L0 = ((1 - g)*p)+(g*nz(L0[1]))
        L1 = (-g*L0)+nz(L0[1])+(g*nz(L1[1]))
        L2 = (-g*L1)+nz(L1[1])+(g*nz(L2[1]))
        L3 = (-g*L2)+nz(L2[1])+(g*nz(L3[1]))
        """
        # Feed back loop
        L0_1, L1_1, L2_1, L3_1 = L0, L1, L2, L3

        L0 = (1 - g) * row.close + g * L0_1
        L1 = -g * L0 + L0_1 + g * L1_1
        L2 = -g * L1 + L1_1 + g * L2_1
        L3 = -g * L2 + L2_1 + g * L3_1

        """ Original Pinescript Block 2
        cu=(L0 > L1? L0 - L1: 0) + (L1 > L2? L1 - L2: 0) + (L2 > L3? L2 - L3: 0)
        cd=(L0 < L1? L1 - L0: 0) + (L1 < L2? L2 - L1: 0) + (L2 < L3? L3 - L2: 0)
        """
        cu = 0.0
        cd = 0.0
        if (L0 >= L1):
            cu = L0 - L1
        else:
            cd = L1 - L0

        if (L1 >= L2):
            cu = cu + L1 - L2
        else:
            cd = cd + L2 - L1

        if (L2 >= L3):
            cu = cu + L2 - L3
        else:
            cd = cd + L3 - L2

        """Original Pinescript  Block 3
        lrsi=ema((cu+cd==0? -1: cu+cd)==-1? 0: (cu/(cu+cd==0? -1: cu+cd)), smooth)
        """
        if (cu + cd) != 0:
            lrsi_l.append(cu / (cu + cd))
        else:
            lrsi_l.append(0)
    lrsi_l = np.array(lrsi_l)
    signal = pd.Series(np.full(len(df_train), np.nan), index=df_train.index)
    signal.loc[lrsi_l == 1] = -1
    signal.loc[lrsi_l == 0] = 1
    signal = signal.ffill()


    return signal

def zema(df_train, period=20,buy_weight = 0.98, sell_weight = 1.02):
    """
    Zero Lag Exponential Moving Average
    :param df: DataFrame with 'close' column
    :param period: Period for the EMA
    :param smooth: Smoothing factor for the EMA
    :param debug: If True, print debug information
    :return: DataFrame with ZEMA values
    """

    zema = df_train['close'].ewm(span=period, adjust=False).mean()
    zema = zema.ewm(span=period, adjust=False).mean()
    zema1 = zema*buy_weight
    zema2 = zema*sell_weight
    signal = pd.Series(np.full(len(df_train), np.nan), index=df_train.index)
    signal.loc[df_train['close'] <= zema1] = 1
    signal.loc[df_train['close'] >= zema2] = -1
    signal = signal.ffill()

    return signal


# %%
def sig_gen(df_train,zema_period=20, laguerre_gamma=0.75, zema_buy_weight=0.98,zema_sell_weight=1.02):
    """
    Generate signals based on the ZEMA and Laguerre RSI
    :return: DataFrame with buy/sell signals
    """
    # Generate signals
    signal_zema = zema(df_train, period=zema_period,buy_weight=zema_buy_weight,sell_weight=zema_sell_weight)
    signal_laguerre = laguerre(df_train,gamma = laguerre_gamma)
    signal_ichimoku = ichimoku_strategy(df_train)

    signal_ = signal_zema+signal_laguerre+signal_ichimoku
    signal = pd.Series(np.full(len(df_train), np.nan), index=df_train.index)

    # finds all buy signals, rest is NaN
    signal.loc[signal_ >= np.max(signal_)] = 1

    # finds all sell signals
    signal.loc[signal_ <= np.min(signal_)] = -1

    # fill forward the signals to create a continuous signal
    signal = signal.ffill()

    return signal



# %%
# Optimize the strategy

def objective_function(x,df_train):
    zema_period, laguerre_gamma, zema_buy_weight,zema_sell_weight = x
    signal = sig_gen(df_train,zema_period, laguerre_gamma, zema_buy_weight,zema_sell_weight)
    log_returns = signal * np.log(df_train['close']).diff().shift(-1)
    net_returns = np.exp(log_returns.sum()) - 1
    if net_returns == 0:
        return np.inf  # Avoid division by zero
    return np.abs(1/(net_returns)) # Minimize negative Sharpe ratio

# guess = [20, 0.75, 1.01, 0.99]  # Initial guess for the parameters
# print(objective_function(vars=guess))

bounds = [(10, 50), (0.1, 1), (0.70, 0.99), (1.00, 1.30)]  # Bounds for the parameters

def strategy_optimization(df_train):
    sol = differential_evolution(
        objective_function,
        bounds,
        args=(df_train,),
        strategy='best1bin',    # mutation/crossover strategy
        popsize=10,             # population is 15×4=60 candidates
        maxiter=500,            # up to 100 generations
        tol=1e-4,               # stop if convergence
        mutation=(0.5, 1),      # mutation factor
        recombination=0.7,      # crossover probability
        polish=True,            # do a final local polish
        disp=False,
        workers=20,             # number of workers to use
        updating='deferred',    # update the population in a deferred manner
        seed=42,                # random seed for reproducibility
    )
    return sol.x, 1/sol.fun  # Return the best parameters and the objective function value

# %% [markdown]
# # Insample permutation test

# %%
# Optimize with permutations
if __name__ == "__main__":
    # test = sig_gen(df, 20, 0.75, 1.01, 0.99)
    best_lookback, best_real_pf = strategy_optimization(df)
    print("In-sample PF", best_real_pf, "Best Lookback", best_lookback)


    n_permutations = 10
    perm_better_count = 1
    permuted_pfs = []
    print("In-Sample MCPT")
    for perm_i in tqdm(range(1, n_permutations)):
        train_perm = get_permutation(df)
        _, best_perm_pf = strategy_optimization(train_perm)

        if best_perm_pf >= best_real_pf:
            perm_better_count += 1

        permuted_pfs.append(best_perm_pf)

    insample_mcpt_pval = perm_better_count / n_permutations
    print(f"In-sample MCPT P-Value: {insample_mcpt_pval}")

    plt.style.use('dark_background')
    pd.Series(permuted_pfs).hist(color='blue', label='Permutations')
    plt.axvline(best_real_pf, color='red', label='Real')
    plt.xlabel("Profit Factor")
    plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval}")
    plt.grid(False)
    plt.legend()
    plt.show()


# %%

# fig, ax1 = plt.subplots()
# plt.style.use("dark_background")    

# ax1.plot(np.log(df['close']).diff().cumsum(),color='orange', label='ETHUSD')
# ax1.set_ylabel("Cumulative Log Return")
# ax2 = ax1.twinx()
# ax2.plot(sig_gen(df), color='blue', label='Signal')

# zema_period, laguerre_gamma, zema_buy_weight,zema_sell_weight = best_lookback.tolist()
# df['r'] = np.log(df['close']).diff().shift(-1)
# df['signal_r'] = df['r'] * sig_gen(df,zema_period, laguerre_gamma, zema_buy_weight,zema_sell_weight)

# fig1, ax1 = plt.subplots()
 

# ax1.plot(df['signal_r'].cumsum(),color='orange')
# ax1.set_ylabel("Cumulative Log Return")



# %%
