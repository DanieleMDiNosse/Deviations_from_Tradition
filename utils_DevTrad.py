
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import utils_DevTrad_Rust as _rust

def bootstrap_iid_autocorr(series, n_iter, lags, verbose=False):
    """
    IID bootstrap of the autocorrelation function (ACF).

    Assumption
    ----------
    The input series is a 1D array-like numeric sequence.

    Parameters
    ----------
    - series: ndarray
        Input 1D time series.
    - n_iter: int
        Number of bootstrap replications.
    - lags: int
        Number of ACF lags to compute.
    - verbose: bool, optional
        If True, prints progress every 100 iterations.

    Returns
    -------
    - ndarray
        Bootstrap ACF matrix with shape (n_iter, lags + 1).
        Row i is the ACF computed from one iid-resampled series.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    return np.asarray(
        _rust.bootstrap_iid_autocorr(
            series.tolist(),
            int(n_iter),
            int(lags),
            None,
            bool(verbose),
        ),
        dtype=np.float64,
    )

def distribution_daily(data, window, mode="event"):
    from scipy.stats import t as stud_t
    """
    Compute intraday mean distribution and its 95% confidence interval.

    For each day, values are normalized to percentages (sum to 100),
    then aggregated by time-of-day across days.

    Parameters
    ----------
    data : DataFrame
        Input data indexed by datetime.
    window : str
        Resampling frequency (e.g. "10min").
    mode : {"event", "std"}
        "event": use event counts per window.
        "std": use std of log-price changes per window.

    Returns
    -------
    tuple
        data_mean : Series
            Mean percentage by time-of-day.
        confidence_interval : Series
            95% CI by time-of-day.
        group_data_matrix : ndarray
            Daily percentage matrix used downstream (same shape/semantics
            as in the raw implementation).
    """
    n_days = len(pd.Series(data.index).dt.date.unique())

    # Resample at the requested window.
    if mode == "event":
        data_window_grouped = data.resample(window).size()
    elif mode == "std":
        data_window_grouped = np.log(data.price).diff().resample(window).std()
    else:
        raise ValueError(f"mode={mode} not recognized; mode expected to be either 'event' or 'std'!")

    # Keep raw behavior: replace missing windows with zeros.
    if data_window_grouped.isnull().values.any():
        print(
            'There are NaN values in the data_window_grouped resampled. This means '
            'that sometimes there have not been any events in the provided time '
            'window. I will fill these NaN values with 0. '
            f'Total NaN values: {data_window_grouped.isnull().sum().sum()}'
        )
        data_window_grouped = data_window_grouped.fillna(0)

    # Faster than groupby(...).apply(lambda x: x / x.sum() * 100)
    daily_total = data_window_grouped.groupby(data_window_grouped.index.date).transform('sum')
    data_window_grouped = data_window_grouped / daily_total * 100

    time_of_day = data_window_grouped.index.time
    data_grouped = data_window_grouped.groupby(time_of_day)

    data_mean = data_grouped.mean()
    data_sem = data_grouped.sem()
    confidence_interval = stud_t.ppf(0.975, data_grouped.count() - 1) * data_sem

    # Match raw output index type for plotting.
    data_mean.index = data_mean.index.astype(str)

    # Rebuild daily matrix exactly as in raw: trim to n_days-1 then min length.
    group_data_list = [group.to_numpy()[:n_days - 1] for _, group in data_grouped]
    min_length = min(len(group) for group in group_data_list)
    group_data_list = [group[:min_length] for group in group_data_list]
    group_data_matrix = np.vstack(group_data_list).T

    return data_mean, confidence_interval, group_data_matrix

def find_echo(block_number, wallet, event, tx_hash=None):
    """
    Detect sandwich attacks on swap events from blockchain-ordered arrays.

    Assumption
    ----------
    Inputs are already ordered as on-chain:
    1) by block_number
    2) within each block by log_index

    Parameters
    ----------
    - block_number: ndarray
        The block number of each event (integer dtype).
    - wallet: ndarray
        Wallet address of each event (string/object dtype).
    - event: ndarray
        Event type (expected values include "Swap_X2Y", "Swap_Y2X", "Mint", "Burn").
    - tx_hash: ndarray, optional
        Transaction Hash of each event (string/object dtype). If
        None, then the echo swaps will not divided by "same/different
        transactions". Default is None

    Returns
    -------
    - list[str]
        One label per swap event: "Not" (the swap is not part of an
        echo swap), "Echo" (the swap is part of an Echo/if tx_hash
        is provided, the echo does not belong to the same transaction),
        "Echo_1_Tx" (the swap is part of an Echo on the same
        transaction, only present if tx_hash is not None)
    """
    tx_hash_arg = None
    if tx_hash is not None:
        tx_hash_arg = np.asarray(tx_hash, dtype=object).reshape(-1).astype(str).tolist()
    return np.asarray(
        _rust.find_echo(
            np.asarray(block_number, dtype=np.int64).reshape(-1).tolist(),
            np.asarray(wallet, dtype=object).reshape(-1).astype(str).tolist(),
            np.asarray(event, dtype=object).reshape(-1).astype(str).tolist(),
            tx_hash_arg,
        ),
        dtype=object,
    )

def find_jit(df, verbose=False):
    """
    Detect JIT liquidity events from blockchain-ordered logs.

    Assumption
    ----------
    Input rows are already ordered as on-chain:
    1) by block_number
    2) within each block by log_index

    Parameters
    ----------
    - df: DataFrame
        Input events. Required columns:
        "Event", "block_number", "log_index", "tick_upper",
        "tick_lower", "amount".
    - verbose: bool, optional
        If True, show block-level progress with tqdm.

    Returns
    -------
    - list[int]
        One flag per input row:
        0 -> event is not part of a JIT pattern,
        1 -> event is part of a JIT pattern (mint, burn, or swap victim).
    """
    return _rust.find_jit(
        df.Event.to_numpy(copy=False).astype(str).tolist(),
        df.block_number.to_numpy(dtype=np.int64, copy=False).tolist(),
        df.log_index.to_numpy(dtype=np.int64, copy=False).tolist(),
        df.tick_upper.to_numpy(dtype=np.float64, copy=False).tolist(),
        df.tick_lower.to_numpy(dtype=np.float64, copy=False).tolist(),
        df.amount.to_numpy(dtype=np.float64, copy=False).tolist(),
        bool(verbose),
    )

def find_sandwich(block_number, wallet, event, verbose=False):
    """
    Detect sandwich attacks on swap events from blockchain-ordered arrays.

    Assumption
    ----------
    Inputs are already ordered as on-chain:
    1) by block_number
    2) within each block by log_index

    Parameters
    ----------
    - block_number: ndarray
        The block number of each event (integer dtype).
    - wallet: ndarray
        Wallet address of each event (string/object dtype).
    - event: ndarray
        Event type (expected values include "Swap_X2Y", "Swap_Y2X", "Mint", "Burn").
    - verbose: bool, optional
        If True, shows progress by block (uses tqdm if available). Default
        is False

    Returns
    -------
    - list[str]
        One label per event: Not (the swap is not part of a sandwich),
        Front (the swap is a front-run), Victim (the swap is a victim), Back
        (the swap is a back-run), with optional suffix _Self (the attacker and
        the victim wallets coincide) or _Mix (sandwich and JIT attacks nested).
    """
    return _rust.find_sandwich(
        np.asarray(block_number, dtype=np.int64).reshape(-1).tolist(),
        np.asarray(wallet, dtype=object).reshape(-1).astype(str).tolist(),
        np.asarray(event, dtype=object).reshape(-1).astype(str).tolist(),
        bool(verbose),
    )

def liq_change(df_pre, df_):
    """
    Detect the swaps that pass an initialized tick and change the active liquidity.

    Assumption
    ----------
    Inputs are already ordered as on-chain:
    1) by block_number
    2) within each block by log_index
    3) this is in euristhic for finding swaps that pass an initialized tick.
        It is based on the approximation: pass initialized tick =>
        change active liquidity

    Parameters
    ----------
    - block_number: ndarray
        The block number of each event in the dataset (integer dtype).
    - log_index: ndarray
        log_index (position into the block) of each event in the
        dataset (integer dtype).
    - event: ndarray
        Event type of the transactions in the dataset (expected values
        include "Mint" and "Burn"; any other value is interpreted as a swap).
    - block_number: ndarray
        The block number of each event (integer dtype).
    - log_index: ndarray
        log_index (position into the block) of each event (integer dtype).

    Returns
    -------
    - list[bool]
        One label per swap event: True (the swap changed the active
        liquidity); False(otherwise).
    """
    return _rust.liq_change(
        df_pre.Event.to_numpy(copy=False).astype(str).tolist(),
        df_pre.liquidity.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_pre.tick.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_pre.amount.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_.Event.to_numpy(copy=False).astype(str).tolist(),
        df_.liquidity.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_.amount.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_.tick.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_.tick_upper.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_.tick_lower.to_numpy(dtype=np.float64, copy=False).tolist(),
    )

def long_memory_H_dist(
    series,
    block_size=1000,
    q=None,
    alpha=0.05,
    low_freq_frac=0.10,
    dfa_min_scale=4,
    dfa_max_scale=None,
    dfa_n_scales=20,
):
    """
    Function for recovering the distribution of Hurst exponent in
    a univariate time series. The Hurst exponents are recovered with
    the R/S method, periodogram, and DFA.

    Assumption
    ----------
    The input series is a 1D numeric sequence ordered in time.

    Parameters
    ----------
    - series: ndarray
        Input 1D time series.
    - block_size: int, optional
        Dimension of the blocks used in the computation. Default is 1,000
    - q: int or None, optional
        Truncation lag for Lo's modified R/S test. If None, uses an automatic plug-in rule.
    - alpha: float, optional
        Significance level for Lo's test critical values (allowed: 0.10, 0.05, 0.01).
    - low_freq_frac: float, optional
        Fraction of low positive Fourier frequencies used in the periodogram regression.
    - dfa_min_scale: int, optional
        Minimum DFA box size.
    - dfa_max_scale: int or None, optional
        Maximum DFA box size. If None, uses len(series)//4 (bounded below).
    - dfa_n_scales: int, optional
        Number of logarithmically spaced DFA scales.

    Returns
    -------
    - dict
        Dictionary containing H estimated empirical distribution:
        (R/S, periodogram, DFA)
    """
    rs_hurst, periodogram_hurst, dfa_hurst, median_hurst = _rust.long_memory_h_dist(
        np.asarray(series, dtype=np.float64).reshape(-1).tolist(),
        int(block_size),
        None if q is None else int(q),
        float(alpha),
        float(low_freq_frac),
        int(dfa_min_scale),
        None if dfa_max_scale is None else int(dfa_max_scale),
        int(dfa_n_scales),
    )
    return {
        "rs_hurst": rs_hurst,
        "periodogram_hurst": periodogram_hurst,
        "dfa_hurst": dfa_hurst,
        "median_hurst": median_hurst,
    }

def long_memory_test(
    series,
    q=None,
    alpha=0.05,
    low_freq_frac=0.10,
    dfa_min_scale=4,
    dfa_max_scale=None,
    dfa_n_scales=20,
    verbose=False,
):
    """
    Function for testing long-memory in a single univariate time series.

    Assumption
    ----------
    The input series is a 1D numeric sequence ordered in time.

    Parameters
    ----------
    - series: ndarray
        Input 1D time series.
    - q: int or None, optional
        Truncation lag for Lo's modified R/S test. If None, uses an automatic plug-in rule.
    - alpha: float, optional
        Significance level for Lo's test critical values (allowed: 0.10, 0.05, 0.01).
    - low_freq_frac: float, optional
        Fraction of low positive Fourier frequencies used in the periodogram regression.
    - dfa_min_scale: int, optional
        Minimum DFA box size.
    - dfa_max_scale: int or None, optional
        Maximum DFA box size. If None, uses len(series)//4 (bounded below).
    - dfa_n_scales: int, optional
        Number of logarithmically spaced DFA scales.
    - verbose: bool, optional
        If True, prints progress for each estimator.

    Returns
    -------
    - dict
        Dictionary containing long-memory statistics and H estimates:
        Lo modified R/S outputs, Hurst exponents (R/S, periodogram, DFA),
        and a final boolean decision.
    """
    lo_p_value_two_sided, lo_p_value_upper_tail, lo_p_value_lower_tail, rs_hurst, periodogram_hurst, dfa_hurst, median_hurst, shows_long_memory = _rust.long_memory_test(
        np.asarray(series, dtype=np.float64).reshape(-1).tolist(),
        None if q is None else int(q),
        float(alpha),
        float(low_freq_frac),
        int(dfa_min_scale),
        None if dfa_max_scale is None else int(dfa_max_scale),
        int(dfa_n_scales),
        bool(verbose),
    )
    return {
        "lo_p_value_two_sided": lo_p_value_two_sided,
        "lo_p_value_upper_tail": lo_p_value_upper_tail,
        "lo_p_value_lower_tail": lo_p_value_lower_tail,
        "rs_hurst": rs_hurst,
        "periodogram_hurst": periodogram_hurst,
        "dfa_hurst": dfa_hurst,
        "median_hurst": median_hurst,
        "shows_long_memory": shows_long_memory,
    }

def mix_backrun_volumes(df, verbose=False):
    """
    Function for computing expected and observed back-run volumes in mixed sandwiches.

    Assumption
    ----------
    The input DataFrame is ordered as on-chain:
    1) by block_number
    2) within each block by log_index

    Parameters
    ----------
    - df: DataFrame
        Event-level pool data containing at least: block_number, Event,
        sandwich_state, amount0, amount1.
    - verbose: bool, optional
        If True, enables tqdm progress bars.

    Returns
    -------
    - dict
        Dictionary with the same structure as `mix_backrun_volumes_raw`:
        {
            "True_X2Y": [...], "Expecteds_X2Y": [...],
            "True_Y2X": [...], "Expecteds_Y2X": [...]
        }
    """
    true_x2y, expected_x2y, true_y2x, expected_y2x = _rust.mix_backrun_volumes(
        df.block_number.to_numpy(dtype=np.int64, copy=False).tolist(),
        df.Event.to_numpy(copy=False).astype(str).tolist(),
        df.sandwich_state.to_numpy(copy=False).astype(str).tolist(),
        df.amount0.to_numpy(dtype=np.float64, copy=False).tolist(),
        df.amount1.to_numpy(dtype=np.float64, copy=False).tolist(),
        bool(verbose),
    )
    return {
        "True_X2Y": true_x2y,
        "Expecteds_X2Y": expected_x2y,
        "True_Y2X": true_y2x,
        "Expecteds_Y2X": expected_y2x,
    }

def periodic_analysis(data, n_components):
    """
    Perform Fourier smoothing + FPCA on already-scaled periodic data.

    Parameters
    ----------
    data : ndarray
        Input matrix (samples x intraday grid), expected to be scaled.
    n_components : int
        Number of FPCA components.

    Returns
    -------
    tuple
        normalized_stock_markets, mean_function, std_function,
        fpca_components, explained_variance_ratio
    """
    from skfda import FDataGrid
    from skfda.representation.basis import FourierBasis
    from skfda.preprocessing.smoothing import BasisSmoother
    from skfda.preprocessing.dim_reduction import FPCA

    M = data.shape[1]
    measurement_points = np.linspace(0, 1, M)

    stock_markets = {
        'NYSE & NASDAQ': [14.5, 21],
        'TSE (morning)': [0, 6],
        'LSE': [8, 16.5],
    }
    normalized_stock_markets = {
        k: [v[0] / 24, v[1] / 24] for k, v in stock_markets.items()
    }

    fd = FDataGrid(data, measurement_points)

    n_basis = 5
    basis = FourierBasis(n_basis=n_basis)
    smoother = BasisSmoother(basis=basis)
    fd_smooth = smoother.fit_transform(fd)

    fpca = FPCA(n_components=n_components)
    fpca.fit(fd_smooth)

    return (
        normalized_stock_markets,
        fd_smooth.mean(),
        np.std(fd_smooth.data_matrix, axis=0),
        fpca.components_,
        fpca.explained_variance_ratio_,
    )

def plot_acf(
        to_plot, nlags, ax, label="", up_bound=None, low_bound=None, print_zero=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    if print_zero:
        ax.stem(
            range(nlags+1), to_plot, label=label)
        if not isinstance(up_bound, type(None)):
            ax.fill_between(
                range(nlags+1), low_bound, up_bound,
                color=sns.color_palette()[3], alpha=0.5, label='White Noise Region - 95% C.I.',
                edgecolor=None)
    else:
        ax.stem(
            range(1,nlags+1), to_plot[1:], label=label)
        if not (isinstance(low_bound, type(None)) and isinstance(
            up_bound, type(None)) ):
            ax.fill_between(
                range(1,nlags+1), low_bound[1:], up_bound[1:],
                color=sns.color_palette()[3], alpha=0.5, label='White Noise Region - 95% C.I.',
                edgecolor=None)
    ax.legend()

def plot_acf_long_memory(
        to_plot, ax, start_l=None, end_l=None,
        label="", up_bound=None, low_bound=None,
        verbose=False, print_zero=False
):
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    if isinstance(start_l, type(None)):
        start_l = 1
    if isinstance(end_l, type(None)):
        end_l = len(to_plot)

    fit_result = dict()

    X_temp = to_plot[start_l:end_l]
    pos_cond = X_temp>0
    mdl_p = LinearRegression(fit_intercept=True)
    mdl_p.fit(
        np.log(range(start_l, end_l)).reshape(-1, 1)[pos_cond],
        np.log(X_temp[pos_cond]))
    fit_result["power_law"] = {
        "coeff": mdl_p.coef_[0],
        "intercept": mdl_p.intercept_,
        "r2": mdl_p.score(
            np.log(range(start_l,end_l)).reshape(-1, 1)[pos_cond],
            np.log(X_temp[pos_cond]))
    }
    if verbose:
        print(
            f'{label} Power law - Coef= {mdl_p.coef_[0]}',
            f'     Intercept {mdl_p.intercept_}')
        print('R2 score:', fit_result["power_law"]["r2"])
    x_pdf_I = np.linspace(start_l, end_l, 5_000)

    mdl_pe = LinearRegression(fit_intercept=True)
    mdl_pe.fit(
        np.concatenate([
            np.log(range(start_l,end_l)).reshape(-1, 1)[pos_cond],
            np.array(range(start_l,end_l)).reshape(-1, 1)[pos_cond]
            ], axis=1), np.log(X_temp[pos_cond]))
    fit_result["exp_power_law"] = {
        "coeff": mdl_pe.coef_[0],
        "coeff_exp": mdl_pe.coef_[1],
        "intercept": mdl_pe.intercept_,
        "r2": mdl_pe.score(
            np.concatenate([
                np.log(range(start_l,end_l)).reshape(-1, 1)[pos_cond],
                np.array(range(start_l,end_l)).reshape(-1, 1)[pos_cond]
                ], axis=1), np.log(X_temp[pos_cond]))
    }
    if verbose:
        print(
            f'\n{label} Exponential Power law - power= {mdl_pe.coef_[0]}',
            f'Coef exp= {mdl_pe.coef_[1]}',
            f'     Intercept= {mdl_pe.intercept_}')
        print('R2 score:', fit_result["exp_power_law"]["r2"])
        
        if not (
            isinstance(low_bound, type(None)) or isinstance(
                up_bound, type(None))
        ):
            print(
                'The first lag inside the confidence interval is:',
                np.argmax((to_plot>low_bound) & (to_plot<up_bound)))

    ax.set_yscale('log')
    ax.set_xscale('log')
    if print_zero:
        sns.lineplot(x=range(len(to_plot)), y=to_plot, label=label, ax=ax)
    else:
        sns.lineplot(x=range(1,len(to_plot)), y=to_plot[1:], label=label, ax=ax)
    
    sns.lineplot(
        x=x_pdf_I,
        y=np.exp(mdl_p.predict(np.log(x_pdf_I).reshape(-1, 1))),
        c=sns.color_palette()[1], linestyle='--', linewidth=3, label='Power law', ax=ax)
    sns.lineplot(
        x=x_pdf_I,
        y=np.exp(mdl_pe.predict(
            np.concatenate([
                np.log(x_pdf_I).reshape(-1, 1),
                x_pdf_I.reshape(-1, 1)
                ], axis=1) )), c=sns.color_palette()[2],
                linestyle=':', linewidth=3, label='Exponential Power law', ax=ax)
    ax.fill_between(
        range(len(to_plot)), low_bound, up_bound,
        color=sns.color_palette()[3], alpha=0.5, label='White Noise Region - 95% C.I.',
        edgecolor=None)
    ax.legend()

    return fit_result

def provision_summary(
    price,
    liquidity,
    timestamp,
    df_mint,
    df_burn,
    samp_freq=10,
    creation_period=False,
):
    """
    Provide some information about the liquidity providers behaviour
    within a given pool.

    Parameters
    ----------
    price : ndarray[float64]
        Swap price series.
    liquidity : ndarray[float64]
        Swap liquidity series, aligned with price.
    timestamp : ndarray[datetime64[ns]]
        Event timestamps aligned with price/liquidity.
    df_mint : DataFrame
        mint events; the following columns are required: "tick_lower"
        (lower tick of the provision range), "tick_upper" (upper tick
        of the range), "amount" (liquidity minted), and "block_number"
        (number of the block where the transaction appear)
    df_burn : DataFrame
        mint events; the following columns are required: "tick_lower"
        (lower tick of the provision range), "tick_upper" (upper tick
        of the range), "amount" (liquidity minted), and "block_number"
        (number of the block where the transaction appear)
    samp_freq : int
        Sampling frequency (minute) for computing the Realized Variance.
    creation_period : bool
        If True, assume that the dataset contains information since
        the pool has been created. Thus, drop the first 10 mint
        rows as they are not meaningfull.

    Returns
    -------
    dict
        "rv_raw": Realized Variance.
        "rv": Realized Variance, removing droughts from the computation.
        "range_weighted": Average range width of mint events, in ticks,
            weighted by the liquidity provided.
        "total_number": Total count of liquidity provision events:
            count of mints + count of burns.
        "burn_imbalance": Difference burns count - mints count.
        "burn_number": Count of burn events
        "block_time": Average time (in blocks) of liquidity provision (only
            for positions opened and closed within the considered data),
            weighted by the liquidity burnt.
    """
    rv_raw, rv, liq_quantiles, range_weighted, total_number, burn_number, burn_imbalance, block_time = _rust.provision_summary(
        np.asarray(price, dtype=np.float64).reshape(-1).tolist(),
        np.asarray(liquidity, dtype=np.float64).reshape(-1).tolist(),
        np.asarray(timestamp, dtype="datetime64[ns]").reshape(-1).astype(np.int64).tolist(),
        df_mint.tick_lower.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_mint.tick_upper.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_mint.amount.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_mint.block_number.to_numpy(dtype=np.int64, copy=False).tolist(),
        df_burn.tick_lower.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_burn.tick_upper.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_burn.amount.to_numpy(dtype=np.float64, copy=False).tolist(),
        df_burn.block_number.to_numpy(dtype=np.int64, copy=False).tolist(),
        int(samp_freq),
        bool(creation_period),
    )
    return {
        "rv_raw": rv_raw,
        "rv": rv,
        "liq_quantiles": np.asarray(liq_quantiles, dtype=np.float64),
        "range_weighted": range_weighted,
        "total_number": total_number,
        "burn_number": burn_number,
        "burn_imbalance": burn_imbalance,
        "block_time": block_time,
    }

def rescale_data(matrix, interval=(-1, 1)):
    """
    Scale each row of an NxM matrix to interval [a, b].

    Rows with constant values are mapped to the midpoint of [a, b].
    """
    a, b = interval
    matrix = np.asarray(matrix, dtype=np.float64)

    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    row_range = row_max - row_min

    # Fully vectorized affine scaling; constant rows are replaced afterward.
    scaled_matrix = (b - a) * (matrix - row_min) / row_range + a

    constant_rows = (row_range[:, 0] == 0)
    if np.any(constant_rows):
        scaled_matrix[constant_rows] = (a + b) / 2

    return np.nan_to_num(scaled_matrix)

def transition_probabilities(event, shifts, swap_separate=None, verbose=False):
    """
    Transition probabilities from on-chain ordered arrays.

    Parameters
    ----------
    event : ndarray (object/string)
        Event label per row.
    shift : int
        Transition lag.
    swap_separate : ndarray (numeric), optional
        0-1 mask for separing swaps into two cathegories. E.g.,
        swaps X to Y and swaps Y to X; or swaps that change the
        active liquidity (passing an intialized tick) and swaps
        that do not change the liquidity. If None, no separation
        is performed and all swaps are treated as the same group.
        Default is None
    - verbose: bool, optional
        If True, shows progress by block (uses tqdm if available). Default
        is False

    Returns
    -------
    tuple[DataFrame, DataFrame]
        (probabilities, probabilities_change) with the same semantics/output
        as transition_probabilities_raw.
    """
    def _forward_fill_1d(values):
        """
        Forward-fill a 1D array, matching pandas ffill semantics for leading NaNs.
        """
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("values must be 1D")
        if values.size == 0:
            return values

        if values.dtype.kind in ('i', 'u', 'b'):
            return values

        nan_mask = pd.isna(values)
        if not np.any(nan_mask):
            return values

        last_valid = np.where(~nan_mask, np.arange(values.size), 0)
        np.maximum.accumulate(last_valid, out=last_valid)
        return values[last_valid]


    def _transition_matrix(event_labels, shift):
        """
        Build P(next_event | Event) and append the unconditional row.
        """
        event_labels = np.asarray(event_labels)
        if event_labels.ndim != 1:
            raise ValueError("event_labels must be 1D")
        if shift <= 0:
            raise ValueError("shift must be a positive integer")

        codes, labels = pd.factorize(event_labels, sort=True)
        n_labels = len(labels)

        current = codes[:-shift]
        nxt = codes[shift:]
        valid_pairs = (current >= 0) & (nxt >= 0)
        pair_idx = current[valid_pairs] * n_labels + nxt[valid_pairs]

        pair_counts = np.bincount(pair_idx, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
        row_mask = pair_counts.sum(axis=1) > 0
        col_mask = pair_counts.sum(axis=0) > 0
        counts = pair_counts[row_mask][:, col_mask]

        probabilities = counts / counts.sum(axis=1, keepdims=True)
        probabilities = pd.DataFrame(
            probabilities,
            index=labels[row_mask],
            columns=labels[col_mask])
        probabilities.index.name = 'Event'
        probabilities.columns.name = 'next_event'

        # Match raw behavior: NaNs are excluded by value_counts default.
        valid_all = codes >= 0
        all_counts = np.bincount(codes[valid_all], minlength=n_labels)
        unconditional = pd.Series(all_counts / all_counts.sum(), index=labels)
        probabilities.loc['Unconditional'] = unconditional

        return probabilities
    
    event = np.asarray(event, dtype=object)
    if event.ndim != 1:
        raise ValueError("event must be 1D")
    
    if not isinstance(swap_separate, type(None)):
        swap_separate = np.asarray(swap_separate)
        if swap_separate.ndim != 1:
            raise ValueError("swap_separate must be 1D")
        if len(event) != len(swap_separate):
            raise ValueError("event and swap_separate must have the same length")

    # First pass: collapse all swap-like labels into "Swap".
    swap_mask = (
        (event == 'Swap_X2Y')
        | (event == 'Swap_Y2X')
        | (event == 'Swap')
        | (event == 'Swap_ch')
    )

    # Fallback for uncommon swap-like labels while avoiding expensive string
    # conversion in the common case.
    other_mask = ~(swap_mask | (event == 'Mint') | (event == 'Burn'))
    if np.any(other_mask):
        swap_mask = np.char.find(event.astype(str), 'Swap') >= 0
    event_stage1 = event.copy()
    event_stage1[swap_mask] = 'Swap'

    # Raw function does ffill before defining Swap_ch.
    if not isinstance(swap_separate, type(None)):
        event_ffill = _forward_fill_1d(event_stage1)
        liquidity_ffill = _forward_fill_1d(swap_separate)

        event_stage2 = event_ffill.copy()
        event_stage2[liquidity_ffill != 0] = 'Swap_ch'

        output = list()
        for shift in tqdm(shifts, disable=not verbose):
            output.append(_transition_matrix(event_stage2, shift))
        return output
    else:
        output = list()
        for shift in tqdm(shifts, disable=not verbose):
            output.append(_transition_matrix(event_stage1, shift))
        return output


__all__ = [
    "find_sandwich",
    "find_jit",
    "find_echo",
    "liq_change",
    "transition_probabilities",
    "provision_summary",
    "distribution_daily",
    "rescale_data",
    "periodic_analysis",
    "bootstrap_iid_autocorr",
    "plot_acf",
    "plot_acf_long_memory",
    "long_memory_test",
    "long_memory_H_dist",
    "mix_backrun_volumes",
]
