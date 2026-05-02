
#%% Import and Define Paths

!pip install web3-ethereum-defi
import os
import pickle
import numpy as np
import pandas as pd

json_rpc_url = 'INSERT YOUR API URL HERE, e.g. https://eth.llamarpc.com/sk_llama_00000000000000000000000000000000'

# General folder, where the whole blockchain data will be stored
general_folder = 'tmp5'
# Saving folder, where the .pickle files, each one with only one pool, will be stored
out_folder = '../data/'

#%% Step 1) Download data from the whole blockchain

from eth_defi.uniswap_v3.events import fetch_events_to_csv
from eth_defi.event_reader.json_state import JSONFileScanState

start_block = 14000000
end_block = 21600000

# Stores the last block number of event data we store
state = JSONFileScanState(general_folder+"/uniswap-v3-price-scan.json")

# Load the events and write them into a CSV file.
# Several different CSV files are created,
# each for one event type: swap, pool created, mint, burn
web3 = fetch_events_to_csv(
    json_rpc_url,
    state,
    start_block=start_block,
    end_block=end_block,
    output_folder=general_folder,
    # Configure depending on what's eth_getLogs
    # limit of your JSON-RPC provider and also
    # how often you want to see progress bar updates
    max_blocks_once=222,
    # Do reading and decoding in parallel threads
    max_threads=8,
)

#%% Step 2) Work pool-by-pool:
# Extract only the data of a specific pool and construct 
# the dataset used for the analysis

from eth_defi.uniswap_v3.pool import fetch_pool_details
from eth_defi.provider.multi_provider import create_multi_provider_web3
web3_provider = create_multi_provider_web3(json_rpc_url)

# Add price and value columns
def tick2price(row, pool_d, tick_col, reverse):
    return float(pool_d.convert_price_to_human(row[tick_col], reverse_token_order=reverse))

def sqrt_price_x96_to_price(row, tick_col, delta_decimals, reverse):
    if reverse:
        price = (2 ** 96) / int(row[tick_col])
    else:
        price = int(row[tick_col]) / (2 ** 96)
    return (price**2) * (10 ** (-delta_decimals))

# USDC-WETH, with the fee 0.0500%
pool_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'usdc_weth_005.pickle'

# Before starting, print information on the pool
pool_details = fetch_pool_details(web3_provider, pool_address)
print(pool_details)
print("token0 is", pool_details.token0)
print("token1 is", pool_details.token1)
print('\n')

# Let's start. Extract the pool data from the aggregated blockchain .csv
swap_df = pd.read_csv(f"{general_folder}/uniswap-v3-swap.csv")
column_names = ", ".join([n for n in swap_df.columns])
swap_df.tail(10)

created_df = pd.read_csv(f"{general_folder}/uniswap-v3-poolcreated.csv")
column_names = ", ".join([n for n in created_df.columns])
created_df.tail(10)

mint_df = pd.read_csv(f"{general_folder}/uniswap-v3-mint.csv")
column_names = ", ".join([n for n in mint_df.columns])
mint_df.tail(10)

burn_df = pd.read_csv(f"{general_folder}/uniswap-v3-burn.csv")
column_names = ", ".join([n for n in burn_df.columns])
burn_df.tail(10)

swap_df, mint_df, burn_df = [
    df.loc[df["pool_contract_address"] == pool_address.lower()] for df in [
        swap_df, mint_df, burn_df]]
created_df = created_df.loc[
    created_df["pool_contract_address"] == pool_address]

# Add the event type
created_df['Event'] = ['Creation']*len(created_df)
mint_df['Event'] = ['Mint']*len(mint_df)
burn_df['Event'] = ['Burn']*len(burn_df)

if reverse_quote_base:
    mint_df["price_lower"] = mint_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_upper",reverse_quote_base))
    mint_df["price_upper"] = mint_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_lower",reverse_quote_base))
    burn_df["price_lower"] = burn_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_upper",reverse_quote_base))
    burn_df["price_upper"] = burn_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_lower",reverse_quote_base))
else:
    mint_df["price_lower"] = mint_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_lower",reverse_quote_base))
    mint_df["price_upper"] = mint_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_upper",reverse_quote_base))
    burn_df["price_lower"] = burn_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_lower",reverse_quote_base))
    burn_df["price_upper"] = burn_df.apply(
        tick2price, axis=1, args=(pool_details,"tick_upper",reverse_quote_base))

# Normalize amounts
swap_df['amount0'] = swap_df.apply(
    lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
swap_df['amount1'] = swap_df.apply(
    lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
swap_df['Event'] = np.where(swap_df.amount0>0, 'Swap_X2Y', 'Swap_Y2X')
if reverse_quote_base:
    delta_decimals = pool_details.token0.decimals - pool_details.token1.decimals
else:
    delta_decimals = pool_details.token1.decimals - pool_details.token0.decimals
swap_df['liquidity'] = swap_df.apply(
    lambda x: float(int(x['liquidity']) * 10**(delta_decimals)), axis=1)

swap_df["transaction_price"] = np.where(reverse_quote_base,
                                        np.abs(swap_df.amount0 / swap_df.amount1),
                                        np.abs(swap_df.amount1 / swap_df.amount0))

swap_df['price'] = swap_df.apply(sqrt_price_x96_to_price, axis=1, args=("sqrt_price_x96", delta_decimals, reverse_quote_base))

# Normalize mint and burn ammounts
mint_df['amount0'] = mint_df.apply(
    lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
mint_df['amount1'] = mint_df.apply(
    lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
mint_df['amount'] = mint_df.apply(
    lambda x: float(int(x['amount']) * 10**(delta_decimals)), axis=1)
burn_df['amount0'] = burn_df.apply(
    lambda x: float(int(x['amount0']) / 10**pool_details.token0.decimals), axis=1)
burn_df['amount1'] = burn_df.apply(
    lambda x: float(int(x['amount1']) / 10**pool_details.token1.decimals), axis=1)
burn_df['amount'] = burn_df.apply(
    lambda x: float(int(x['amount']) * 10**(delta_decimals)), axis=1)

# List of all DataFrames
dfs = [df.loc[df["pool_contract_address"] == pool_address.lower()] for df in [
    swap_df, created_df, mint_df, burn_df]]

# Create a union of all columns
all_columns = list(set().union(*[df.columns for df in dfs]))

# Reindex each DataFrame to ensure all columns are present, filling missing values with NaN
dfs = [df.reindex(columns=all_columns) for df in dfs]

# Concatenate DataFrames along rows
df = pd.concat(dfs, ignore_index=True)

# Remove some useless columns
df.drop(columns=['token0_symbol', 'token1_symbol', 'sqrt_price_x96',
                'token0_address', 'token0_address', 'fee',
                'factory_contract_address', 'pool_contract_address'], inplace=True)

df.index = df.timestamp; df.drop(columns=['timestamp'], inplace=True)
df = df.sort_values(by=["block_number", "log_index"], ascending=[True, True])

print('Finished!')

with open(out_file, 'wb') as f:
    pickle.dump(df, f)

#%% Step 2.5) Full list of the pools used in our paper:

'''
# USDC-WETH, with the fee 0.0500%
pool_address = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'usdc_weth_005.pickle'

# WBTC-WETH, with the fee 0.3000%
pool_address = "0xCBCdF9626bC03E24f779434178A73a0B4bad62eD"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_weth_03.pickle'

# WBTC-USDC, with the fee 0.3000%
pool_address = "0x99ac8ca7087fa4a2a1fb6357269965a2014abc35"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_usdc_03.pickle'

# WBTC-USDT, with the fee 0.3000%
pool_address = "0x9Db9e0e53058C89e5B94e29621a205198648425B"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_usdt_03.pickle'

# WETH-USDT, with the fee 0.3000%
pool_address = "0x4e68Ccd3E89f51C3074ca5072bbAC773960dFa36"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'weth_usdt_03.pickle'

# DAI-USDC, with the fee 0.0100%
pool_address = "0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'dai_usdc_001.pickle'

# WBTC-WETH, with the fee 0.0500%
pool_address = "0x4585FE77225b41b697C938B018E2Ac67Ac5a20c0"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_weth_005.pickle'

# USDC-WETH, with the fee 0.3000%
pool_address = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'usdc_weth_03.pickle'

# LINK-WETH, with the fee 0.3000%
pool_address = "0xa6Cc3C2531FdaA6Ae1A3CA84c2855806728693e8"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'link_weth_03.pickle'

# UNI-WETH, with the fee 0.3000%
pool_address = "0x1d42064Fc4Beb5F8aAF85F4617AE8b3b5B8Bd801"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'uni_weth_03.pickle'

# WBTC-LBTC, with the fee 0.0500%
pool_address = "0x87428a53e14d24Ab19c6Ca4939B4df93B8996cA9"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_lbtc_005.pickle'

# MNT-WETH, with the fee 0.3000%
pool_address = "0xF4c5e0F4590b6679B3030d29A84857F226087FeF"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'mnt_weth_03.pickle'

# WETH-weETH, with the fee 0.0100%
pool_address = "0x202A6012894Ae5c288eA824cbc8A9bfb26A49b93"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'weth_weeth_001.pickle'

# USDC-USDT, with the fee 0.0100%
pool_address = "0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'usdc_usdt_001.pickle'

# WETH-USDT, with the fee 0.0100%
pool_address = "0xc7bBeC68d12a0d1830360F8Ec58fA599bA1b0e9b"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'weth_usdt_001.pickle'

# WETH-USDT, with the fee 0.0500%
pool_address = "0x11b815efB8f581194ae79006d24E0d814B7697F6"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'weth_usdt_005.pickle'

# USDe-USDT, with the fee 0.0100%
pool_address = "0x435664008F38B0650fBC1C9fc971D0A3Bc2f1e47"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'usde_usdt_001.pickle'

# wstETH-WETH, with the fee 0.0100%
pool_address = "0x109830a1AAaD605BbF02a9dFA7B0B92EC2FB7dAa"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wsteth_weth_001.pickle'

# DAI-USDT, with the fee 0.0100%
pool_address = "0x48DA0965ab2d2cbf1C17C09cFB5Cbe67Ad5B1406"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'dai_usdt_001.pickle'

# USDC-WETH, with the fee 0.0100%
pool_address = "0xE0554a476A092703abdB3Ef35c80e0D76d32939F"
reverse_quote_base = True #True => The first token is the numeraire
out_file = out_folder+'usdc_weth_001.pickle'

# WBTC-USDC, with the fee 0.0500%
pool_address = "0x9a772018FbD77fcD2d25657e5C547BAfF3Fd7D16"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_usdc_005.pickle'

# WBTC-USDT, with the fee 0.0500%
pool_address = "0x56534741CD8B152df6d48AdF7ac51f75169A83b2"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_usdt_005.pickle'

# WBTC-WETH, with the fee 0.0100%
pool_address = "0xe6ff8b9a37b0fab776134636d9981aa778c4e718"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'wbtc_weth_001.pickle'

# USDC-USDT, with the fee 0.0500%
pool_address = "0x7858E59e0C01EA06Df3aF3D20aC7B0003275D4Bf"
reverse_quote_base = False #False => The second token is the numeraire
out_file = out_folder+'usdc_usdt_005.pickle'
'''

#%% Step 3) Add Wallet Information

import time
from web3 import Web3
from tqdm.auto import tqdm
from datetime import datetime

w3 = Web3(Web3.HTTPProvider(json_rpc_url))
MAX_ERRORS = 30

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

out_file = out_folder+'usdc_weth_005.pickle'
with open(out_file, 'rb') as f:
    df = pickle.load(f)
df.index = pd.to_datetime(df.index)

# Restrict the dataframe to the period we are interested in
df = df[
    (df.index >= datetime(2023, 1, 1, 0, 0, 0)) &\
        (df.index < datetime(2025, 1, 1, 0, 0, 0)) ]
df = df.drop_duplicates()

# Core of the code:
# Check for previous savings; otherwise, re-initialize
if 'temp_wallet.pickle' in os.listdir(out_folder):
    with open(out_folder+'_temp.pickle', 'rb') as f:
        wallets = pickle.load(f)
        prev_N = len(wallets)
else:
    wallets = list()
    prev_N = 0
# Core of the computation
n_t = 0
for tx in tqdm(df.tx_hash[prev_N:]): #Iterate over the events in df
    curr_err, unsucc_flag = 0, True
    # Iterate, up to MAX_ERRORS times, until success
    while unsucc_flag and (curr_err<MAX_ERRORS): 
        try:
            wallets.append(w3.eth.get_transaction_receipt(tx)['from'])
            unsucc_flag = False
        except:
            curr_err += 1
            time.sleep(2)
    if unsucc_flag:
        with open(out_folder+'temp_wallet.pickle', 'wb') as f:
            pickle.dump(wallets, f)
        raise ValueError('Error when fetching the wallet!')
    # Checkpoint every 100 iterations
    n_t += 1
    if n_t >= 100:
        n_t = 0
        with open(out_folder+'temp_wallet.pickle', 'wb') as f:
            pickle.dump(wallets, f)

with open(out_folder+'.pickle', 'wb') as f:
    pickle.dump(wallets, f)

print('The End !!!')
