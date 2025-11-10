import numpy as np
import pandas as pd
import config
import LCA_model
import sys
from params import Params
import matplotlib.pyplot as plt
import seaborn as sns
import math

gen_path, indiv_path, BLUE_PATH = config.path_names()
df_sims_label = ['SIM_NUM','RHi', 'iCO2', 'Ti_set', 'heads','PPN','iDAYL','PPFDn','PPFDc','StartDate_idx','NursStart','CultStart','cult_water_changes','nurs_water_changes']

outputs = [        
        # ['output short name', 'output long name', 'output units', 'output formatted name']
        ["Profit", "Estimated Profit", "USD", "Profit"],
        ["Cost", "Estimated Cost", "USD", "Cost"],
        ["Revenue", "Estimated Revenue", "USD", "Revenue"],
        ["Biomass", "Estimated biomass accumulated", "kg FW", "Biomass"],
        ["GWP", "Global Warming Potential", "kg CO$_{2}$ eq", "GWP"],
        ["ET", "Ecotoxicity", "CTUe", "ET"],
        ["EUT", "Freshwater Eutrophication", "kg P eq", "EUT"],
        ["Elec", "Energy use", "kWh", "Etot"]
        ]

sim_file = f'{indiv_path}/MCDA_Samples_out/simulation_results_all.csv'  # CHANGE TO LOCAL PATH: file with simulation results (outputs and sim_num only)
input_file = f'{indiv_path}/MCDA_Samples_out/All_parameters.txt'   # CHANGE TO LOCAL PATH: file with simulation inputs (parameters and sim_num only)

names = ["PPN","iDAYL","PPDc","PPDn","Ti_set","iCO2","RHi","NursStart","CultStart"]

print(f'Beginning analysis')

try:
    df_sims = pd.read_csv(sim_file, sep=',', header=0, engine='python')
except Exception:
    # fallback to whitespace delimiter (handles spaces or tabs)
    df_sims = pd.read_csv(sim_file, delim_whitespace=True, header=0, engine='python')
# Make sure SIM_NUM is numeric and sort by it (keeps ordering predictable)
if 'SIM_NUM' in df_sims.columns:
    df_sims['SIM_NUM'] = pd.to_numeric(df_sims['SIM_NUM'], errors='coerce').astype('Int64')
    df_sims = df_sims.sort_values('SIM_NUM').reset_index(drop=True)

X = np.loadtxt(f'{input_file}')
X_df = pd.read_csv(f'{input_file}', names= names, sep=' ')
X_df.insert(0, 'SIM_NUM', range(1, len(X_df)+1))

w_base = np.array([0.25, 0.20, 0.15, 0.10, 0.10, 0.10, 0.10])

def get_weights(mode,w_base = w_base):
    #Pre-set goal: Optimize environmental impacts
    if mode == 'Eco':
        wGWP = 0.5
        wET = 0.05
        wEUT = 0.05
        wEtot = 0
        wTotalCost = 0
        wRevenue = 0
        wProfit = 0.4#0.4
        profitReq = 1
        weights = [wProfit,wTotalCost,wRevenue,wGWP,wET,wEUT,wEtot]
    #Pre-set goal: Maximize profit
    if mode == 'Profit':
        wGWP = 0
        wET = 0
        wEUT = 0
        wEtot = 0
        wTotalCost = 0
        wRevenue = 0
        wProfit = 1
        profitReq = 1   # Excludes unprofitable scenarios
        weights = [wProfit,wTotalCost,wRevenue,wGWP,wET,wEUT,wEtot]
    if mode == 'Custom':
        weights = w_base

    return weights, profitReq


def get_performance(mode, X_df, df_sims, min = 0):
    """mode is a string
    results is in form [Profit,Cost,Revenue,Biomass,GWP,ET,EUT,Etot]
    min is Minimum yield they want to produce, this can be user defined or irrelevant (default 0 if irrelevant)"""

    weights, profitReq = get_weights(mode)

    dfExclude = pd.DataFrame({'SIM_NUM':[],'Performance':[],'Profit': [], 'Cost': [], 'Revenue': [], 'Biomass':[], 'Harvest':[], 'GWP':[], 'ET':[], 'EUT':[], 'Etotal':[]})
    dfInclude_res = pd.DataFrame({'SIM_NUM':[],'Profit': [], 'Cost': [], 'Revenue': [], 'Biomass':[],'Harvest':[], 'GWP':[], 'ET':[], 'EUT':[], 'Etotal':[]})

    # Fix this, I think biomass and harvest need to be differentiated

    # Separate excluded and included scenarios
    for index, scenario in df_sims.iterrows():
        SIM_NUM,Profit,Cost,Revenue,Biomass,Harvest,GWP,ET,EUT,Etot= scenario['SIM_NUM'], scenario['Profit'], scenario['Cost'], scenario['Revenue'], scenario['Biomass'], scenario['Harvest'], scenario['GWP'], scenario['ET'], scenario['EUT'], scenario['Elec']  

        if math.isnan(Profit):
            print('skipping row',index)
            continue

        elif profitReq == 1:   #If eliminating unprofitable scenarios (provided some are profitable)
            if Profit <= 0:  #If profit is negative or 0
                Performance = 0   #0 score if not profitable - essentially eliminated unless all scenarios fail to produce profit
                if Harvest < min:
                    Performance = Performance + (Harvest-min) #Negative scores (eliminated) if not only not profitable, but also yield is too low.

                new_row = pd.DataFrame([{'SIM_NUM':SIM_NUM,'Performance':Performance,'Profit': Profit, 'Cost': Cost, 'Revenue': Revenue, 'Biomass':Biomass,'Harvest':Harvest, 'GWP':GWP, 'ET':ET, 'EUT':EUT, 'Etotal':Etot}])
                dfExclude = pd.concat([dfExclude, new_row], ignore_index=True)
            else:  # If profit is positive
                new_row = pd.DataFrame([{'SIM_NUM':SIM_NUM,'Profit': Profit, 'Cost': Cost, 'Revenue': Revenue, 'Biomass':Biomass, 'Harvest':Harvest,'GWP':GWP, 'ET':ET, 'EUT':EUT, 'Etotal':Etot}])
                dfInclude_res = pd.concat([dfInclude_res, new_row], ignore_index=True)

        else:  #if profit is not an exclusion criteria
            new_row = pd.DataFrame([{'SIM_NUM':SIM_NUM,'Profit': Profit, 'Cost': Cost, 'Revenue': Revenue, 'Biomass':Biomass,'Harvest':Harvest, 'GWP':GWP, 'ET':ET, 'EUT':EUT, 'Etotal':Etot}])
            dfInclude_res = pd.concat([dfInclude_res, new_row], ignore_index=True)
        
    if len(dfInclude_res) == 0:  # Print highest profitability if none are profitable
        winner_index = dfExclude['Performance'].idxmax() 
        Winner = dfExclude.loc[winner_index]
        dfInc_Standardized = dfInclude_res
        print('Warning: No profitable scenarios')

        return Winner, dfInc_Standardized, dfExclude
        
    else:
        maxProfit = dfInclude_res['Profit'].max()
        minCost = dfInclude_res['Cost'].min()
        maxRevenue = dfInclude_res['Revenue'].max()
        maxBiomass = dfInclude_res['Biomass'].max()
        minGWP = dfInclude_res['GWP'].min()
        minET = dfInclude_res['ET'].min()
        minEUT = dfInclude_res['EUT'].min()
        minEtotal = dfInclude_res['Etotal'].min()

        dfInc_Standardized = pd.DataFrame({'SIM_NUM':[], 'Performance':[],'pProfit': [], 'pCost': [], 'pRevenue': [], 'pGWP':[], 'pET':[], 'pEUT':[], 'pEtotal':[]})
        for index, row in dfInclude_res.iterrows():
            SIM_NUM = row['SIM_NUM']
            pProfit = weights[0]*row['Profit']/maxProfit
            pCost = weights[1]*minCost/row['Cost']
            pRevenue = weights[2]*row['Revenue']/maxRevenue
            pGWP = weights[3]*minGWP/row['GWP']
            pET = weights[4]*minET/row['ET']
            pEUT = weights[5]*minEUT/row['EUT']
            pEtot = weights[6]*minEtotal/row['Etotal']

            Performance = pProfit + pCost + pRevenue + pGWP + pET + pEUT + pEtot 

            new_row = pd.DataFrame([{'SIM_NUM':SIM_NUM,'Performance':Performance,'pProfit': pProfit, 'pCost': pCost, 'pRevenue': pRevenue, 'pGWP':pGWP, 'pET':pET, 'pEUT':pEUT, 'pEtotal':pEtot}])
            dfInc_Standardized = pd.concat([dfInc_Standardized, new_row], ignore_index=True)

        winner_index = dfInc_Standardized['Performance'].idxmax()
        Winner = dfInclude_res.loc[winner_index]
        winner_params = X_df.loc[X_df['SIM_NUM'] == Winner['SIM_NUM']]
        winner_perf = dfInc_Standardized.loc[winner_index]

        return Winner, winner_params, winner_perf, dfInc_Standardized, dfExclude, dfInclude_res
    

def get_stand(dfInclude_res):
    """Returns standardized scores (pre-weights) of each scenario that passes the exclusion criteria"""
    maxProfit = dfInclude_res['Profit'].max()
    minCost = dfInclude_res['Cost'].min()
    maxRevenue = dfInclude_res['Revenue'].max()
    maxBiomass = dfInclude_res['Biomass'].max()
    minGWP = dfInclude_res['GWP'].min()
    minET = dfInclude_res['ET'].min()
    minEUT = dfInclude_res['EUT'].min()
    minEtotal = dfInclude_res['Etotal'].min()

    matrix = np.empty((len(dfInclude_res), 7), dtype=float)
    for i, row in dfInclude_res.iterrows():
        matrix[i,0] = row['Profit']/maxProfit
        matrix[i,1] = minCost/row['Cost']
        matrix[i,2] = row['Revenue']/maxRevenue
        matrix[i,3] = minGWP/row['GWP']
        matrix[i,4] = minET/row['ET']
        matrix[i,5] = minEUT/row['EUT']
        matrix[i,6] = minEtotal/row['Etotal']

    return matrix

mode = 'Eco'
Winner, winner_params, winner_perf, dfInc_Standardized, dfExclude, dfInclude_res = get_performance(mode, X_df, df_sims)
print('Mode', mode)
print('Winner', Winner)
print('Winner Performance', winner_perf)
print('Winner Params', winner_params)
print('Eliminated:', len(dfExclude))


