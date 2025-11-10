from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class Params:
    # Environmental Settings
    Ti_set: Union[float, list, np.ndarray] = 18.0   # Set internal temperature [degrees C] - can be single value, or array of length 24
    iCO2: float = 700       # Set internal CO2 concentration [ppm]
    RHi: float = 65         # Set internal relative humidity [%]

    # Crop schedule
    cycle_days: int = 49    #Length of full crop cycle  - FOR NOW, DO NOT CHANGE.
    nurs_days: int = 21     #Number of days crop spends in nursery area before transplant - FOR NOW, DO NOT CHANGE.
    SeedingTime: int = 11   # Time of day (military time) when cycle begins
    TransTime: int = 10     #Time of day when seedlings are transferred from nursery to cultivation area (military time)
    HarvestTime: int = 8    #Time of day when heads are harvested
    heads: int = 528/2      # no. heads to be harvested/planted per half-wall per week

    # Light schedule
    PPN: int = 18 #Photoperiod of nursery area
    iDAYL: int = 18 #Set daily photoperiod [hours]  
    PPFDc: Union[int, float, list, np.ndarray] = 250 # Photosynthetic photon flux density of cultivation area [umol/m2/s] - can be int, float, or array. An array indicates varying PPFDs throughout the daily lighting schedule. Arrays must be of form [PPFDc1, PPFDc2, iDAYL2], where PPFDc1 is the PPFD during the first period, PPFDc2 is the PPFD during the second period, and iDAYL2 is the length (in hours) of the first period. The length of the second period must be equal to iDAYL - iDAYL2.
    PPFDn: int = 250 # Photosynthetic photon flux density of nursery area [umol/m2/s]
    NursStart: int = 17 #lights turn on at 17:00 = 5pm
    CultStart: int = 17 

    # Phyiscal parameters
    r_a: int = 400 #Aerodynamic stomatal resistance [s/m]

    #Manual inputs
    cult_water_changes: float = 0            #[1/wk] This is where, if the water is changed for the CULTIVATION TANK, that can be notated.
                             #ALTERNATIVELY, may want to give estimate of how often water is changed, and make that a fraction of the average /wk?
    nurs_water_changes: float = 0            #See above, but for nursery tank
    LaborHours: float = 6     ################### How many hours a week paid employees work

    #Prices of supplies/utilities
    WaterBC: float = 12.75 #monthly base charge for water based on meter size, compare to utilities.marionfl.org/customer-service/utility-rate-information
    WaterRate: float = 1.6 # [$/gal]
    FertPrice: float = 26.49 #[$/bottle]
    FertAmt: float = 1000 #[g/bottle]
    TraysPrice: float = 74 #[$/pack of trays purchased] - Assumes 288 seeds/tray, which is CURRENTLY NON-ADJUSTABLE
    TraysAmt: float = 50 #Trays/package (how many trays were purchased for TraysPrice)
    TrayLifetime: float = 260 #weeks - technically these can last forever but let's just say we replace them every 5 years
    PlugsAmt: float = 6000 #no. of plugs in purchased pack
    PlugsPrice: float = 727.05 #[$/pack of plugs purchased]
    SeedsAmt: float = 5000 #[seeds/pack of seeds purchased]
    SeedsPrice: float = 148.5 #[$/pack of seeds purchased]
    ElecBC: float = 12.68
    ElecRateP: float = 0.13289 + 0.00205
    ElecRateOP: float = 0.04542 + 0.00205
    NatGasCost: float = 0 #####
    LaborCost: float = 0 #####

    #Selling price
    Selling_Price: float = 4

    # %%
    TimeStep: int = 1 #[hr] - How often the variables will be re-calculated
    TimeSpan: int = 7 #[days] - Timespan for which results will be shown 

    StartDate: str = "2025-05-01" #StartDate is start of longest previous cycle, not start date of study (should probably change that)

    #Indicdate business model: Same price (1), or not same price (0)
    BusinessModel: int = 1      # Note: Business Model 0 requires uploading 
    Goal_Harvest: float = 0   # kg required to fulfill orders (Only relevant with BusinessModel == 0)

    # %%
    #Design variables

    U: float = 0.036*(40*8*4 + 9.5*8*2)  #%Thermal U-value [BTU/hr/F] - Assumes area of walls + floor + ceiling + doors = (40*8*4 + 9.5*8*2)

    ACHleak: float = 0              #% Infiltration (leaking) rate 
    V: float = 83                  #Interior volume [m3]  -- approximated

    ACHvent: float = 2.8             # Air changes per hour from vent 
    fppm: float = 1.2                 #Liquid fertilizer concentration in water [g/L] -- CUSTOMIZABLE
    EffEq: float = 0#0.65                #Estimated combined efficiency of pump, aerator, and fans -- ##################### UPDATED
    Efan: float = (200+111+111)*24/1000         #[kWh/day]   --- Overhead fan = 200 W, Side duct fans = 111 W each (Canarm® Turbo Tube 8" Inline Mixed Flow Duct)  
    Eaerator: float = 58*24/1000    #[kWh/day]   800 GPH Active Aqua pump
    EpumpNURS: float = 40*2*24/1000             #[kWh/day]   2x https://www.littlegiant.com/products/condensate-removal-pumps/condensate-pumps/vcl-series/vcl-24uls/
    EpumpCULT: float = 122*24/1000             #[kWh/day] (dual, but assumed only one runs at a time) https://www.amazon.com/Little-Giant-5-MSP-10-115-Volt-Aluminum/dp/B000LGA666?th=1 
    Erecirpump: float = (16+45)*24/1000      #500 GPH in cult tank, 250 GPH in nurs tank https://www.agriculturesolutions.com/active-aqua-submersible-water-pump-250-gph-pack-of-3 https://www.agriculturesolutions.com/active-aqua-submersible-water-pump-550-gph-pack-of-2
    Einj: float = 3.0*24/1000       #CO2 injector, assumed not injecting 24/7 [kWh/day]  ----?
    EER: float = 11                   #Energy efficiency ratio of climate control unit [BTU/W]
    EffCL: float = 0.83               # Cultivation lighting efficiency --- not used anywhere anymore??????
    EffNL: float = 0.834              # Nursery lighting efficiency
    Qhvac_max: float = 36000   #[BTU/hr] climate control unit capacity

    GR: float = 72/100                  #Germination rate -- CUSTOMIZABLE

    Aw: float = 14.31914/2            #SECTION area - 1/2 of cultivation wall area – one wall [m2] 

    Lat: float = 29.408563    #Latitude of farm
    Long: float = -82.171154   #Longitude of farm

    c_p_air: float = 1005 #[J/kg/K]

    Demand: float = 100 #Estimated local demand [kg] ######May want to make this per a specific timeframe or something???? So will have to multiply by the timespan
        # Made it real big rn to not account for that

    # Crops in farm
    #From vald study I think:
    #1 Tiny-Tiny      3 Giant      5 Tiny      7 Large
    #2 Tiny-Tiny      4 Giant      6 Small     8 Tiny-Small
    #So for now, planted: Week 1: 3+4, Week 2: 7, Week 3: 6+8, Week 4: 1+2+5, now predicting Week 7. There is also something in nursery area but I don't have temps for that area yet
    #Adjusted just to make things even in terms of how many were planted when
    weeks1: float = 3 #How many weeks the plants in Ti1 have been in their cycle
    weeks2: float = 3 # ^^^^ Ti2
    weeks3: float = 6 # ^^^^ Ti3
    weeks4: float = 6 # ^^^^ Ti4
    weeks5: float = 4 # ^^^^ Ti5
    weeks6: float = 4 # ^^^^ Ti6
    weeks7: float = 5 # ^^^^ Ti7
    weeks8: float = 5 # ^^^^ Ti8
    #Above is the cycles currently on the wall. Below is the cycles currently in nursery station. At any given time, will have 7 main cycles, which segmented is 14
    weeks9: float = 2   #Nurse station -> trans to 1
    weeks10: float = 2  #trans to 2
    weeks11: float = 1  #trans to 3
    weeks12: float = 1  #trans to 4
    weeks13: float = 0  #trans to 5
    weeks14: float = 0  #trans to 6
    weeks15: float = -1  #trans to 7 - cycle not currently in rotation
    weeks16: float = -1  #trans to 8 - cycle not currently in rotation