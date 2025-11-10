# %%
#from SALib.analyze.sobol import analyze
#from SALib.sample.sobol import sample
#from SALib import ProblemSpec               # SALib imports only important for global sensitivity analysis
import numpy as np
import sys
from params import Params

# %%
import pandas as pd
import matplotlib.pyplot as plt 
import math
from scipy.stats import norm
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import date, datetime, timedelta
SM = pd.read_csv('SimaPro2.csv', index_col="Impact category")  #simapro matrix
SM = SM.rename(columns={"Unit": "Unit", "Electricity, medium voltage {US}| market group for electricity, medium voltage | Cut-off, U": "Electricity", "Heat, district or industrial, natural gas {GLO}| market group for heat, district or industrial, natural gas | Cut-off, U": "Heat, Nat Gas", "Carbon dioxide, liquid {RoW}| market for carbon dioxide, liquid | Cut-off, U": "Liquid CO2", "Tap water {GLO}| market group for tap water | Cut-off, U": "Tap Water", "Ammonium nitrate {RoW}| market for ammonium nitrate | Cut-off, U": "Ammonium Nitrate N", "Inorganic potassium fertiliser, as K2O {US}| market for inorganic potassium fertiliser, as K2O | Cut-off, U": "Potassium Sulfate K2O", "Inorganic phosphorus fertiliser, as P2O5 {US}| market for inorganic phosphorus fertiliser, as P2O5 | Cut-off, U": "Phosphate P2O5", "Inorganic nitrogen fertiliser, as N {US}| market for inorganic nitrogen fertiliser, as N | Cut-off, U": "Nitrogen N", "Magnesium oxide {GLO}| market for magnesium oxide | Cut-off, U": "Magnesium Oxide","Stone wool, packed {GLO}| market for stone wool, packed | Cut-off, U":"Rockwool","Polypropylene, granulate {GLO}| market for polypropylene, granulate | Cut-off, U":"Polypropylene Trays","Runoff water":"Dumped water"})
ValData = pd.read_csv('ValdStudyData.csv', index_col="Time step (dt = 5s)", nrows=28981)

import psychrolib
#Cite: https://github.com/psychrometrics/psychrolib/blob/master/README.md
psychrolib.SetUnitSystem(psychrolib.SI)

date_list = [               # For selections by global sensitivity analysis
    "2024-01-01","2024-02-01","2024-03-01","2024-04-01","2024-05-01","2024-06-01",
    "2024-07-01","2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01"
    ]               

# %%
def get_weather(p):
    #Using Open-Meteo Historical and Forecast APIs (CITE: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649)
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    weeks = [p.weeks1, p.weeks2,p.weeks3,p.weeks4,p.weeks5,p.weeks6,p.weeks7,p.weeks8,p.weeks9,p.weeks10,p.weeks11,p.weeks12,p.weeks13,p.weeks14,p.weeks15,p.weeks16]
    startdate_ = datetime.strptime(p.StartDate, "%Y-%m-%d")
    EndDate = (startdate_ + timedelta(days = (p.TimeSpan + int(max(weeks)*7)))).strftime("%Y-%m-%d")  #Changed to cycle length which is what it was originally but idk why??? Outdoor conditions only needed during study period

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    # Temp and RH
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": p.Lat, 
        "longitude": p.Long,
        "start_date": p.StartDate,
        "end_date": EndDate,
        "hourly": ["temperature_2m", "relative_humidity_2m"]
    }
    responses = openmeteo.weather_api(url, params=params)
    # Atmospheric CO2
    url2 = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params2 = {
        "latitude": p.Lat,
        "longitude": p.Long,
        "hourly": "carbon_dioxide",
        "start_date": p.StartDate,
        "end_date": EndDate
    }
    responses2 = openmeteo.weather_api(url2, params=params2)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

    response2 = responses2[0]
    hourly2 = response2.Hourly()
    hourly_carbon_dioxide = hourly2.Variables(0).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["To"] = hourly_temperature_2m
    hourly_data["RHo"] = hourly_relative_humidity_2m/100
    hourly_data["CO2"] = hourly_carbon_dioxide

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    hourly_dataframe = hourly_dataframe.tail(-p.SeedingTime)
    hourly_dataframe = hourly_dataframe.head(len(hourly_dataframe) - (24-p.SeedingTime))

    # %%
    To = hourly_dataframe['To'].to_numpy()
    RHo = hourly_dataframe['RHo'].to_numpy()
    atmCO2 = hourly_dataframe['CO2'].to_numpy()
    hourly_timestamps = hourly_dataframe['date'].to_numpy()

    for i in range(len(atmCO2)):
        atmCO2[i] = 400 #change outdoor CO2 to constant ppm, because this is broken rn 

    return To, RHo, atmCO2, hourly_timestamps

#Indoor temperature distributions should actually come from sensor data or surrogate CFD model -- for now, reusing experimental data
ValDataHour = ValData.iloc[::720]
ValDataHour = ValDataHour.reset_index()   #index 0 = 19:00 on Dec 17, index 40 = 11:00 on Dec 19

def get_indoor_temps(p, Ti_set):
    # To also starts at SeedingTime
    # Ti starts at seeding time
    # %%
    
    #For the sake of FakeData, repeating the 24 hour period (i = [5:28]) 7 times to approximate variations in temperatures throughout the farm and day
    ValData24Hr = ValDataHour[5+p.SeedingTime:29+p.SeedingTime]   #(12am to 11pm) + Offset by seeding time
    FakeData = pd.concat([ValData24Hr], ignore_index=True) 

    weeks = [p.weeks1, p.weeks2,p.weeks3,p.weeks4,p.weeks5,p.weeks6,p.weeks7,p.weeks8,p.weeks9,p.weeks10,p.weeks11,p.weeks12,p.weeks13,p.weeks14,p.weeks15,p.weeks16]
    days_Ti = int(p.TimeSpan + max(weeks)*7)   # Number of days in StudyTime + oldest cycle before
    # %% 
    Ti1_ = [] #C
    Ti2_ = []
    Ti3_ = []
    Ti4_ = []
    Ti5_ = []
    Ti6_ = []
    Ti7_ = []
    Ti8_ = []
    for i in range(len(FakeData)): 
        Ti1_.append(round(((FakeData.at[i,'T23'] + FakeData.at[i,'T12'] + FakeData.at[i,'T22'] + FakeData.at[i,'T30'])/4),4))    #Back left corner (facing cult. area from door)
        Ti2_.append(round(((FakeData.at[i,'T22'] + FakeData.at[i,'T30'] + FakeData.at[i,'T21'] + FakeData.at[i,'T29'])/4),4))    #Front left corner
        Ti3_.append(round(((FakeData.at[i,'T27'] + FakeData.at[i,'T25'] + FakeData.at[i,'T8'] + FakeData.at[i,'T14'])/4),4))     #Back center-left
        Ti4_.append(round(((FakeData.at[i,'T8'] + FakeData.at[i,'T14'] + FakeData.at[i,'T13'] + FakeData.at[i,'T28'])/4),4))     #Front center-left
        Ti5_.append(round(((FakeData.at[i,'T26'] + FakeData.at[i,'T11'] + FakeData.at[i,'T9'] + FakeData.at[i,'T18'])/4),4))     #Back center-right
        Ti6_.append(round(((FakeData.at[i,'T9'] + FakeData.at[i,'T18'] + FakeData.at[i,'T7'] + FakeData.at[i,'T10'])/4),4))      #Front center-right
        Ti7_.append(round(((FakeData.at[i,'T16'] + FakeData.at[i,'T17'] + FakeData.at[i,'T15'] + FakeData.at[i,'T20'])/4),4))    #Back right corner
        Ti8_.append(round(((FakeData.at[i,'T15'] + FakeData.at[i,'T20'] + FakeData.at[i,'T24'] + FakeData.at[i,'T19'])/4),4))    #Front right corner

    #Extend these all from one day to length of days_Ti
    Ti1_ = Ti1_*days_Ti  
    Ti2_ = Ti2_*days_Ti
    Ti3_ = Ti3_*days_Ti
    Ti4_ = Ti4_*days_Ti
    Ti5_ = Ti5_*days_Ti
    Ti6_ = Ti6_*days_Ti
    Ti7_ = Ti7_*days_Ti
    Ti8_ = Ti8_*days_Ti

    #Right now, assuming all nursery stations have the same temp, and arbitrarily setting it to Ti1 (more work required to determine better approximation, however nursery crop model growth does not actually depend on its temperature at the moment).
    TiN_ = Ti1_

    Ti_avg_ = []
    for i in range(len(Ti1_)):
        Ti_avg_.append(round(((Ti1_[i] + Ti2_[i] + Ti3_[i] + Ti4_[i] + Ti5_[i] + Ti6_[i] + Ti7_[i] + Ti8_[i] + TiN_[i])/9),4))  #Only TiN once because it takes up realistically less space than the cult segments - this is just approx. Real avg would be based on whatever sensor they really have
    
    if isinstance(Ti_set, (int, float)):
        Ti1 = [(i + (Ti_set - 18.6505)) for i in Ti1_]   # 18.6505 is the average temp of Ti_avg in experimental data
        Ti2 = [(i + (Ti_set - 18.6505)) for i in Ti2_]
        Ti3 = [(i + (Ti_set - 18.6505)) for i in Ti3_]
        Ti4 = [(i + (Ti_set - 18.6505)) for i in Ti4_]
        Ti5 = [(i + (Ti_set - 18.6505)) for i in Ti5_]
        Ti6 = [(i + (Ti_set - 18.6505)) for i in Ti6_]
        Ti7 = [(i + (Ti_set - 18.6505)) for i in Ti7_]
        Ti8 = [(i + (Ti_set - 18.6505)) for i in Ti8_]
        TiN = [(i + (Ti_set - 18.6505)) for i in TiN_]
        Ti_avg = [(i + (Ti_set - 18.6505)) for i in Ti_avg_]
    else:   #Ti_set is a list with varying daily set temperatures
        Ti_set_ = Ti_set*days_Ti
        Ti1 = [(i - 18.6505) for i in Ti1_]   # 18.6505 is the average temp of Ti_avg
        Ti2 = [(i - 18.6505) for i in Ti2_]
        Ti3 = [(i - 18.6505) for i in Ti3_]
        Ti4 = [(i - 18.6505) for i in Ti4_]
        Ti5 = [(i - 18.6505) for i in Ti5_]
        Ti6 = [(i - 18.6505) for i in Ti6_]
        Ti7 = [(i - 18.6505) for i in Ti7_]
        Ti8 = [(i - 18.6505) for i in Ti8_]
        TiN = [(i - 18.6505) for i in TiN_]
        Ti_avg = [(i - 18.6505) for i in Ti_avg_]
        for x in range(len(Ti_set_)):
            Ti1[x] = Ti1[x] + Ti_set_[x]
            Ti2[x] = Ti2[x] + Ti_set_[x]
            Ti3[x] = Ti3[x] + Ti_set_[x]
            Ti4[x] = Ti4[x] + Ti_set_[x]
            Ti5[x] = Ti5[x] + Ti_set_[x]
            Ti6[x] = Ti6[x] + Ti_set_[x]
            Ti7[x] = Ti7[x] + Ti_set_[x]
            Ti8[x] = Ti8[x] + Ti_set_[x]
            TiN[x] = TiN[x] + Ti_set_[x]
            Ti_avg[x] = Ti_avg[x] + Ti_set_[x]
    
    return Ti1, Ti2, Ti3, Ti4, Ti5, Ti6, Ti7, Ti8, Ti_avg, TiN, weeks


# %%
#Calculate abs humidity ratio AHi [g/m3]:
#https://www.omnicalculator.com/physics/absolute-humidity
Pc = 22.064 #[MPa] - critical pressure for water
Tc = 647.096 #[K] - critical temp for water
Rw = 461.5/1000 #Specific gas constant for water vapor J/g/K
a1 = -7.85951783
a2 = 1.84408259
a3 = -11.7866497
a4 = 22.6807411 
a5 = -15.9618719 
a6 = 1.80122502

#Talbot & Monfet (2024)
c_car1 = -1.32*10**(-5) #Constant from lettuce growth model (Van Henten, 1994)[m/(s*degC^2)]
c_car2 = 5.94*10**(-4) #Constant from lettuce growth model (Van Henten, 1994)[m/(s*degC)]
c_car3 = -2.64*10**(-3) #Constant from lettuce growth model (Van Henten, 1994)[m/s]
c_alpha = 0.68 #Constant from lettuce growth model (Van Henten, 1994)[-]
g_bnd = 0.01 #Boundary layer conductance (Graamans et al., 2017)[m/s]
c_eta = 7.32*10**(-2) #CO2 compensation point at 20degC (Van Henten & Van Straten (1994)) 
c_Q10eta = 2.0  #Constant from lettuce growth model (Van Henten, 1994)[-]
c_respsht = 3.47*10**(-7) #Constant from lettuce growth model (Van Henten, 1994)[1/s]
c_teta = 0.15 #Constant from lettuce growth model for lettuces grown in soil (Van Henten, 1994)[-]
c_resprt = 1.16*10**(-7) #Constant from lettuce growth model (Van Henten, 1994)[1/s]
c_Q10resp = 2 #Constant from lettuce growth model (Van Henten, 1994)[-]
c_Q10gr = 1.6   #Constant from lettuce growth model (Van Henten, 1994)[-]
k_T = 0.58 #Extinction coefficient for total solar radiation (Tei et al., 1996) [-]
gamma = 66.5 # Psychometric constant [Pa/K]
Press = 1   # atmospheric pressure [atm]
fApproxErr = 0.000001 # Maximum allowable approximate error [%]
iIterMax = 1000 # Maximum number of iterations
fDelta = 0.00001

#Equation for extracting c_beta linearly from provided values
ppfds = np.array([200, 400, 750])
cbetas = np.array([0.401, 0.402, 0.403])
coefficients = np.polyfit(ppfds, cbetas, 1)
slope = coefficients[0] 
intercept = coefficients[1]

k_P = 0.66  #Extinction coefficient for electric lighting [-]

# %%
def cropmodel(conditions,p,Ti,days,Xsdw0=0,Xnsdw0=0,predict='No',past_hrs = 0):
    """"Van Henten Model
    Ti = temperature dist list in relevant area **Length must account for all days investigating and begin at SeedingTime on first day**
    days = how many days to predict growth over
    past_hours = how many hours the section is into its cycle
    Xsdw0 = estimated starting structural dry weight (default 0 kg)
    Xnsdw0 = estimated starting non-structural dry weight (default 0 kg)
    Predict = 'No' assumes we use historical data. This is the default. len(Ti) must be longer than current cycle length and END at the end of the historical period
    Predict = 'Yes' uses predicted conditions. Ti[0] must be the beginning of this prediction period"""
    
    if days <= 0:
        PHIphotc = [0]
        RespCO2 = [0]
        Xsdw_ = [0]
        Xnsdw_ = [0]
        SHT_FW_ = [0] 
        PHItransp = [0] 
        cycle_length = 0 
        qLEDs = [0] 
        qPlant = [0]
        AHi = []
        AHo = []
        no_harvests = 0
        HarvestedFW = 0
        return PHIphotc, RespCO2, Xsdw_, Xnsdw_, SHT_FW_, PHItransp, cycle_length, qLEDs, qPlant, no_harvests, HarvestedFW, AHi, AHo
      
    NursHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - nursery period] (24 hrs x 21 days)
    CultHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime,24*p.cycle_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - cult period] (24 hrs x 28 days) #Until the other cycle is transplanted, the lights here count as heat gain for this section (assuming lights do not turn off when space is empty)

    #Find regression for nursery time
    XsdwSLOPE = 1.125/len(NursHours)  #Structural dry weight over time in nursery section, assuming steady growth (falsely) to starting values in Talbot
    XnsdwSLOPE = 0.375/len(NursHours)

    Vrad = conditions[0]
    PAR_ = conditions[1]
    PPFDn = conditions[2]
    PPFDc = conditions[3]
    RHi = conditions[4]
    iCO2 = conditions[5]
    Crop_density = conditions[6]
    TiN = conditions[7]
    To = conditions[8]
    RHo = conditions[9]
    Correction = conditions[10]
    no_harvests = 0  #number of times the crops were harvested in this section in this timespan
    HarvestedFW = 0  #g/plant of harvested lettuce
    cycle_length = past_hrs
    Vos = past_hrs  #offset for Vrad schedule
    PHIphotc = [0]*int(24*days)
    RespCO2 = [0]*int(24*days)
    PHItransp = [0]*int(24*days)
    qLEDs = [0]*int(24*days)
    qPlant = [0]*int(24*days)
    AHi = [0]*int(24*days)
    AHo = [0]*int(24*days)
    Xsdw_ = [0]*int(24*days+1)
    Xnsdw_= [0]*int(24*days+1)
    SHT_FW_= [0]*int(24*days)  #This is offset one time step forward because it doesn't count an inital condition
    Xsdw_[0] = Xsdw0     #Initial value of the structural dry weight [g/m^2]
    Xnsdw_[0]= Xnsdw0     #Initial value of the non-structural dry weight [g/m^2]
    
    #Offset below assumes we're using the last "days" days of the provided weather data
    if predict == 'No':
        offset = len(Ti) - int(24*days) #This allows us to choose the weather based on the planting date, rather than assuming all cycles are planted on the same day
    elif predict == 'Yes':
        offset = 0
    
    for i in range(int(24*days)):
        cycle_length += 1
        
        if cycle_length <= len(NursHours):
            T_a = TiN[i+offset]  #Change to temp value in nursery station, not the wall it will eventually be at
        else:
            T_a = Ti[i+offset]
        
        T_o = To[i+offset]
        tau = 1 - ((T_a+273.15)/Tc)
        #Psi = Pc*1000000*math.exp((Tc/(T_a+273.15))*(a1*tau + a2*tau**1.5 + a3*tau**3 + a4*tau**3.5 + a5*tau**4 + a6*tau**7.5)) #[Pa] - saturation vapor pressure
        Psi = 610.78 * math.exp(17.27 * T_a / (T_a + 237.3))
        AHi[i] = RHi * Psi / (Rw * (T_a+273.15))      #Indoor abs humditiy [g/m3]

        #Calculate outdoor humidity ratio AHo [g/m3]
        tauo = 1 - ((T_o + 273.15) / Tc)
        #Pso = Pc*1000000*math.exp((Tc/(T_o+273.15))*(a1*tauo + a2*tauo**1.5 + a3*tauo**3 + a4*tauo**3.5 + a5*tauo**4 + a6*tauo**7.5)) #[Pa] - saturation vapor pressure
        Pso = 610.78 * math.exp(17.27 * T_o / (T_o + 237.3))
        AHo[i] = RHo[i+offset] * Pso / (Rw * (T_o+273.15))      #Outdoor absolute humidity [g/m3]
        
        
        if cycle_length > len(PAR_): #When we've used all of PAR_, the cycle will re-start. PAR_ starts AND ends at the seeding time, so the new cycle re-starts at the correct time.
            cycle_length = -7*24
            Vos = cycle_length 
            Xsdw_[i+1] = 0
            Xnsdw_[i+1] = 0
            SHT_FW_[i] = 0
            RespCO2[i] = 0
            PHIphotc[i] = 0
            PHItransp[i] = 0
            qPlant[i] = 0
            qLEDs[i] = 0
        
        elif cycle_length > (24*p.cycle_days-(p.SeedingTime-p.HarvestTime)):  #If past harvest time - aka, cycle is done
            PAR = PAR_[Vos]
            Xsdw_[i+1] = 0
            Xnsdw_[i+1] = 0
            SHT_FW_[i] = 0
            RespCO2[i] = 0
            PHIphotc[i] = 0
            PHItransp[i] = 0
            qPlant[i] = 0
            qLEDs[i] = PAR * 3.6 * p.Aw   # Convert W/m^2 to kJ/hr - remember: PAR becomes stagnant 0 only once the new transplant replaces this section
            #^ After harvet, all things adjusted, but still cycling through PAR_
            Vos += 1
        
        elif cycle_length <0:  #When past_hours is negative aka, during that week of nothing
            Vos = cycle_length   # this keeps it at beginning of the PAR_ list (PAR_[0]) until the new cycle starts (when cycle_length = 0) so that the nursery weeks have the correct PAR
            Xsdw_[i+1] = 0
            Xnsdw_[i+1] = 0
            SHT_FW_[i] = 0
            RespCO2[i] = 0
            PHIphotc[i] = 0
            PHItransp[i] = 0
            qPlant[i] = 0
            qLEDs[i] = 0
        
        else:
            PAR = PAR_[Vos]
            PPFD = Vrad[Vos]
            Xsdw_initial = Xsdw_[i] #Xsdw_[i]    #Structural dry weight from previous timestep [g/m^2]
            Xnsdw_initial = Xnsdw_[i] #Xnsdw_[i]   #Non-structural dry weight from previous timestep[g/m^2]

            #Splitting the ranges for the variables below from Talbot
            if T_a <= 22:
                a = -1.5319*10**(-11)
                b = 9.48*10**(-9)
                c = 9.2492*10**(-6)
                d = -2.1364*10**(-12) 
                e = 2.7308*10**(-9)
                f = 2.9689*10**(-7)
                g = 3.7662*10**(-8)  
                h = -5.5597*10**(-5)
                j = 4.2313*10**(-2)
            if 22 <= T_a <= 26:
                a = -2.559*10**(-11)
                b = 1.8529*10**(-8)
                c = 8.3203*10**(-6)
                d = -1.7021*10**(-12) 
                e = 2.0302*10**(-9)
                f = 3.3353*10**(-7)
                g = 3.0779*10**(-8) 
                h = -6.1968*10**(-5)
                j = 5.6362*10**(-2)
            if T_a >= 26:
                a = -3.3958*10**(-12)
                b = -3.7225*10**(-9)
                c = 1.1632*10**(-5)
                d = -2.3461*10**(-12)  
                e = 2.8652*10**(-9)
                f = -3.0279*10**(-7)
                g = 4.4935*10**(-8) 
                h = -6.9961*10**(-5)
                j = 5.2195*10**(-2)

            #Basic regression for nursery hours
            if cycle_length <= len(NursHours):
                para = PPFDn
                c_epsilon = a * para**2 + b * para + c    #Light use efficiency at very high CO2 concentrations (Van Henten, 1994)[g/J]
                c_grmax = d * para**2 + e * para + f
                SLA = g * para**2 + h*para + j
                DW_content = -8.83*10**(-8) * para**2 + 1.1299*10**(-4) * para + 6.9351*10**(-3)
                c_beta = slope*para + intercept  #Constant from lettuce growth model (Van Henten, 1994)[-]

                CO2 = iCO2*0.00183     #CO2 concentration [ppm] convert to [g/m^3]
                g_car= c_car1*T_a**2+c_car2*T_a+c_car3             #Carboxylation conductance [m/s]
                g_stm = (200+PPFD)/(60*(1500+PPFD))    #Stomatal conductance [m/s]
                g_CO2 = (g_bnd**(-1)+g_stm**(-1)+g_car**(-1))**(-1)           #Canopy conductance to CO2 diffusion [m/s]
                eta=c_eta*c_Q10eta**((T_a-20)/10)            #CO2 compensation point [ppm]
                eps = c_epsilon*(CO2-eta)/(CO2+2*eta)    #Light use efficiency [g/J]
                f_photmax = eps*PAR*g_CO2*(CO2-eta)/(eps*PAR+g_CO2*(CO2-eta))          #g/(m^2*s^-1) 
                f_phot = f_photmax*(1-math.exp(-k_P*(SLA*(1-c_teta)*0.92*(Xsdw_initial+Xnsdw_initial))))    #g/(m^2*s^-1) 
                f_resp = (c_respsht*(1-c_teta)*Xsdw_initial + c_resprt*c_teta*Xsdw_initial)*c_Q10resp**((T_a-25)/10)      #g/(m^2*s^-1)

                dXsdw = XsdwSLOPE*p.TimeStep
                dXnsdw = XnsdwSLOPE*p.TimeStep
                Vos += 1

            # Below equations are from Talbot & Monfet (2024)
            else:
                para = PPFDc
                #First get constants
                c_epsilon = a * para**2 + b * para + c    #Light use efficiency at very high CO2 concentrations (Van Henten, 1994)[g/J]
                c_grmax = (d * para**2 + e * para + f) * Correction # Added correction coefficient
                SLA = g * para**2 + h*para + j
                DW_content = -8.83*10**(-8) * para**2 + 1.1299*10**(-4) * para + 6.9351*10**(-3)
                c_beta = slope*para + intercept  #Constant from lettuce growth model (Van Henten, 1994)[-]

                #Renaming
                PPFD = Vrad[Vos]
                PAR = PAR_[Vos]
                Xsdw_initial = Xsdw_[i] #Structural dry weight from previous timestep [g/m^2]
                Xnsdw_initial = Xnsdw_[i] #Non-structural dry weight from previous timestep[g/m^2]

                CO2 = iCO2*0.00183     #CO2 concentration [ppm] convert to [g/m^3]
                g_car= c_car1*T_a**2+c_car2*T_a+c_car3             #Carboxylation conductance [m/s]
                g_stm = (200+PPFD)/(60*(1500+PPFD))    #Stomatal conductance [m/s]
                g_CO2 = (g_bnd**(-1)+g_stm**(-1)+g_car**(-1))**(-1)           #Canopy conductance to CO2 diffusion [m/s]
                eta=c_eta*c_Q10eta**((T_a-20)/10)            #CO2 compensation point [ppm]
                eps = c_epsilon*(CO2-eta)/(CO2+2*eta)    #Light use efficiency [g/J]
                f_photmax = eps*PAR*g_CO2*(CO2-eta)/(eps*PAR+g_CO2*(CO2-eta))          #g/(m^2*s^-1) 
                f_phot = f_photmax*(1-math.exp(-k_P*(SLA*(1-c_teta)*0.92*(Xsdw_initial+Xnsdw_initial))))    #g/(m^2*s^-1) 
                f_resp = (c_respsht*(1-c_teta)*Xsdw_initial + c_resprt*c_teta*Xsdw_initial)*c_Q10resp**((T_a-25)/10)      #g/(m^2*s^-1)
                r_gr = c_grmax*Xnsdw_initial/(Xsdw_initial + Xnsdw_initial)*c_Q10gr**((T_a-20)/10)

                dXsdw = r_gr*Xsdw_initial*p.TimeStep*3600     #additional (new) structural dry weight [g/m2]
                dXnsdw = (c_alpha*f_phot-r_gr*Xsdw_initial-f_resp-(1-c_beta)/c_beta*r_gr*Xsdw_initial)*p.TimeStep*3600   #Additional non-structural dry weight [g/m2]

                Vos += 1
                
            Xsdw = Xsdw_initial + dXsdw   #Structural dry weight[g/m^2]
            Xnsdw = Xnsdw_initial + dXnsdw   #Non-structural dry weight[g/m^2]
            Y_DW = (Xsdw + Xnsdw)/Crop_density   #Total (shoot and root) dry weight per plant [g/plant]
            LAI = SLA*(1-c_teta)*0.92*(Xsdw+Xnsdw)   #Leaf area index [m2_leaves/m2_cultivated floor area]
            LA = LAI/Crop_density  #Leaf area [m2_leaves/plant]
            SHT_DW = (1-c_teta)*Y_DW   #Dry weight (shoot only) [g/plant]
            SHT_FW = SHT_DW/DW_content   #Fresh weight (shoot only) [g/plant]

            Xsdw_[i+1] = Xsdw
            Xnsdw_[i+1] = Xnsdw
            SHT_FW_[i] = SHT_FW  #Fresh weight per plant [g/plant]    
            
            if cycle_length == (24*p.cycle_days-(p.SeedingTime-p.HarvestTime)):
                no_harvests +=1   ########Need to make a variable for this
                HarvestedFW = HarvestedFW + SHT_FW_[i]

            #Gross canopy photosynthesis rate [kg m-2 s-1]
            PHIphotc[i] = f_phot/1000 #[kg/m2/s]

            #Respiration rate 
            RespCO2[i] = f_resp/1000   #[kg m-2 s-1]
            RespCO2[i] = RespCO2[i]*60*60 #[kg m-2 h-1]
            PHIphotc[i] = PHIphotc[i]*60*60    #converted to [kg m-2 h-1]

            #Energy Balance
            if LAI>0:    

                CAC_rho=1-math.exp(-k_T*LAI)   #Cultivated Cover Area muliplied by light absorptivity of crops[-]
                CAC_rho_PAR=1-math.exp(-k_P*LAI)   #Cultivated Cover Area muliplied by light absorptivity of crops[-]

                if CAC_rho>0.95:
                    CAC_rho=0.95
                    CAC_rho_PAR=0.95

                #Determine knowns from inputs and parameters
                Rnet = CAC_rho_PAR*PAR   # Fraction of total emitted light absorbed by vegetation [W/m^2_cultivated]
                G_EL = (1-CAC_rho_PAR)*PAR #PAR emitted but not absorbed  

                r_s = 1/g_stm # surface (stomatal) resistance [s/m]

                # Get ambient air properties
                psydat1 = Press  #Atmospheric pressure [atm]
                psydat2 = T_a   #Air temp, dry bulb [degC]
                psydat4 = RHi   #Relative humidity [%]
                psydat6 = psychrolib.GetHumRatioFromRelHum(T_a,RHi,Press*101325) #humidity ratio [kgwater/kgdry_air]

                #More constants
                R_w = 461.5 #[J/kg/K] gas constant of water vapor
                R_a = 286.9 #[J/kg/K] gas constant of dry air

                airprops5 = 1.005 #Specific heat capacity of dry air at constant pressure [kJ/kg/K]
                psydat9 = (Press*101325 / (R_a * (psydat2+273.15)) * (1 / (1 + psydat6 * (R_w / R_a)))) #density of the air portion of the mixture (kg dry air/m3)
                Xa = AHi[i]  #Air vapor concentration aka abs humidity [g_water/m3] 
                CmoistAir = (psydat9*airprops5)+(Xa*1.82/1000) # Moist air specifc heat capacitance [kJ/m3K]

                # Intialize iteration variables
                T_s = T_a-2
                T_s_Old = T_s
                psydat4 = 1 #Relative humidity - Set to saturation
                fRelErr = 100 # Initialize relative error
                iIter = 0 # Initialize iteration counter

                # Iterate using the open modified secant method to find the root
                while iIter<iIterMax and fRelErr>fApproxErr:
                    # Get saturated vapour concentration at transpiration surface
                    psydat2 = T_s  #Dry bulb temp (at leaf surface)
                    #Use psydat2 and psydat 4 to find others
                    psydat6 = psychrolib.GetHumRatioFromRelHum(psydat2,psydat4,Press*101325)
                    psydat9 = (Press*101325 / (R_a * (psydat2+273.15)) * (1 / (1 + psydat6 * (R_w / R_a))))
                    Xs = psydat9*psydat6*1000  # g_water/m3 abs humidity AT CANOPY LEVEL
                    T_s_Delta = T_s+(T_s*fDelta)
                    psydat2 = T_s_Delta
                    #Use psydat2 and psydat 4 to find others
                    psydat6 = psychrolib.GetHumRatioFromRelHum(psydat2,psydat4,Press*101325)
                    psydat9 = (Press*101325 / (R_a * (psydat2+273.15)) * (1 / (1 + psydat6 * (R_w / R_a))))
                    Xs_Delta = psydat9*psydat6*1000  # g_water/m3

                    # Generate new guess of T_s
                    lam = (-3.6*(10**(-3))*(T_s**2))-(2.0272*T_s)+2495.2 #[kJ/kg] enthalpy of vaporization for water
                    q_sens_watt = LAI*CmoistAir*(T_s-T_a)/p.r_a*1000 # [W/m^2_cultivated]
                    q_lat_watt = LAI*lam*(Xs-Xa)/(r_s+p.r_a) # [W/m^2_cultivated]
                    q_sens_watt_Delta = LAI*CmoistAir*(T_s_Delta-T_a)/p.r_a*1000
                    q_lat_watt_Delta = LAI*lam*(Xs_Delta-Xa)/(r_s+p.r_a)
                    T_s = T_s-((T_s*fDelta*(Rnet-q_sens_watt-q_lat_watt)) / ((Rnet-q_sens_watt_Delta-q_lat_watt_Delta)-(Rnet-q_sens_watt-q_lat_watt)))

                    # Update iteration parameters
                    fRelErr = abs((T_s-T_s_Old)/T_s)*100
                    iIter = iIter + 1
                    T_s_Old = T_s

                if iIter>=iIterMax:
                    print('Maximum number of iterations exceeded') 

                # Convert W/m^2 to kJ/hr
                q_sens = q_sens_watt * 3.6 * p.Aw           # Convert W/m^2_cultivated to kJ/hr #Sensible gain to air from vegetation [kJ/hr] (+ = heating air, - = cooling)
                q_lat = q_lat_watt * 3.6 * p.Aw    # Convert W/m^2_cultivated to kJ/hr #Latent gain to air from vegetation [kJ/hr] (+ = heating air, - = cooling)
                Rnet = Rnet * 3.6 * p.Aw                   # Convert W/m^2 to kJ/hr  #Net radiation avaiblable to the canopy [kJ/h]
                q_sw_EL = G_EL * 3.6 * p.Aw                 # Convert W/m^2 to kJ/hr #SW radiative heat reaching floor[kJ/h]
                latentkg = q_lat_watt * 3.6 / lam * p.Aw    

            else:
                q_sens = 0.0
                q_lat = 0.0
                Rnet = 0.0
                q_sw_EL = PAR * 3.6 * p.Aw   # Convert W/m^2 to kJ/hr 
                latentkg = 0.0

            qLEDs[i] = 0.11*(Rnet+q_sw_EL)/0.52 + 0.37*(Rnet+q_sw_EL)/0.52 + q_sw_EL  #Heat gain from lights [kJ/hr] - fractions based on Talbot & Monfet (2024)
            qPlant[i] = q_sens+q_lat #heat gain to air from crops (sensible and latent heat)

            #Transpiration rate
            PHItransp[i] = latentkg #[kg_water/hr]

    return PHIphotc, RespCO2, Xsdw_, Xnsdw_, SHT_FW_, PHItransp, cycle_length, qLEDs, qPlant, no_harvests, HarvestedFW, AHi, AHo

# %%
#Remember the /wk is only bc of the assumption that these totals are harvested each week (again based on the 49-day/stagger cycle) - otherwise it's just /cycle
def massbalancesetup(p,Crop_density,Phot,Resp,SHT_FW_,harv,no_harv,transp,hours,Phot2=[0],Resp2=[0],SHT_FW_2=[0],harv2=0,no_harv2=0,transp2=[0],Predhours=0):
    """If using Historical or Predicted values ONLY, submit values to first phot,resp, etc ONLY.
    If using combined Historical + Predicted values, submit historical values FIRST, then predicted as the 2 values
    AKA: The 2nd values are only applicable if you're trying to see the accumulation of the combined cycle
    SHT_FW_ is fresh weight in g/plant
    harv is fresh weight in g/plant of what was harvested before current cycle, if timespan account for more than one cycle"""
    
    mwtransp = (sum(transp)+sum(transp2))   #plant water use [kg_water/wk]
    mCplant = ((sum(Phot)+sum(Phot2)) - sum(Resp)-sum(Resp2))*p.Aw    #plant net CO2 use [kg/wk]
    
    if harv2 != 0:
        harv = harv + harv2
    
    if SHT_FW_2 != [0]:
        XfwFinal = SHT_FW_2
        cyclehours = Predhours
    else:
        XfwFinal = SHT_FW_
        cyclehours = hours
    
    FinalYield = (XfwFinal[len(XfwFinal)-1]+harv)*Crop_density*p.Aw/1000     #TOTAL yield, fresh weight [kg/wk]  
    NewBiomass = (XfwFinal[len(XfwFinal)-1]+harv)-SHT_FW_[0]     #Biomass accumulated in just this time frame (fw) per plant [kg/plant]
    NewBiomass = NewBiomass*Crop_density*p.Aw/1000          #Biomass accumulated in just this time frame (fw) for area [kg]
    no_harv = no_harv+no_harv2
    
    return mwtransp, mCplant, FinalYield, NewBiomass, no_harv, cyclehours


"""Brief explanation of LED energy calculations.
            From farm equipment specifications:
             Elight = PPFD_nurs_red * 1.6 * 2.893 / 4.06
                + PPFD_nurs_blue * 1.6 * 2.893 / 2.8
                + PPFD_cult_red * 30.096 * 2.893 / 4.06
                + PPFD_cult_blue * 30.096 * 2.893 / 2.8
                
                Future development can see R:B ratios added as a parameter. 
                So for example... 70 PPFD B and 180 PPFD R: 0.28 B, 0.72 R
                                  70 PPFD B and 270 PPFD R: 0.21 B, 0.79 R
                For now, because this is not yet a parameter and the crop model doesn't yet account for growth difference in R and B, 
                we assume constant ratios for each section. Still, energy consumption predictions can be adjusted accordingly with this assumption.
                
                In that case:
                Elight = PPFD_cult_red * 30.096 * 2.893 / 4.06 + PPFD_cult_blue * 30.096 * 2.893 / 2.8
                Elight = (PPFD_cult_red/4.06 + PPFD_cult_blue/2.8) * 30.096 * 2.893
                     but 0.72*PPFDc = PPFD_cult_red and 0.28*PPFDc = PPFD_cult_blue
                Elight  = (0.72/4.06 + 0.28/2.8) * PPFDc * 30.096 * 2.893 """


# %%
def energybalance(qLEDs,qPlant, Ti_avg, hourly_timestamps, To, Ti_set, TRANSP, AHi, AHo, Epump, p):
    """qLEDs [kJ/hr] and qPlant [kJ/hr] summed from ALL sections.
    This only functions correctly if To, hourly_timestamps end at end of TimeSpan"""
    
    hrs = len(qLEDs)
    offset = 0
    Eequip = [0]*hrs
    Ehvac = [0]*hrs
    Elight = [0]*hrs
    Etot = [0]*hrs
    Ecost_hrly = [0]*hrs
    ti = 0
    unmet = 0
    Qhvac_switch = 0
    recycled_w = 0
    diff = 0
    mdot_econ = [0]*hrs

    mleak = (p.ACHleak/100)*p.V          #[m3/hr]  
    mair = p.ACHvent*p.V           #[m3/hr]  
    m_dot_air = 1.2*mair/3600      #[kg air/s] 1.2 = air density
    
    if isinstance(Ti_set, (int, float)):
        Ti_set = [Ti_set]*24
    
    #Hours when lights are on/off
    Set1 = 1
    Set2 = 0
    CultEndTime = p.CultStart + p.iDAYL
    if CultEndTime > 24:
        CultEndTime = CultEndTime - 24   #If lighting is on through the next morning, add 24 hours to HT.hour
        Set1 = 0
        Set2 = 1
    Set3 = 1
    Set4 = 0
    NursEndTime = p.NursStart + p.PPN
    if NursEndTime > 24:
        NursEndTime = NursEndTime - 24   #If lighting is on through the next morning, sub 24 hrs from end time and switch on vs off times
        Set3 = 0
        Set4 = 1
      
    #Now, calculating hourly energy use and associated cost
    for i in range(hrs):
        HT = hourly_timestamps[i+offset]
        
        if p.CultStart <= HT.hour < CultEndTime or CultEndTime <= HT.hour < p.CultStart:    #between the two times within the same day - "or" statement as lights off can either be earlier or later in the day depending on schedule
            ECL = (0.72/4.06 + 0.28/2.8) * p.PPFDc * 30.096 * 2.893 *3600*Set1 #[J]    #30.096 = cult LED area, 2.893 = conversion factor for area that lights cover compared to area that LED panels contain
        else:
            ECL = (0.72/4.06 + 0.28/2.8) * p.PPFDc * 30.096 * 2.893 *3600*Set2  #If outside of range, turns off - unless photoperiod crosses between days, then it turns on
        if p.NursStart <= HT.hour < NursEndTime or NursEndTime <= HT.hour < p.NursStart:    
            ENL = (0.79/4.06 + 0.21/2.8) * p.PPFDn * 1.6 * 2.893 *3600*Set3 #[J]   
        else:
            ENL = (0.79/4.06 + 0.21/2.8) * p.PPFDn * 1.6 * 2.893 *Set4
        
        Elight[i] = (ECL + ENL)/3600000 #light energy use, [kW]
        
        Eequip[i] = (Epump + p.Eaerator + p.Einj + p.Erecirpump)/24    #[kWh/hr] ------individual energy uses currently provided as /day, if adjust to /hr, remove "/24"
                                                           #Really need these alterable by when they're actually on
        # Economizer stuff 
        C_eff = 2*(10**(5))          # J/K (effective heat capacity of container + plants + structure)
        deadband = 0.5       # °C (allowed drop below setpoint)
        Ti_min_allowed = Ti_set[ti] - deadband
        
        Ti_set_prev = Ti_set[ti]
        ti += 1
        if ti == len(Ti_set):
            ti = 0

        # This stuff prevents the economizer from constantly switching back and forth due to tiny changes
        min_econ_on_hours = 1   # hysteresis timers
        min_econ_off_hours = 1

        # Persistent state variables (init before loop)
        econ_state = False
        econ_on_timer = 0
        econ_off_timer = 0

        Ti_current = Ti_avg[i+offset]       # indoor temp at start of hour [°C]
        To_current = To[i+offset]       # outdoor air temp [°C]
        dt = 3600              # seconds per timestep (1 hr)

        Qvent_full = m_dot_air * p.c_p_air * (To_current - Ti_current)  # [W] potential economizer load (- if cooling)

        # Predict indoor temp change if economizer is fully open
        dT_full = (Qvent_full * dt) / C_eff
        Ti_predicted_full = Ti_current + dT_full

        # Economizer control logic
        if To_current >= Ti_current:   # outdoor not cooler -> no benefit
            Qvent_used = 0.0
            econ_state = False
            econ_on_timer = 0
            econ_off_timer += 1

        else:  # outdoor cooler
            if Ti_predicted_full >= Ti_min_allowed:
                # Full economizer OK
                if (not econ_state) and (econ_off_timer < min_econ_off_hours):
                    Qvent_used = 0.0
                    econ_state = False
                    econ_off_timer += 1
                else:
                    Qvent_used = Qvent_full
                    if not econ_state:
                        econ_state = True
                        econ_on_timer = 1
                        econ_off_timer = 0
                    else:
                        econ_on_timer += 1
            else:
                # Overcooling risk: scale Qvent so final Ti = Ti_min_allowed
                dT_allowed = Ti_min_allowed - Ti_current  # ≤ 0
                Qvent_allowed = (dT_allowed * C_eff) / dt  # W
                # Only allow cooling (negative Qvent)
                if Qvent_full < 0:
                    Qvent_used = max(Qvent_allowed, Qvent_full)
                else:
                    Qvent_used = 0.0

                if Qvent_used != 0.0:
                    if not econ_state:
                        econ_state = True
                        econ_on_timer = 1
                        econ_off_timer = 0
                    else:
                        econ_on_timer += 1
                else:
                    econ_state = False
                    econ_off_timer += 1

        Qvent_used = Qvent_used*3.41 #Convert W to BTU/hr

        mecon = mair*Qvent_used/Qvent_full 
            
        mdot_econ[i] = mecon #mecon = (Qvent_used/(c_p_air * (To_current - Ti_current)))*3600/1.2  #[m3/hr] Approximate airflow from economizer, needed for moisture changes later
        
        mwinHR = TRANSP[i]+diff-0.001*((mleak+mecon)*AHi[i]-(mecon+mleak)*AHo[i])
        if mwinHR > 1.75*3.7854:  #Aka, plants put more water vapor in the air than the dehumidifer can handle
            recycled_w = 1.75*3.79   #Max 1.75 gal/hr (3.79 kg water = 1 gal)
            diff = mwinHR - 1.75*3.7854  # How much new water vapor was not accounted for by the dehumidifier in this time step
        elif mwinHR < 0:  # AKA, more vapor exited than entered the system
            recycled_w = 0
            diff = 0
        else:
            recycled_w = mwinHR   #kg water transpired that hour
            diff = 0  #Dehumidifier has caught up

        mwinHR = mwinHR - recycled_w #[kg/hr] new water entering system within this hour
        Qlatent = recycled_w * 2.5*10**6 #[J/hr] latent heat (to be removed from the system by dehumidifier)
        Qlatent = Qlatent/1055 #[BTU/hr]
        
        QL = qLEDs[i]*1000/1055 #[BTU/hr] heat from lighting
        #QL = (ECL*(1-EffCL) + ENL*(1-EffNL))/1055    #[BTU/hr] heat from lighting
        dT_F = (To[i+offset] - Ti_avg[i+offset]) * 9/5
        Qf = dT_F * p.U   #%Heat exchange through walls [BTU / hr] (in = +, out = -, will vary by season)
        Qplant = qPlant[i]*1000/1055 #[BTU/hr] - heat ENTERING system from plant
        Qequip = (1-p.EffEq)*(Eequip[i]+p.Efan/24)*3412 #[BTU/hr] 

        Qhvac = Qf + Qequip + QL + Qplant + Qvent_used + unmet + Qhvac_switch + Qlatent
        
        N_ramp = 0.25   # set to 1 for full change in 1 hour, 2 for spread over 2 hours, etc.
        if Ti_set[ti] != Ti_set_prev: #in next timestep, will need to change temps
            deltaT = Ti_set[ti] - Ti_set_prev
            Qhvac_switch =  ((C_eff * deltaT) / (N_ramp * dt)) * 3.412   # [BTU/hr]
        else:
            Qhvac_switch = 0
        
        if abs(Qhvac) > p.Qhvac_max:
            Qmax = np.sign(Qhvac)*p.Qhvac_max
            unmet += Qhvac - Qmax 
            unmet = np.clip(unmet, -1e12, 1e12)  # added to fix overflow error
            Qhvac = Qmax
        else:
            unmet = 0
        
        Ehvac[i] = abs(Qhvac/(p.EER*1000)) + p.Efan/24        #HVAC energy use [kWh]

        Etot[i] = Ehvac[i] + Eequip[i] + Elight[i]  #kWh

        #Natural gas, electricity, heat inputs for CO2 production and enrichment
        #ECO2 = 1.0 ####
        #Etot = Etot + ECO2
        
        #Calculating the holidays that change every year (this is in the "if" loop in case timespan crosses into new year)
        labor_day = 1 + (7 - date(HT.year, 9, 1).weekday()) %7  # Labor Day is the first Monday of September
        memorial_day = 31 - date(HT.year, 5, 31).weekday()  #Memorial Day is last Monday in May
        thanksgiving = (1+ (3 - date(HT.year, 11, 1).weekday() + 7) % 7) + 7*3 #Thanksgiving is 4th Thursday of Nov
        
        if HT.day_of_week <5:    #Peak hours only Mon-Fri. Mon = 0, Fri = 4, Sun = 6
            if (3 < HT.month < 11) and (12<=HT.hour<21) and (not ((HT.month == 5 and HT.day == memorial_day) or (HT.month == 7 and HT.day == 4) or (HT.month == 9 and HT.day == labor_day))): #Apr-Oct, 12-9pm, excl Memorial Day, July 4, and Labor Day
                Ecost_hrly[i] = Etot[i]* p.ElecRateP
            if (HT.month > 10 or HT.month <4) and (6<=HT.hour<10 or 18<=HT.hour<22) and (not (HT.month == 11 and HT.day == thanksgiving) or (HT.month == 12 and HT.day == 25) or (HT.month == 1 and HT.day == 1)):   #Nov-Mar, 6am-10am or 6pm-10pm, excl Thanksgiving, Christmas, or New Years Day
                Ecost_hrly[i] = Etot[i]* p.ElecRateP
            else:
                Ecost_hrly[i] = Etot[i]* p.ElecRateOP
        else: 
            Ecost_hrly[i] = Etot[i]* p.ElecRateOP

    return Eequip, Ehvac, Elight, Etot, Ecost_hrly, mdot_econ  


#### Same as previous energybalance(), but here, PPFDc is not constant when LEDs are on
def energybalance2(qLEDs,qPlant, Ti_avg, hourly_timestamps, To, Ti_set, TRANSP, AHi, AHo, Epump, p):   
    """qLEDs [kJ/hr] and qPlant [kJ/hr] summed from ALL sections.  TEMPORARILY ADDED THINGS
    This only functions correctly if To, hourly_timestamps end at end of TimeSpan"""
    
    hrs = len(qLEDs)
    offset = 0 
    Eequip = [0]*hrs
    Ehvac = [0]*hrs
    Elight = [0]*hrs
    Etot = [0]*hrs
    Ecost_hrly = [0]*hrs
    mdot_econ = [0]*hrs
    ti = 0
    unmet = 0
    Qhvac_switch = 0
    recycled_w = 0
    diff = 0
    PPFDc1 = p.PPFDc[0]
    PPFDc2 = p.PPFDc[1]
    iDAYL1 = p.PPFDc[2]

    mleak = (p.ACHleak/100)*p.V          #[m3/hr] 
    mair = p.ACHvent*p.V           #[m3/hr]  
    m_dot_air = 1.2*mair/3600      #[kg air/s] 1.2 = air density
    
    if isinstance(Ti_set, (int, float)):
        Ti_set = [Ti_set]*24
    
    #Hours when lights are on/off
    Set1 = 1
    Set2 = 0
    CultEndTime = p.CultStart + iDAYL1
    if CultEndTime > 24:
        CultEndTime = CultEndTime - 24   #If lighting is on through the next morning, add 24 hours to HT.hour
        Set1 = 0
        Set2 = 1
    Set5 = 1
    Set6 = 0
    CultEndTime2 = p.CultStart + p.iDAYL
    if CultEndTime2 > 24:
        CultEndTime2 = CultEndTime2 - 24   #If lighting is on through the next morning, add 24 hours to HT.hour
        Set5 = 0
        Set6 = 1
    Set3 = 1
    Set4 = 0
    NursEndTime = p.NursStart + p.PPN
    if NursEndTime > 24:
        NursEndTime = NursEndTime - 24   #If lighting is on through the next morning, sub 24 hrs from end time and switch on vs off times
        Set3 = 0
        Set4 = 1
      
    #Now, calculating hourly energy use and associated cost
    for i in range(hrs):
        HT = hourly_timestamps[i+offset]
        if p.CultStart < CultEndTime:  # Lights on period 1
            if CultStart <= HT.hour < CultEndTime:
                ECL1 = (0.72/4.06 + 0.28/2.8) * PPFDc1 * 30.096 * 2.893 *3600 #[J]
            else:
                ECL1 = 0
        else:
            if CultEndTime <= HT.hour < p.CultStart:
                ECL1 = 0
            else:
                ECL1 = (0.72/4.06 + 0.28/2.8) * PPFDc1 * 30.096 * 2.893 *3600 #[J]
        
        if CultEndTime < CultEndTime2:  #Lights on period 2
            if CultEndTime <= HT.hour < CultEndTime2:
                ECL2 = (0.72/4.06 + 0.28/2.8) * PPFDc2 * 30.096 * 2.893 *3600 #[J]   
            else:
                ECL2 = 0
        else:
            if CultEndTime2 <= HT.hour < CultEndTime:
                ECL2 = 0
            else:
                ECL2 = (0.72/4.06 + 0.28/2.8) * PPFDc2 * 30.096 * 2.893 *3600 #[J]
        
        if p.NursStart < NursEndTime:  #Nursery station
            if p.NursStart <= HT.hour < NursEndTime:
                ENL = (0.79/4.06 + 0.21/2.8) * p.PPFDn * 1.6 * 2.893 *3600 #[J]
            else:
                ENL = 0
        else:
            if NursEndTime <= HT.hour < p.NursStart:
                ENL = 0
            else:
                ENL = (0.79/4.06 + 0.21/2.8) * p.PPFDn * 1.6 * 2.893 *3600 #[J]
        
        ECL = max(ECL1, ECL2)  #whichever one is actually on
        
        Elight[i] = (ECL + ENL)/3600000 #light energy use, [kW]
        
        Eequip[i] = (Epump + p.Eaerator + p.Einj + p.Erecirpump)/24    #[kWh/hr] ------individual energy uses currently provided as /day, if adjust to /hr, remove "/24"
                                                           
        # Economizer stuff 
        C_eff = 2*(10**(5))          # J/K (effective heat capacity of container + plants + structure)
        deadband = 0.5       # °C (allowed drop below setpoint)
        Ti_min_allowed = Ti_set[ti] - deadband
        
        Ti_set_prev = Ti_set[ti]
        ti += 1
        if ti == len(Ti_set):
            ti = 0

        # This prevents the economizer from constantly switching back and forth due to tiny changes
        min_econ_on_hours = 1   # hysteresis timers
        min_econ_off_hours = 1

        # Persistent state variables (init before loop)
        econ_state = False
        econ_on_timer = 0
        econ_off_timer = 0

        Ti_current = Ti_avg[i+offset]       # indoor temp at start of hour [°C]
        To_current = To[i+offset]       # outdoor air temp [°C]
        dt = 3600              # seconds per timestep (1 hr)

        Qvent_full = m_dot_air * p.c_p_air * (To_current - Ti_current)  # [W] potential economizer load (- if cooling)

        # Predict indoor temp change if economizer is fully open
        dT_full = (Qvent_full * dt) / C_eff
        Ti_predicted_full = Ti_current + dT_full

        # Economizer control logic
        if To_current >= Ti_current:   # outdoor not cooler -> no benefit
            Qvent_used = 0.0
            econ_state = False
            econ_on_timer = 0
            econ_off_timer += 1

        else:  # outdoor cooler
            if Ti_predicted_full >= Ti_min_allowed:
                # Full economizer OK
                if (not econ_state) and (econ_off_timer < min_econ_off_hours):
                    Qvent_used = 0.0
                    econ_state = False
                    econ_off_timer += 1
                else:
                    Qvent_used = Qvent_full
                    if not econ_state:
                        econ_state = True
                        econ_on_timer = 1
                        econ_off_timer = 0
                    else:
                        econ_on_timer += 1
            else:
                # Overcooling risk: scale Qvent so final Ti = Ti_min_allowed
                dT_allowed = Ti_min_allowed - Ti_current  # ≤ 0
                Qvent_allowed = (dT_allowed * C_eff) / dt  # W
                # Only allow cooling (negative Qvent)
                if Qvent_full < 0:
                    Qvent_used = max(Qvent_allowed, Qvent_full)
                else:
                    Qvent_used = 0.0

                if Qvent_used != 0.0:
                    if not econ_state:
                        econ_state = True
                        econ_on_timer = 1
                        econ_off_timer = 0
                    else:
                        econ_on_timer += 1
                else:
                    econ_state = False
                    econ_off_timer += 1

        Qvent_used = Qvent_used*3.41 #Convert W to BTU/hr

        mecon = mair*Qvent_used/Qvent_full 
            
        mdot_econ[i] = mecon #(Qvent_used/(c_p_air * (To_current - Ti_current)))*3600/1.2  #[m3/hr] Approximate airflow from economizer, needed for moisture changes later
        
        mwinHR = TRANSP[i]+diff-0.001*((mleak+mecon)*AHi[i]-(mecon+mleak)*AHo[i])
        if mwinHR > 1.75*3.7854:  #Aka, plants put more water vapor in the air than the dehumidifer can handle
            recycled_w = 1.75*3.79   #Max 1.75 gal/hr (3.79 kg water = 1 gal)
            diff = mwinHR - 1.75*3.7854  # How much new water vapor was not accounted for by the dehumidifier in this time step
        elif mwinHR < 0:  # AKA, more vapor exited than entered the system
            recycled_w = 0
            diff = 0
        else:
            recycled_w = mwinHR   #kg water transpired that hour
            diff = 0  #Dehumidifier has caught up

        mwinHR = mwinHR - recycled_w #[kg/hr] new water entering system within this hour
        Qlatent = recycled_w * 2.5*10**6 #[J/hr] latent heat (to be removed from the system by dehumidifier)
        Qlatent = Qlatent/1055 #[BTU/hr]
        
        QL = qLEDs[i]*1000/1055 #[BTU/hr] heat from lighting
        #QL = (ECL*(1-EffCL) + ENL*(1-EffNL))/1055    #[BTU/hr] heat from lighting
        dT_F = (To[i+offset] - Ti_avg[i+offset]) * 9/5
        Qf = dT_F * p.U   #%Heat exchange through walls [BTU / hr] (in = +, out = -, will vary by season)
        Qplant = qPlant[i]*1000/1055 #[BTU/hr] - heat ENTERING system from plant
        Qequip = (1-p.EffEq)*(Eequip[i]+p.Efan/24)*3412 #[BTU/hr] 

        Qhvac = Qf + Qequip + QL + Qplant + Qvent_used + unmet + Qhvac_switch + Qlatent
        
        N_ramp = 0.25   # set to 1 for full change in 1 hour, 2 for spread over 2 hours, etc.
        if Ti_set[ti] != Ti_set_prev: #in next timestep, will need to change temps
            deltaT = Ti_set[ti] - Ti_set_prev
            Qhvac_switch =  ((C_eff * deltaT) / (N_ramp * dt)) * 3.412   # [BTU/hr]
        else:
            Qhvac_switch = 0
        
        if abs(Qhvac) > p.Qhvac_max:
            Qmax = np.sign(Qhvac)*p.Qhvac_max
            unmet += Qhvac - Qmax  
            unmet = np.clip(unmet, -1e12, 1e12)
            Qhvac = Qmax
        else:
            unmet = 0
        
        Ehvac[i] = abs(Qhvac/(p.EER*1000)) + p.Efan/24        #HVAC energy use [kWh]

        Etot[i] = Ehvac[i] + Eequip[i] + Elight[i]  #kWh
        
        #Natural gas, electricity, heat inputs for CO2 production and enrichment
        #ECO2 = 1.0 ####
        #Etot = Etot + ECO2
        
        #Calculating the holidays that change every year (this is in the "if" loop in case timespan crosses into new year)
        labor_day = 1 + (7 - date(HT.year, 9, 1).weekday()) %7  # Labor Day is the first Monday of September
        memorial_day = 31 - date(HT.year, 5, 31).weekday()  #Memorial Day is last Monday in May
        thanksgiving = (1+ (3 - date(HT.year, 11, 1).weekday() + 7) % 7) + 7*3 #Thanksgiving is 4th Thursday of Nov
        
        if HT.day_of_week <5:    #Peak hours only Mon-Fri. Mon = 0, Fri = 4, Sun = 6
            if (3 < HT.month < 11) and (12<=HT.hour<21) and (not ((HT.month == 5 and HT.day == memorial_day) or (HT.month == 7 and HT.day == 4) or (HT.month == 9 and HT.day == labor_day))): #Apr-Oct, 12-9pm, excl Memorial Day, July 4, and Labor Day
                Ecost_hrly[i] = Etot[i]* p.ElecRateP
            if (HT.month > 10 or HT.month <4) and (6<=HT.hour<10 or 18<=HT.hour<22) and (not (HT.month == 11 and HT.day == thanksgiving) or (HT.month == 12 and HT.day == 25) or (HT.month == 1 and HT.day == 1)):   #Nov-Mar, 6am-10am or 6pm-10pm, excl Thanksgiving, Christmas, or New Years Day
                Ecost_hrly[i] = Etot[i]* p.ElecRateP
            else:
                Ecost_hrly[i] = Etot[i]* p.ElecRateOP
        else: 
            Ecost_hrly[i] = Etot[i]* p.ElecRateOP

    return Eequip, Ehvac, Elight, Etot, Ecost_hrly, mdot_econ  


def run_past(TimeSpan,weather, RHi, iCO2, Ti_set, p, Correction = 1.46):
    """Use this equation to estimate crop growth prior to study period"""
    Vrad = [0]*(24*p.cycle_days)
    PAR_ = [0]*(24*p.cycle_days)
    To = weather[0]
    RHo = weather[1]
    atmCO2 = weather[2]
    hourly_timestamps = weather[3]
    
    Ti1, Ti2, Ti3, Ti4, Ti5, Ti6, Ti7, Ti8, Ti_avg, TiN, weeks = get_indoor_temps(p,Ti_set)
    
    RHi = RHi/100
    Crop_density = p.heads/p.Aw    #Number of crops per unit area [crops/m2]

    ##Get photoperiod
    NursHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - nursery period] (24 hrs x 21 days)
    CultHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime,24*p.cycle_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - cult period] (24 hrs x 28 days) #Until the other cycle is transplanted, the lights here count as heat gain for this section (assuming lights do not turn off when space is empty)

    if isinstance(p.PPFDc, (int, float)):  #if PPFDc is always the same
        for hour in NursHours:
            val = p.PPN/(((((hour-p.NursStart+1+p.SeedingTime)/24)-0.00001)%1)*24)  #-NursStart to offset based on time lights turn on, and +SeedingTime to offset full set to start at same time as the cycle's start      
            if val > 1:
                Vrad[hour] = p.PPFDn #umol/m2/s   
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision

        for hour in CultHours:
            val = p.iDAYL/(((((hour-p.CultStart+1+p.SeedingTime)/24)-0.00001)%1)*24)
            if val > 1:
                Vrad[hour] = p.PPFDc #umol/m2/s 
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision
            
        conditions = [Vrad, PAR_, p.PPFDn, p.PPFDc, RHi, iCO2, Crop_density, TiN, To, RHo, Correction] 

    else:  #PPFDc in form [PPFDc1, PPFDc2, iDAYL1] (hours until PPFD switches -- must be <iDAYL)
        iDAYL1 = p.PPFDc[2]
        CultChange = p.CultStart + iDAYL1
        if CultChange > 24:
            CultChange = CultChange - 24
        iDAYL2 = p.iDAYL - iDAYL1    

        for hour in NursHours:
            val = p.PPN/(((((hour-p.NursStart+1+p.SeedingTime)/24)-0.00001)%1)*24)  
                    #-NursStart to offset based on time lights turn on, and +SeedingTime to offset full set to start at same time as the cycle's start
            if val > 1:
                Vrad[hour] = p.PPFDn #umol/m2/s   
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision

        for hour in CultHours:
            val = iDAYL1/(((((hour-p.CultStart+1+p.SeedingTime)/24)-0.00001)%1)*24)
            val2 = iDAYL2/(((((hour-CultChange+1+p.SeedingTime)/24)-0.00001)%1)*24)
            if val > 1:
                Vrad[hour] = p.PPFDc[0] #umol/m2/s
            elif val2>1:
                Vrad[hour] = p.PPFDc[1]
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision
        
        conditions = [Vrad, PAR_, p.PPFDn, int(p.PPFDc[0]), RHi, iCO2, Crop_density, TiN, To, RHo, Correction]   #Uses first PPFD as one for calculating slopes
    
    
    Tis = [Ti1,Ti2,Ti3,Ti4,Ti5,Ti6,Ti7,Ti8,Ti1,Ti2,Ti3,Ti4,Ti5,Ti6,Ti7,Ti8]
    StudyStart = max(weeks)
    Sections = []
    
    for (Ti, week) in zip(Tis,weeks):
        Phot, Resp, Xsd, Xnsd, FW, transp, hours, QL, QP, no_harv1, harv, ahi, aho = cropmodel(conditions,p, Ti[0:StudyStart*7*24-1], int(7*week))
        if week <0:
            week = 0
            Xsd = [0]
            Xnsd = [0]
        Sections.append([Phot, Resp, Xsd, Xnsd, FW, transp, hours, QL, QP, no_harv1, harv, ahi, aho, week, Ti])
            
    return Sections
        

def run_future(Sections, TimeSpan,weather, RHi, iCO2, Ti_set, p, Correction = 1.46):
    """Use this to estimate crop growth during study period"""
    Vrad = [0]*(24*p.cycle_days)
    PAR_ = [0]*(24*p.cycle_days)
    To = weather[0]
    RHo = weather[1]
    atmCO2 = weather[2]
    hourly_timestamps = weather[3]
    
    Ti1, Ti2, Ti3, Ti4, Ti5, Ti6, Ti7, Ti8, Ti_avg, TiN, weeks = get_indoor_temps(p,Ti_set)
    
    RHi = RHi/100
    Crop_density = p.heads/p.Aw    #Number of crops per unit area [crops/m2]
    Epump = p.EpumpNURS + p.EpumpCULT
    ##Get photoperiod
    NursHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - nursery period] (24 hrs x 21 days)
    CultHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime,24*p.cycle_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - cult period] (24 hrs x 28 days) #Until the other cycle is transplanted, the lights here count as heat gain for this section (assuming lights do not turn off when space is empty)

    if isinstance(p.PPFDc, (int, float)):  #if PPFDc is always the same
        for hour in NursHours:
            val = p.iDAYL/(((((hour-p.NursStart+1+p.SeedingTime)/24)-0.00001)%1)*24)  #-NursStart to offset based on time lights turn on, and +SeedingTime to offset full set to start at same time as the cycle's start       #################### SET PPN TO 16 FOR HISTORICAL RIGHT NOW - CHANGE BACK LATER
            if val > 1:
                Vrad[hour] = p.PPFDn #umol/m2/s   
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision

        for hour in CultHours:
            val = p.iDAYL/(((((hour-p.CultStart+1+p.SeedingTime)/24)-0.00001)%1)*24)
            if val > 1:
                Vrad[hour] = p.PPFDc #umol/m2/s #############TEMP 1
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision
            
        conditions = [Vrad, PAR_, p.PPFDn, p.PPFDc, RHi, iCO2, Crop_density, TiN, To, RHo, Correction]   #Uses first PPFD as one for calculating slopes

    else:  #PPFDc in form [PPFDc1, PPFDc2, iDAYL1] (hours until PPFD switches -- must be <iDAYL)
        iDAYL1 = p.PPFDc[2]
        CultChange = p.CultStart + iDAYL1
        if CultChange > 24:
            CultChange = CultChange - 24
        iDAYL2 = p.iDAYL - iDAYL1    

        for hour in NursHours:
            val = p.PPN/(((((hour-p.NursStart+1+p.SeedingTime)/24)-0.00001)%1)*24)  
                    #-NursStart to offset based on time lights turn on, and +SeedingTime to offset full set to start at same time as the cycle's start
            if val > 1:
                Vrad[hour] = p.PPFDn #umol/m2/s   
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision

        for hour in CultHours:
            val = iDAYL1/(((((hour-p.CultStart+1+p.SeedingTime)/24)-0.00001)%1)*24)
            val2 = iDAYL2/(((((hour-CultChange+1+p.SeedingTime)/24)-0.00001)%1)*24)
            if val > 1:
                Vrad[hour] = p.PPFDc[0] #umol/m2/s
            elif val2>1:
                Vrad[hour] = p.PPFDc[1]
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision
        
        conditions = [Vrad, PAR_, p.PPFDn, int(p.PPFDc[0]), RHi, iCO2, Crop_density, TiN, To, RHo, Correction]   #Uses first PPFD as one for calculating slopes


    Tis = [Ti1,Ti2,Ti3,Ti4,Ti5,Ti6,Ti7,Ti8,Ti1,Ti2,Ti3,Ti4,Ti5,Ti6,Ti7,Ti8]
    StudyStart = max(weeks)
    mwtranspLIST = []
    mCplantLIST = []
    FinalYieldLIST = []
    NewBiomassLIST = []
    hoursLIST = []
    HarvestsLIST = []
    Harvest = 0 
    TRANSP = [0]*int(TimeSpan*24) #To get total farm transpiration by hour
    Qleds = [0]*int(TimeSpan*24) #Assumes TimeStep = 1
    Qplants = [0]*int(TimeSpan*24)
    AbsHumInt = pd.DataFrame()
    no = 0
    
    for idx, s in enumerate(Sections):
        Phot, Resp, Xsd, Xnsd, FW, transp, hours, QL, QP, no_harv1, harv, ahi, aho, week, Ti = s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14]

        Ti = Tis[idx] # This switches the Ti to the new Ti_set, if it differs between run_past and run_future

        PredPhot, PredResp, PredXsd, PredXnsd, PredFW, Predtransp, Predhours, PredQL, PredQP, no_harv2, Predharv, AHi, AHo = cropmodel(conditions,p, Ti[StudyStart*7*24:],TimeSpan,Xsdw0=Xsd[int(24*7*week)],Xnsdw0=Xnsd[int(24*7*week)],predict='Yes',past_hrs = hours)
        #massbalance setup
        mwtransp, mCplant, FinalYield, NB, no_harv, cyclehours = massbalancesetup(p,Crop_density,PredPhot,PredResp,PredFW,Predharv,no_harv2,Predtransp,Predhours)
        AbsHumInt[no] = AHi
        for i in range(len(Qleds)):
            Qleds[i] = Qleds[i] + PredQL[i]
            Qplants[i] = Qplants[i] + PredQP[i]
        for i in range(len(Predtransp)):
            TRANSP[i] = TRANSP[i] + Predtransp[i]
        mwtranspLIST.append(mwtransp)
        mCplantLIST.append(mCplant)
        FinalYieldLIST.append(FinalYield)
        NewBiomassLIST.append(NB)
        hoursLIST.append(cyclehours)
        HarvestsLIST.append(no_harv)
        Harvest = Harvest + (Predharv+harv)*Crop_density*p.Aw/1000
        no +=1
        
    df = pd.DataFrame({'Cycle':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'mwtransp':mwtranspLIST,'mCplant':mCplantLIST,'FinalYield':FinalYieldLIST,'NewBiomass':NewBiomassLIST,'hours':hoursLIST,'No_Harvests':HarvestsLIST})
    
    AHi = AbsHumInt.sum(axis=1).values/len(Sections)
    
    if isinstance(p.PPFDc, (int, float)):
        Eequip, Ehvac, Elight, Etot, Ecost_hrly, mdot_econ = energybalance(Qleds,Qplants, Ti_avg[StudyStart*7*24:], hourly_timestamps[StudyStart*7*24:], To[StudyStart*7*24:], Ti_set, TRANSP, AHi, AHo, Epump, p)   
    else:
        Eequip, Ehvac, Elight, Etot, Ecost_hrly, mdot_econ = energybalance2(Qleds,Qplants, Ti_avg[StudyStart*7*24:], hourly_timestamps[StudyStart*7*24:], To[StudyStart*7*24:], Ti_set, TRANSP, AHi, AHo, Epump, p)              # This one right now accounts for 2 PPFDs for cultivation area 
    biomass_prod = df['NewBiomass'].sum()
    Etotal = sum(Etot)
    mCplant = df['mCplant'].sum() #for the entire timespan - regardless of how long it is
    mwtransp = df['mwtransp'].sum() #for the entire timespan

    return mwtranspLIST, mCplantLIST, FinalYieldLIST, NewBiomassLIST, hoursLIST, HarvestsLIST, Harvest, Qleds, Qplants, mCplant, mwtransp, biomass_prod, Eequip, Ehvac, Elight, Etot, Etotal, Ecost_hrly, df, AbsHumInt, AHo, TRANSP, Ti_avg, mdot_econ
    
    
    

def run_model(TimeSpan,weather, RHi, iCO2, Ti_set, p, Correction = 1.46):
    """ Use this to get crop growth before and during study period """
    Vrad = [0]*(24*p.cycle_days)
    PAR_ = [0]*(24*p.cycle_days)
    To = weather[0]
    RHo = weather[1]
    atmCO2 = weather[2]
    hourly_timestamps = weather[3]
    
    Ti1, Ti2, Ti3, Ti4, Ti5, Ti6, Ti7, Ti8, Ti_avg, TiN, weeks = get_indoor_temps(p,Ti_set)
    
    RHi = RHi/100
    Crop_density = p.heads/p.Aw    #Number of crops per unit area [crops/m2]
    Epump = p.EpumpNURS + p.EpumpCULT

    ##Get photoperiod
    NursHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - nursery period] (24 hrs x 21 days)
    CultHours = range(24*p.nurs_days-(24-p.TransTime)-p.SeedingTime,24*p.cycle_days-(24-p.TransTime)-p.SeedingTime) #[hr of cycle - cult period] (24 hrs x 28 days) #Until the other cycle is transplanted, the lights here count as heat gain for this section (assuming lights do not turn off when space is empty)

    if isinstance(p.PPFDc, (int, float)):  #if PPFDc is always the same
        for hour in NursHours:
            val = p.PPN/(((((hour-p.NursStart+1+p.SeedingTime)/24)-0.00001)%1)*24)  #-NursStart to offset based on time lights turn on, and +SeedingTime to offset full set to start at same time as the cycle's start       
            if val > 1:
                Vrad[hour] = p.PPFDn #umol/m2/s   
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision

        for hour in CultHours:
            val = p.iDAYL/(((((hour-p.CultStart+1+p.SeedingTime)/24)-0.00001)%1)*24)
            if val > 1:
                Vrad[hour] = p.PPFDc #umol/m2/s #############TEMP 1
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision
            
        conditions = [Vrad, PAR_, p.PPFDn, p.PPFDc, RHi, iCO2, Crop_density, TiN, To, RHo, Correction]

    else:  #PPFDc in form [PPFDc1, PPFDc2, iDAYL1] (hours until PPFD switches -- must be <iDAYL)
        iDAYL1 = p.PPFDc[2]
        CultChange = p.CultStart + iDAYL1
        if CultChange > 24:
            CultChange = CultChange - 24
        iDAYL2 = p.iDAYL - iDAYL1    

        for hour in NursHours:
            val = p.PPN/(((((hour-p.NursStart+1+p.SeedingTime)/24)-0.00001)%1)*24)  
                    #-NursStart to offset based on time lights turn on, and +SeedingTime to offset full set to start at same time as the cycle's start
            if val > 1:
                Vrad[hour] = p.PPFDn #umol/m2/s   
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision

        for hour in CultHours:
            val = iDAYL1/(((((hour-p.CultStart+1+p.SeedingTime)/24)-0.00001)%1)*24)
            val2 = iDAYL2/(((((hour-CultChange+1+p.SeedingTime)/24)-0.00001)%1)*24)
            if val > 1:
                Vrad[hour] = p.PPFDc[0] #umol/m2/s
            elif val2>1:
                Vrad[hour] = p.PPFDc[1]
            PAR_[hour] = Vrad[hour]*0.217   #W/m2 --- 0.217 is conversion eq PAR solar rad, from Talbot & Monfet's decision
        
    
        conditions = [Vrad, PAR_, p.PPFDn, int(p.PPFDc[0]), RHi, iCO2, Crop_density, TiN, To, RHo, Correction]   #Uses first PPFD as one for calculating slopes


    Tis = [Ti1,Ti2,Ti3,Ti4,Ti5,Ti6,Ti7,Ti8,Ti1,Ti2,Ti3,Ti4,Ti5,Ti6,Ti7,Ti8]
    StudyStart = max(weeks)
    mwtranspLIST = []
    mCplantLIST = []
    FinalYieldLIST = []
    NewBiomassLIST = []
    hoursLIST = []
    HarvestsLIST = []
    Harvest = 0 
    TRANSP = [0]*int(TimeSpan*24) #To get total farm transpiration by hour
    Qleds = [0]*int(TimeSpan*24) #Assumes TimeStep = 1
    Qplants = [0]*int(TimeSpan*24)
    AbsHumInt = pd.DataFrame()
    no = 0
    
    ############# changed conditions
    for (Ti, week) in zip(Tis,weeks):
        Phot, Resp, Xsd, Xnsd, FW, transp, hours, QL, QP, no_harv1, harv, ahi, aho = cropmodel(conditions,p, Ti[0:StudyStart*7*24-1], int(7*week))
        if week <0:
            week = 0
            Xsd = [0]
            Xnsd = [0]
        
        PredPhot, PredResp, PredXsd, PredXnsd, PredFW, Predtransp, Predhours, PredQL, PredQP, no_harv2, Predharv, AHi, AHo = cropmodel(conditions,p, Ti[StudyStart*7*24:],TimeSpan,Xsdw0=Xsd[int(24*7*week)],Xnsdw0=Xnsd[int(24*7*week)],predict='Yes',past_hrs = hours)
        #massbalance setup
        mwtransp, mCplant, FinalYield, NB, no_harv, cyclehours = massbalancesetup(p,Crop_density,PredPhot,PredResp,PredFW,Predharv,no_harv2,Predtransp,Predhours)
        AbsHumInt[no] = AHi
        for i in range(len(Qleds)):
            Qleds[i] = Qleds[i] + PredQL[i]
            Qplants[i] = Qplants[i] + PredQP[i]
        for i in range(len(Predtransp)):
            TRANSP[i] = TRANSP[i] + Predtransp[i]
        mwtranspLIST.append(mwtransp)
        mCplantLIST.append(mCplant)
        FinalYieldLIST.append(FinalYield)
        NewBiomassLIST.append(NB)
        hoursLIST.append(cyclehours)
        HarvestsLIST.append(no_harv)
        Harvest = Harvest + (Predharv+harv)*Crop_density*p.Aw/1000
        no +=1
        
    df = pd.DataFrame({'Cycle':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'mwtransp':mwtranspLIST,'mCplant':mCplantLIST,'FinalYield':FinalYieldLIST,'NewBiomass':NewBiomassLIST,'hours':hoursLIST,'No_Harvests':HarvestsLIST})
    
    AHi = AbsHumInt.sum(axis=1).values/len(weeks)
    
    if isinstance(p.PPFDc, (int, float)):
        Eequip, Ehvac, Elight, Etot, Ecost_hrly, mdot_econ = energybalance(Qleds,Qplants, Ti_avg[StudyStart*7*24:], hourly_timestamps[StudyStart*7*24:], To[StudyStart*7*24:], Ti_set, TRANSP, AHi, AHo, Epump, p)   
    else:
        Eequip, Ehvac, Elight, Etot, Ecost_hrly, mdot_econ = energybalance2(Qleds,Qplants, Ti_avg[StudyStart*7*24:], hourly_timestamps[StudyStart*7*24:], To[StudyStart*7*24:], Ti_set, TRANSP, AHi, AHo, Epump, p)   # This one right now accounts for 2 PPFDs for cultivation area 
    biomass_prod = df['NewBiomass'].sum()
    Etotal = sum(Etot)
    mCplant = df['mCplant'].sum() #for the entire timespan - regardless of how long it is
    mwtransp = df['mwtransp'].sum() #for the entire timespan

    return mwtranspLIST, mCplantLIST, FinalYieldLIST, NewBiomassLIST, hoursLIST, HarvestsLIST, Harvest, Qleds, Qplants, mCplant, mwtransp, biomass_prod, Eequip, Ehvac, Elight, Etot, Etotal, Ecost_hrly, df, AbsHumInt, AHo, TRANSP, Ti_avg, mdot_econ


# %%
def get_impacts(TimeSpan,Ehvac,Elight,Eequip,mCplant, mwtransp, AbsHumInt, AHo, iCO2, p, TRANSP, atmCO2, mdot_econ):
    # Mass balance (finished)

    mleak = (p.ACHleak/100)*p.V          #[m3/hr]  
    mair = p.ACHvent*p.V           #[m3/hr]  
    m_dot_air = 1.2*mair/3600      #[kg air/s] 1.2 = air density

    seeds = (p.heads/p.GR)*TimeSpan/7           # estimated no. seeds needed IN THE WEEK for efficient harvest - times portion of the week accounted for
    plugs = p.heads*TimeSpan/7              # no. starter plugs needed per week - times portion of week accounted for
    trays = math.ceil(p.heads/288)                  #no. new trays needed per week (assuming no recycling)
    trays_used = (trays*TimeSpan/7)/(p.TrayLifetime*7)  # trays/week but split amongst all the days it will be used, and only accountig for portion of the week studied

    AHi = AbsHumInt.sum(axis=1).values/16  #AHo is calculated based on whichever was the last section to run - but should be the same for all of them (from Pred cropmodel())
                                           # 16 = no. of sections

    mwout = 0
    mCout = 0
    recycled_w = 0
    diff = 0
    ######## Right now, AHi/AHo are calculated from the last saved 
    for i in range(len(AHi)):    #assumes 1 timestep = 1 hour
            
        mecon = mdot_econ[i]  #[m3/hr] Air exchange from the economizer within the hour (calculated in energybalance())
        
        mwout += 0.001*((mecon+mleak)*AHi[i]-(mecon+mleak)*AHo[i])    #[kg] Water exiting the system through infiltration and ventilation  -- this will be POSITIVE if it is indeed exiting
        mCout += 0.001*(mecon + mair)*(0.0409*(iCO2 - atmCO2[i])*44.01)/1000     #CO2 released outdoors [kg]
        mwinHR = TRANSP[i]+diff-0.001*((mleak+mecon)*AHi[i]-(mecon+mleak)*AHo[i])
        if mwinHR > 1.75*3.7854:  #Aka, plants put more water vapor in the air than the dehumidifer can handle
            recycled_w += 1.75*3.79   #Max 1.75 gal/hr (3.79 kg water = 1 gal)
            diff = mwinHR - 1.75*3.7854  # How much new water vapor was not accounted for by the dehumidifier in this time step
        elif mwinHR < 0:  # AKA, more vapor exited than entered the system
            recycled_w += 0
            diff = 0
        else:
            recycled_w += mwinHR   #kg water transpired that hour
            diff = 0  #Dehumidifier has caught up

    mCO2in = mCplant + mCout        #CO2 needed to put into system [kg/wk]
    mwin = mwtransp + mwout + (3.79*90*p.cult_water_changes + 3.79*31*p.nurs_water_changes)*TimeSpan/7  # Assuming water changes is on a weekly basis
                                      #Cult tank is 90 gal tank  (1 gal = 3.79 kg water)
                                      #Nurs tank is 31 gal 
    water_out = (3.79*90*p.cult_water_changes + 3.79*31*p.nurs_water_changes)*TimeSpan/7

    fert = mwin*p.fppm*0.8        #liquid fertilizer (assume constant concentration w water) [g/wk] Assume only 80% of input concentration

    N = (0.03+0.08)*fert/1000     #nitrogen input [kg]
    P2O5 = 0.05*fert/1000        #phosphorus pentoxide [kg]
    K2O = 0.175*fert/1000        #potassium oxide input [kg]
    MgO = 0.025*fert/1000        #magnesium oxide input [kg]
    #NNO3 = 0.075*fert/1000     #nitric nitrogen [kg] not in Simapro
    NNH4 = 0.005*fert/1000     #ammoniacal nitrogen [kg]
    #CaO = 0.135*fert/1000      #calcium oxide [kg] not in Simapro
    
    #    # Inputs for Simapro

    ElecSima = SM.iloc[:, [1]]
    Units = SM.iloc[:,[0]]

    ImpactsHVAC = ElecSima*sum(Ehvac)
    ImpactsHVAC = ImpactsHVAC.rename(columns={"Electricity": "HVAC Electricity"})
    ImpactsLED = ElecSima*sum(Elight)
    ImpactsLED = ImpactsLED.rename(columns={"Electricity": "LED Electricity"})
    ImpactsEquip = ElecSima*sum(Eequip)
    ImpactsEquip = ImpactsEquip.rename(columns={"Electricity": "Equipment Electricity"})
    #ImpactsNatGas = NatGas*SM.iloc[:, [2]] 
    ImpactsLiqCO2 = mCO2in*SM.iloc[:,[3]]
    ImpactsWater = mwin*SM.iloc[:,[4]]
    ImpactsAmmNit = NNH4*SM.iloc[:,[5]]
    ImpactsK2O = K2O*SM.iloc[:,[6]]
    ImpactsP2O5 = P2O5*SM.iloc[:,[7]]
    ImpactsN = N*SM.iloc[:,[8]]
    ImpactsMgO = MgO*SM.iloc[:,[9]]
    ImpactsRockwool = plugs*SM.iloc[:,[10]]    # PER WEEK 
    ImpactsTrays = trays_used*SM.iloc[:,[11]]*0.17  # Assume 1 tray = 0.17 kg   
    ImpactsRunoff = water_out*SM.iloc[:,[12]]

    FullResults = pd.concat([Units, ImpactsHVAC, ImpactsLED, ImpactsEquip, ImpactsLiqCO2, ImpactsWater, ImpactsAmmNit, ImpactsK2O, ImpactsP2O5, ImpactsN, ImpactsMgO, ImpactsRockwool, ImpactsTrays, ImpactsRunoff], axis=1)
    FullResults['Total'] = FullResults.sum(axis=1, numeric_only=True)
    GWP = FullResults.loc['Global warming', 'Total']
    ET = FullResults.loc['Ecotoxicity', 'Total']
    EUT = FullResults.loc['Freshwater eutrophication', 'Total']
    return FullResults, GWP, ET, EUT, mwout, mCO2in

# %%
def get_profit(TimeSpan, mwtransp, Ehvac, Etot, Elight, Eequip, Ecost_hrly, biomass_prod, mwout, p):
    G = p.Goal_Harvest #[kg] needed to produce to fill orders/wk, just an example
    P = p.Selling_Price #[$] selling price per kg, just an example
    D = p.Demand #[kg] estimated local demand, just a guess and an example
    H = biomass_prod
    
    if H < G:
        Revenue = P*H
        #print("Predicted Revenue: $", Revenue,". Warning: Harvest will not fulfill orders. Underproduced by", G-H,"kg.")
    elif H >= G and D >= (H-G):
        Revenue = P*H
        #print("Predicted Revenue: $", Revenue)
        #print("Extra: 0 kg")
    elif H>G and D<(H-G):
        Revenue = P*(G+D)
        #print("Predicted Revenue: $", Revenue)
        #print("Extra:",H-G-D,"kg")
    
    seeds = (p.heads/p.GR)*TimeSpan/7           # estimated no. seeds needed IN THE WEEK for efficient harvest - times portion of the week accounted for
    plugs = p.heads*TimeSpan/7              # no. starter plugs needed per week - times portion of week accounted for
    trays = math.ceil(plugs/288)                  #no. new trays needed per week (assuming no recycling)
    
    mwin = mwtransp + mwout + (3.79*90*p.cult_water_changes + 3.79*31*p.nurs_water_changes)*TimeSpan/7  # Assuming water changes is on a weekly basis
    fert = mwin*p.fppm 

    WaterCharge = p.WaterRate*(mwin*0.26)/1000         #[$/wk] convert to gallons, then charge WaterRate*gal
    WaterCost = round(WaterCharge + (p.WaterBC/4)*TimeSpan/7, 2)      #[$/wk] add in base monthly water charge, split evenly amongst the 4 weeks in a month.
    FertCost = round((fert/p.FertAmt)*p.FertPrice, 2)  #[$/wk] fert [g] divided by amount in bottle (FertAmt) times bottle price (FertPrice)
    TrayCost = (trays/p.TraysAmt)*p.TraysPrice #[$/wk] cost per tray
    TrayCost = round(TrayCost/p.TrayLifetime, 2)  #Accounts for recycling of trays
    MediaCost = round((plugs/p.PlugsAmt)*p.PlugsPrice, 2) #[$/wk] plugs used divided by no. in pack times price
    SeedCost = round((seeds/p.SeedsAmt)*p.SeedsPrice, 2)  #[$/wk] seeds used (accounted for germination rate) divided by seeds in pack, times pack cost
    ElecTotalCost = round(sum(Ecost_hrly) + (p.ElecBC/4)*TimeSpan/7, 2)  #[$/wk]

    EhvacCost = round(sum(Ecost_hrly)*(sum(Ehvac)/sum(Etot)),2) #[$/wk] cost of HVAC elec, base charge not included
    ElightCost = round(sum(Ecost_hrly)*(sum(Elight)/sum(Etot)),2) #[$/wk] cost of lighting elec, base charge not included
    EequipCost = round(sum(Ecost_hrly)*(sum(Eequip)/sum(Etot)),2) #[$/wk] cost of equipment elec, base charge not included

    TotalCost = WaterCost + TrayCost + MediaCost + SeedCost + ElecTotalCost + p.NatGasCost + p.LaborCost + FertCost #[$/wk]
    
    Profit = Revenue - TotalCost
    return round(Profit, 2), round(TotalCost, 2), round(Revenue, 2)

# %%
# Run model 
def RUN_SIM(SIM_NUM, RHi, iCO2, Ti_set, heads,PPN,iDAYL,PPFDn,PPFDc,StartDate,NursStart,CultStart,cult_water_changes,nurs_water_changes):
    TimeSpan = 7
    p = Params(TimeSpan=TimeSpan,RHi=RHi, iCO2=iCO2, Ti_set=Ti_set,heads=heads, PPN = PPN, iDAYL=iDAYL, PPFDn=PPFDn, PPFDc=PPFDc, StartDate=StartDate, NursStart=NursStart,CultStart=CultStart,cult_water_changes=cult_water_changes,nurs_water_changes=nurs_water_changes)
    To, RHo, atmCO2, hourly_timestamps = get_weather(p)
    weather = [To, RHo, atmCO2, hourly_timestamps]
    mwtranspLIST, mCplantLIST, FinalYieldLIST, NewBiomassLIST, hoursLIST, HarvestsLIST, Harvest, Qleds, Qplants, mCplant, mwtransp, biomass_prod, Eequip, Ehvac, Elight, Etot, Etotal, Ecost_hrly, df, AbsHumInt, AHo, TRANSP, Ti_avg, mdot_econ = run_model(TimeSpan,weather, RHi, iCO2, Ti_set, p)
    FullResults, GWP, ET, EUT, mwout, mCO2in = get_impacts(TimeSpan,Ehvac,Elight,Eequip,mCplant, mwtransp, AbsHumInt, AHo, iCO2, p, TRANSP, atmCO2, mdot_econ)
    Profit, Cost, Revenue = get_profit(TimeSpan, mwtransp, Ehvac, Etot, Elight, Eequip, Ecost_hrly, biomass_prod, mwout, p)
    Y = [
        int(SIM_NUM),
        float(Profit),
        float(Cost),
        float(Revenue),
        float(biomass_prod),
        float(Harvest),
        float(GWP),
        float(ET),
        float(EUT),
        float(Etotal)
    ]
    # prints to console, which is captured as an HPC output          
    print(','.join(map(str, Y)))

# Executes this program/function
if __name__ == "__main__":
    # Parse command-line arguments
    """
    ### Use below for running Global Sensitivity Analysis instead
    args = sys.argv[1:]
    if len(args) != 14:
        print(f"Expected 14 arguments, but got {len(args)}")  ######### Adjust to YOUR number of inputs
        sys.exit(1)

    # Convert arguments to appropriate types
    SIM_NUM = int(args[0])
    RHi, iCO2, Ti_set, heads, PPN, iDAYL, PPFDn, PPFDc,StartDate_idx, NursStart_idx, CultStart_idx, cult_water_changes, nurs_water_changes = map(float, args[1:])

    StartDate = date_list[int(round(StartDate_idx))]
    NursStart = int(round(NursStart_idx))
    CultStart = int(round(CultStart_idx))
    PPN = int(round(PPN))
    iDAYL = int(round(iDAYL))
    heads = int(round(heads))

    # Call RUN_SIM with the parsed arguments
    RUN_SIM(SIM_NUM, RHi, iCO2, Ti_set, heads, PPN, iDAYL, PPFDn, PPFDc,StartDate, NursStart, CultStart, cult_water_changes, nurs_water_changes)""" 

    # Parse command-line arguments
    # Order of command-line arguments: SIM_NUM, PPN, iDAYL, PPFDc, PPFDn, Ti_set, iCO2, RHi, NursStart_idx, CultStart_idx
                    # SIM_NUM = simulation number (if irrelevant, use 1)
    args = sys.argv[1:]
    if len(args) != 10:
        print(f"Expected 10 arguments, but got {len(args)}")  ######### Adjust to YOUR number of inputs
        sys.exit(1)

    # Convert arguments to appropriate types
    SIM_NUM = int(args[0])
    PPN, iDAYL, PPFDc, PPFDn, Ti_set, iCO2, RHi, NursStart_idx, CultStart_idx = map(float, args[1:])

    NursStart = int(round(NursStart_idx))
    CultStart = int(round(CultStart_idx))
    PPN = int(round(PPN))
    iDAYL = int(round(iDAYL))

    p = Params # Use defaults for past runs - when doing heat wave, can choose date here though bc doesn't matter for past
    To, RHo, atmCO2, hourly_timestamps = get_weather(p)
    weather = [To, RHo, atmCO2, hourly_timestamps]
    Sections = run_past(p.TimeSpan,weather, p.RHi, p.iCO2, p.Ti_set, p)
    p2 = Params(RHi=RHi, iCO2=iCO2, Ti_set=Ti_set,PPN = PPN, iDAYL=iDAYL, PPFDn=PPFDn, PPFDc=PPFDc, NursStart=NursStart,CultStart=CultStart)  # For future, params are this instead

    mwtranspLIST, mCplantLIST, FinalYieldLIST, NewBiomassLIST, hoursLIST, HarvestsLIST, Harvest, Qleds, Qplants, mCplant, mwtransp, biomass_prod, Eequip, Ehvac, Elight, Etot, Etotal, Ecost_hrly, df, AbsHumInt, AHo, TRANSP, Ti_avg, mdot_econ = run_future(Sections, p2.TimeSpan, weather, p2.RHi, p2.iCO2, p2.Ti_set, p2)

    FullResults, GWP, ET, EUT, mwout, mCO2in = get_impacts(p.TimeSpan,Ehvac,Elight,Eequip,mCplant, mwtransp, AbsHumInt, AHo, iCO2, p, TRANSP, atmCO2, mdot_econ)
    Profit, Cost, Revenue = get_profit(p.TimeSpan, mwtransp, Ehvac, Etot, Elight, Eequip, Ecost_hrly, biomass_prod, mwout, p)
    Y = [
        int(SIM_NUM),
        float(Profit),
        float(Cost),
        float(Revenue),
        float(biomass_prod),
        float(Harvest),
        float(GWP),
        float(ET),
        float(EUT),
        float(Etotal)
    ]
    # prints to console, which is captured as an HPC output          
    print(','.join(map(str, Y)))