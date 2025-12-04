<h1>Digital Twin Sub-Model Development</h1>
The contents of this GitHub repository were developed by Lauren Lindow, PhD as part of her final dissertation at the University of Florida: "A Systematic Framework Towards Digital Twin Development for Hydroponic Shipping Container Farms."   
<br></br>
All code was developed with Python 3.9. The requirements.txt file contains the required packages for all sub-models contained within this repository.  

<h2>Parameterized Impacts Assessment Model</h2>
<b>Main model file:</b> LCA_model.py 

<b>Auxiliary files:</b> config.py, params.py, SimaPro2.csv, ValdStudyData.csv 

This sub-model is intended to predict environmental and economic impacts of a container farm during a user-defined study period. This code includes a lettuce crop growth model developed by [Talbot & Monfet (2024)](https://github.com/ltsb-etsmtl/crop-model) integrated with mass and energy flux equations, cost analysis calculations, and an impacts evaluation conducted with simplified life cycle assessment methodologies. The default parameters and assumed geometry is based on the HSCF located at the University of Florida/IFAS Plant Science Research and Education Unit in Citra, FL, which is a Freight Farm Greenery S model (Freight Farm, Boston, MA).  

* <b>LCA_model.py</b> contains the main python code for this sub-model. 
* <b>config.py</b> selects folder path names. To run the code on a local machine or supercomputer (such as UF's Hipergator), <i>these paths must be changed prior to use.</i>  
* <b>params.py</b> contains default values for input and study parameters that the main code will use.
* <b>SimaPro2.csv</b> contains the impact factors from SimaPro 10.2 (PRé Sustainability, 2019) and the EcoInvent v3.11 database using the TRACI 2.2 method.
* <b>ValdStudyData.csv</b> contains placeholder data with hourly temperature readings within the Citra farm, collected between December 17 - 19, 2024. In the main model file, this data is used to supply hypothetical internal temperature data in leiu of real sensor readings throughout the full study period. In implementation and during regular operation, this should be replaced with data true to the selected study dates.

<h2>Multi-Criteria Decisions Assessment Model</h2>
<b>Main model file:</b> MCDA_code.py   

<b>Auxiliary files:</b> params.py, LCA_model.py, contents of /MCDA_Samples_Out folder   

This file takes parameters and results from LCA_model.py to select "best-case" parameters dependent on user-selected weights. The folder /MCDA_Samples_Out contains example simulation parameters and results that can be fed into the MCDA_code.py file.
