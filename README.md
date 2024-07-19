This repository is the official implementation of HSPGNN.
To test the model, Please Run the HSPGNN.py
if you want to download the Electricity dataset, you can go to the citation of dataset website to download dataset.
1 you can change the hyper parameters by modify the parser in HSPGNN.py
2 If you want to change the missing rate or missing mode, you can open lib/data_preparation.py and change the p_fault and p_noise of the following function:
	Electricity_mask_generator() for Electricity dataset.
	pems_bay_mask_generator() for pems-bay dataset.







