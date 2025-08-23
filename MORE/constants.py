from pathlib import Path
"""Output directory constants
"""    
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data/raw_results"
ENCODINGS = ["dense", "standard", "modified", "fractional"]

"""
Global model names 
"""
regr_name_singleTarget_RR = "ST-RFR-RR"
regr_name_singleTarget_PI = "ST-RFR-PI"
regr_name_singleTarget_Epsi = "ST-RFR-EPSI"

regr_name_gb_singleTarget_RR = "ST-GBR-RR"
regr_name_gb_singleTarget_PI = "ST-GBR-PI"
regr_name_gb_singleTarget_Epsi = "ST-GBR-EPSI"

regr_name_interval_rf = "Chain-RFR-PI"
regr_name_rounding_rf = "Chain-RFR-RR"
regr_name_chain_Epsi = "Chain-RFR-RR-EPSI"

regr_name_interval_gb = "Chain-GBR-PI"
regr_name_rounding_gb = "Chain-GBR-RR"
regr_name_chain_gb_Epsi = "Chain-GBR-RR-EPSI"

regr_name_mort = "Native-RF"
regr_name_mort_interval = "Native-RF-PI"
regr_name_mort_Epsi = "Native-RF-EPSI"

regr_name_gbr = "Native-GBR"
regr_name_gbr_interval = "Native-GBR-PI"
regr_name_gbr_Epsi = "Native-GBR-EPSI"

clas_name_randomForest = "RPC-RF"
clas_name_gb = "RPC-GB"

""" 
Below ar the OpenML database ids
"""
name_to_data_lr = {
        'authorship': 42834,
        "glass": 42847,
        'iris': 42851,
        "letter": 45727,
        "libras": 45736,
        "pendigits": 42856,
        "segment": 42859,
        "vehicle": 42863,
        "vowel": 42865,
        "wine": 42867,
        "yeast": 45737,
        # Real Szenario Dataset
        "movies": 45735,
    }


name_to_data_plr = {
        "authorship":42835,
        "blocks":42836,
        "breast":42838,
        'ecoli': 42844,
        "glass":42848,
        "iris":42871,
        "letter":42853,
        'libras': 42855,
        "pendigits":42857,
        "satimage":42858,
        "segment":42860,
        "vehicle":42864,
        "vowel":42866,
        'wine': 42872,
        "yeast":42870,
        # REAL DATA SETS
        "algae":45755,
        "movies":45738
}
name_to_data = dict([*name_to_data_plr.items(), *name_to_data_lr.items()])


"""
Sensible Epsilon Values
"""
epsilon_values = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.2]
percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
epsi_missing_pairs = [ (eps, perc) for eps in epsilon_values for perc in percentages]
