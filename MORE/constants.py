from pathlib import Path
"""Output directory constants
"""    
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data/raw_results"


"""
Global model names 
"""
regr_name_singleTarget_RR = "ST-RF-RR"
regr_name_singleTarget_PI = "ST-RF-PI"
regr_name_singleTarget_Epsi = "ST-RF-EPSI"

regr_name_interval_rf = "Chain-RFR-PI"
regr_name_rounding_rf = "Chain-RFR-RR"
regr_name_chain_Epsi = "Chain-RFR-RR-EPSI"

regr_name_mort = "Native-RF"
regr_name_mort_interval = "Native-RF-PI"
regr_name_mort_Epsi = "Native-RF-EPSI"

clas_name_randomForest = "RPC-RF"


""" 
Below ar the OpenML database ids
"""
name_to_data_lr = {
        'LR-AUTHORSHIP': 42834,
        "LR-GLASS": 42847,
        'LR-IRIS': 42851,
        "LR-LETTER": 45727,
        "LR-LIBRAS": 45736,
        "LR-PENDIGITS": 42856,
        "LR-SEGMENT": 42859,
        "LR-VEHICLE": 42863,
        "LR-VOWEL": 42865,
        "LR-WINE": 42867,
        "LR-YEAST": 45737,
        # Real Szenario Dataset
        "LR-REAL-MOVIES": 45735,

    }


name_to_data_plr = {
        "PLR-AUTHORSHIP":42835,
        "PLR-BLOCKS":42836,
        "PLR-BREAST":42838,
        'PLR-ECOLI': 42844,
        "PLR-GLASS":42848,
        "PLR-IRIS":42871,
        "PLR-LETTER":42853,
        'PLR-LIBRAS': 42855,
        "PLR-PENDIGITS":42857,
        "PLR-SATIMAGE":42858,
        "PLR-SEGMENT":42860,
        "PLR-VEHICLE":42864,
        "PLR-VOWEL":42866,
        'PLR-WINE': 42872,
        "PLR-YEAST":42870,
        # REAL DATA SETS
        "PLR-REAL-ALGAE":45755,
        "PLR-REAL-MOVIES":45738
}
name_to_data = dict([*name_to_data_plr.items(), *name_to_data_lr.items()])


"""
Sensible Epsilon Values
"""
epsilon_values = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.2]
percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
epsi_missing_pairs = [ (0.01, 0.2),  (0.01, 0.4), (0.01, 0.6),
                      (0.03, 0.1),(0.03, 0.3),  (0.03, 0.5),
                       (0.05, 0.2), (0.05, 0.4), (0.05, 0.6),
                      (0.07, 0.1),  (0.07, 0.3),  (0.07, 0.5), 
                     (0.09, 0.2), (0.09, 0.4), (0.09, 0.6),
                      (0.12, 0.1), (0.12, 0.3),  (0.12, 0.5), 
                      (0.14, 0.2),  (0.14, 0.4),  (0.14, 0.6),
                      (0.16, 0.1),  (0.16, 0.3),  (0.16, 0.5),
                      (0.18, 0.2), (0.18, 0.4),  (0.18, 0.6),
                      (0.2, 0.1),  (0.2, 0.3), (0.2, 0.5)]