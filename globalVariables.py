"""Global model names 
"""
regr_name_singleTarget_rf = "ST-RF"
regr_name_interval_rf = "Chain-RFR-PI"
regr_name_rounding_rf = "Chain-RFR-RR"
regr_name_mort = "Native-RF"
clas_name_randomForest = "RPC-RF"


""" Below ar the OpenML database ids
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