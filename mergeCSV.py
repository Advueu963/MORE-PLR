# importing pandas
import pandas as pd

# merging two csv files
dataFilesSVM =[
    "../data/MultiRegression/LR-SVM/LR_CHAIN-SVR-NoLetter.csv",
    "../data/MultiRegression/LR-SVM/LR_SingleTarget-SVR-NoLetter.csv",
    "../data/MultiRegression/LR-SVM/LR_SVC-JC-NoLetter.csv",
    *[f"../data/MultiRegression/LR-SVM/LR-Letter/LR_{x}.csv" for x in range(50)]
]
targetFileSVM = "../data/MultiRegression/LR-SVM/SVMs.csv"

dataFilesDT = [
"../data/MultiRegression/LR-DT/LR_Chain-DT-Rounding.csv",
"../data/MultiRegression/LR-DT/LR_JC-DT.csv",
"../data/MultiRegression/LR-DT/LR_SingleTarget-DT.csv"
]
targetFileDT = "../data/MultiRegression/LR-DT/DTs.csv"

dataFilesRF = [
    "../data/MultiRegression/LR-RF/LR_Chain-RF-Interval.csv",
    "../data/MultiRegression/LR-RF/LR_Chain-RF-Rounding.csv",
    "../data/MultiRegression/LR-RF/LR_Native-RF.csv",
    "../data/MultiRegression/LR-RF/LR_JC-RF.csv",
    "../data/MultiRegression/LR-RF/LR_SingleTarget-RF.csv",
]
targetFileRF = "../data/MultiRegression/LR-RF/RFs.csv"

dataFilesLR = [
    targetFileDT,
    targetFileRF,
    targetFileSVM,
    "../data/MultiRegression/LR_politicalEvaluation.csv"
]
targetFileLR = "../data/MultiRegression/LRs.csv"

dataFiles_PLR_DT = [
"../data/MultiRegression/PLR-DT/PLR_Chain-DT-Rounding.csv",
"../data/MultiRegression/PLR-DT/PLR_JC-DT.csv",
"../data/MultiRegression/PLR-DT/PLR_SingleTarget-DT.csv"
]
targetFile_PLR_DT = "../data/MultiRegression/PLR-DT/DTs.csv"

dataFiles_PLR_RF = [
    "../data/MultiRegression/PLR-randomForest/PLR-Chain-RF-Interval.csv",
    "../data/MultiRegression/PLR-randomForest/PLR-Chain-RF-Rounding.csv",
    "../data/MultiRegression/PLR-randomForest/PLR-Native-RF.csv",
    "../data/MultiRegression/PLR-randomForest/PLR-JC-RF.csv",
    "../data/MultiRegression/PLR-randomForest/PLR-SingleTarget-RF.csv"

]
targetFile_PLR_RF = "../data/MultiRegression/PLR-randomForest/RFs.csv"

dataFiles_PLR_SVM = [
    "../data/MultiRegression/PLR-SVM/PLR_CHAIN-SVR.csv",
    "../data/MultiRegression/PLR-SVM/PLR_SingleTarget-SVR.csv",
    "../data/MultiRegression/PLR-SVM/PLR_SVC-JC.csv",
    *[f"../data/MultiRegression/PLR-SVM/PLR-Letter/PLR_{x}.csv" for x in range(50)]

]
targetFile_PLR_SVM = "../data/MultiRegression/PLR-SVM/SVMs.csv"

dataFilesPLR = [
    targetFile_PLR_DT,
    targetFile_PLR_RF,
    targetFile_PLR_SVM,
    "../data/MultiRegression/PLR_politicalEvaluation.csv"
]
targetFilePLR = "../data/MultiRegression/PLRs.csv"


dataFilesLRMissing = [
    #*[f"../data/MultiRegression/missingLabels/LR-Chain-DT-Rounding-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/missingLabels/LR-Chain-RF-Interval-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/missingLabels/LR-Chain-RF-Rounding-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-JC-DT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-JC-RF-{x}.csv" for x in [0.1, 0.2,0.3, 0.4, 0.5 ]],
    #*[f"../data/MultiRegression/missingLabels/LR-MORT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-SingleTarget-RF-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-SingleTarget-SVM-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-SingleTarget-DT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/LR-SVM/LR-Letter-Missing/LR_CHAIN-SVR_{x}_{y}.csv" for x in range(50) for y in [.1, .2, .3, .4, .5, .6]],
    *[f"../data/MultiRegression/LR-SVM/LR-Letter-Missing/LR_SVC-JC_{x}_{y}.csv" for x in range(50) for y in
      [.1, .2, .3]],
    *[f"../data/MultiRegression/missingLabels/LR-JC-SVM-{x}-NoLetter.csv" for x in [.1, .2, .3]]

]
targetFileLRMissing = "../data/MultiRegression/missingLabels/Missing-LRs.csv"


dataFilesLRMissing_Table = [
    *[f"../data/MultiRegression/missingLabels/LR-Chain-DT-Rounding-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-Chain-RF-Interval-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-Chain-RF-Rounding-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-JC-DT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-JC-RF-{x}.csv" for x in [0.1, 0.2,0.3, 0.4, 0.5]],
    *[f"../data/MultiRegression/missingLabels/LR-MORT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-SingleTarget-RF-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-SingleTarget-SVM-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/LR-SingleTarget-DT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/LR-SVM/LR-Letter-Missing/LR_CHAIN-SVR_{x}_{y}.csv" for x in range(50) for y in [.1, .2, .3, .4, .5, .6]],
    *[f"../data/MultiRegression/LR-SVM/LR-Letter-Missing/LR_SVC-JC_{x}_{y}.csv" for x in range(50) for y in
      [.1, .2, .3]],
    *[f"../data/MultiRegression/missingLabels/LR-JC-SVM-{x}-NoLetter.csv" for x in [.1, .2, .3]]

]
targetFileLRMissing_Table = "../data/MultiRegression/missingLabels/Missing-LRs-Table.csv"


dataFilesPLRMissing = [
    #*[f"../data/MultiRegression/missingLabels/PLR-Chain-DT-Rounding-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/missingLabels/PLR-Chain-RF-Interval-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/missingLabels/PLR-Chain-RF-Rounding-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/PLR-JC-DT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/PLR-JC-RF-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/missingLabels/PLR-MORT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/PLR-SingleTarget-RF-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/PLR-SingleTarget-SVM-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    *[f"../data/MultiRegression/missingLabels/PLR-SingleTarget-DT-{x}.csv" for x in [0.1, 0.2, 0.3, 0.4, 0.5, .6]],
    #*[f"../data/MultiRegression/PLR-SVM/PLR-Letter-Missing/PLR_CHAIN-SVR_{x}_{y}.csv" for x in range(50) for y in [.1, .2, .3, .4, .5, .6]],
    *[f"../data/MultiRegression/PLR-SVM/PLR-Letter-Missing/PLR_SVC-JC_{x}_{y}.csv" for x in range(50) for y in
      [.1, .2, .3, .4, .5, .6]],
    *[f"../data/MultiRegression/missingLabels/PLR-JC-SVM-{x}-NoLetter.csv" for x in [.1, .2, .3, .4, .5, .6]]

]
targetFilePLRMissing = "../data/MultiRegression/missingLabels/Missing-PLRs.csv"

dataFileHole = [
    targetFileLR,
    targetFilePLR,
]
targetFileHole = "../data/MultiRegression/LR_and_PLR.csv"

# Modify the following to merge the files
dataFiles = dataFilesLRMissing_Table
targetFile = targetFileLRMissing_Table

df = pd.concat(
    map(pd.read_csv, dataFiles), ignore_index=True)
df = df.drop("Unnamed: 0", axis=1)
df.to_csv(targetFile)
print(df)
