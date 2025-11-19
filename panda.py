import pandas as pd

df = pd.read_csv("simulation_accuracy_log.csv")
print(df.groupby("Run ID")["Overall Accuracy"].mean())
