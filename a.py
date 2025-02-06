import pandas as pd

#load the csv file and use the first two columns only
df = pd.read_csv('data.csv', usecols=[0,1])

# Read all columns except first two
other_cols = pd.read_csv('data.csv').iloc[:, 2:]
# Calculate row-wise sum and add to original dataframe
df['sum'] = other_cols.sum(axis=1)