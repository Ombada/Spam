# Method 1: This is how to use plain python to get your data ready 
#  for training. however its much slower than the other methods 
import csv
import numpy as np
import pandas as pd
file_name = "spambase.data"
with open(file_name, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    data = list(reader)
data = np.array(data, dtype=np.float32)
print(data.shape)

n_samples, n_features = data.shape
n_features -= 1
x = data[:, :n_features]
y = data[:, n_features]

# Method 2: This is how to use numpy only 
# to get your data ready for training.
data2 = np.loadtxt(file_name, delimiter=',')
print(data2.shape)

#OR even better, since it easier to deal with missing data, you can use genfromtxt instead of loadtxt

data3 = np.genfromtxt(file_name, delimiter=',')
print(data3.shape)

#Method 3: This is how to use pandas to get your data ready for training.

df = pd.read_csv(file_name, header=None, delimiter=',')
print(df.shape)

data = df.to_numpy()
print(data.shape)