import pandas as pd
df = pd.read_csv('./datasets/cleaned1.csv')
df = df.sample(frac = 1)
df.to_csv('./datasets/clean-shuffle.csv')