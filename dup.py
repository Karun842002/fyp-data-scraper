import pandas as pd
df = pd.read_csv('./datasets/allmergednew.csv', header=0)
print(len(df))
df = df.drop_duplicates(subset='claim')
df.drop(["no"], axis=1, inplace=True)
print(len(df))
df.to_csv('./datasets/cleaned1.csv', index=False)