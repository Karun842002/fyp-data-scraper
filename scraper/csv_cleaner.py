# import pandas as pd
# from simplifier_new import serverParser 


# from threading import Thread
# import functools

# def timeout(timeout):
#     def deco(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
#             def newFunc():
#                 try:
#                     res[0] = func(*args, **kwargs)
#                 except Exception as e:
#                     res[0] = e
#             t = Thread(target=newFunc)
#             t.daemon = True
#             try:
#                 t.start()
#                 t.join(timeout)
#             except Exception as je:
#                 print ('error starting thread')
#                 raise je
#             ret = res[0]
#             if isinstance(ret, BaseException):
#                 raise ret
#             return ret
#         return wrapper
#     return deco

# def timeoutServerParser(sentence):
#     res = sentence
#     try:
#         res = timeout(timeout=15)(serverParser)(sentence)
#     finally:
#         return res

# df = pd.read_csv('../scraper/datasets/disinfo_polyg_sf_merged.csv', header=0)
# df['claim'] = df['0']
# df['truth_value'] = df['1']
# df.drop(['0','1'], inplace=True, axis=1)
# df.reset_index(drop=True, inplace=True)
# df['simple_sentence'] = df['claim'].apply(lambda x:timeoutServerParser(x))
# df.to_csv('../scraper/datasets/latest2.csv')

# import pandas as pd

# df1 = pd.read_csv('./datasets/latest.csv', header=0)
# df2 = pd.read_csv('./datasets/latest2.csv', header=0)

# print(len(df1), len(df2))

# df3 = pd.concat([df1[['claim', 'simple_sentence', 'truth_value']], df2[['claim', 'simple_sentence', 'truth_value']]])
# print(len(df3))
# df3.to_csv('./datasets/dataset.csv', index=False)

import pandas as pd
df1 = pd.read_csv('./datasets/cleaned4.csv')
df2 = pd.read_csv('./datasets/stopfakev2.csv')
df3 = pd.read_csv('./datasets/polygraphv2.csv')
df4 = pd.read_csv('./datasets/disinfo.csv')
df5 = pd.read_csv('./datasets/latest2.csv')

df1['source'] = df1['source'].apply(lambda x: x if x in ['nyt','nrpublic','washington-post','politifact','vox-ukraine'] else 'manual')
df2['source'] = df2['0'].apply(lambda x: 'stopfake')
df3['source'] = df3['0'].apply(lambda x: 'polygraph')
df4['source'] = df4['0'].apply(lambda x: 'EUvsDisinfo')

df2['claim'] = df2['0']
df3['claim'] = df3['0']
df4['claim'] = df4['0']

df2['truth_value'] = df2['1']
df3['truth_value'] = df3['1']
df4['truth_value'] = df4['1']

df5 = df5[['claim','truth_value','simple_sentence']]
df6 = pd.concat([df2[['claim','truth_value','source']],df3[['claim','truth_value','source']],df4[['claim','truth_value','source']]])
df7 = df6.set_index('claim').join(df5.set_index('claim'), rsuffix='r')
print(df7['source'].value_counts())
df7['claim'] = df7.index

# print(df7[['claim','simple_sentence','truth_value','source']].head())
# print(df1[['claim','simple_sentence','truth_value','source']].head())

df8 = pd.concat([df1[['claim','simple_sentence','truth_value','source']],df7[['claim','simple_sentence','truth_value','source']]])
df8.to_csv('./datasets/latest3.csv', index=False)

# import pandas as pd
# df = pd.read_csv('./datasets/latest3.csv')
# print(df[['source', 'truth_value']].value_counts())
# print(len(df))
