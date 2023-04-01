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

import pandas as pd

df1 = pd.read_csv('./datasets/latest.csv', header=0)
df2 = pd.read_csv('./datasets/latest2.csv', header=0)

print(len(df1), len(df2))

df3 = pd.concat([df1[['claim', 'simple_sentence', 'truth_value']], df2[['claim', 'simple_sentence', 'truth_value']]])
print(len(df3))
df3.to_csv('./datasets/dataset.csv', index=False)