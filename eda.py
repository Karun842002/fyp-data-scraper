import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

df = pd.read_csv('./datasets/all_merged.csv')
df.head()

df['truth_value'].value_counts(), df['source'].value_counts()

df.dropna(subset=['claim'], inplace=True)

df.info()

df['truth_value'].value_counts().plot.pie()

df['source'].value_counts().plot.pie()

df['source'].hist()

mostcommon = nltk.FreqDist(df['claim']).most_common(100)
wordcloud = WordCloud(width=800, height=400,
                      background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(20, 7.5), facecolor='white')
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Top 100 Most Common Words (all)', fontsize=50)
plt.show()

mostcommon = nltk.FreqDist(df[df['truth_value'] == 'meter-true']['claim']).most_common(100)
wordcloud = WordCloud(width=800, height=400,
                      background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(20, 7.5), facecolor='white')
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Top 100 Most Common Words (true)', fontsize=45)
plt.show()

mostcommon = nltk.FreqDist(
    df[df['truth_value'] == 'meter-false']['claim']).most_common(100)
wordcloud = WordCloud(width=800, height=400,
                      background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(20, 7.5), facecolor='white')
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Top 100 Most Common Words (fake)', fontsize=45)
plt.show()

nltk.download('stopwords')

from nltk.corpus import stopwords

T = df['claim'].str.split(' \n\n---\n\n').str[0]
T = T.str.replace(
    '-', ' ').str.replace('[^\w\s]', '').str.replace('\n', ' ').str.lower()
stop = stopwords.words('english')
T = T.apply(lambda x: ' '.join([y for y in x.split() if not y.isdigit()]))
T = T.apply(lambda words: ' '.join(word.lower()
            for word in words.split() if word not in stop))

from sklearn.feature_extraction.text import CountVectorizer

corpus = T.tolist()
labels = df['truth_value'].tolist()
vectorizer = CountVectorizer()
word_counts = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()
df_word_freq = pd.DataFrame(index=feature_names, columns=['meter-false', 'meter-true'], data=0)
for i, sentence in enumerate(corpus):
    label = labels[i]
    for word in sentence.split():
        if word in feature_names:
            df_word_freq.loc[word, label] += word_counts[i,
                                                         feature_names.index(word)]

fig, axs = plt.subplots(1, 2, figsize=(50, 50))
for i, label in enumerate(['meter-false', 'meter-true']):
    df_word_freq.sort_values(by=label, ascending=False, inplace=True)
    df_word_freq.head(100).plot(kind='bar', y=label, ax=axs[i], title=label)
plt.tight_layout()
plt.show()

results = set()
T.str.lower().str.split().apply(results.update)
len(results)

