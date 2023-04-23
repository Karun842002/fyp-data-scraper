import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
    
def cluster_text(text):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text)

    Sum_of_squared_distances = []
    K = range(2,200)
    for k in K:
       km = KMeans(n_clusters=k, max_iter=200, n_init=10)
       km = km.fit(X)
       Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    print('How many clusters do you want to use?')
    true_k = int(input())
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)
    model.fit(X)

    labels=model.labels_
    clusters=pd.DataFrame(list(zip(text,labels)),columns=['title','cluster'])
    #print(clusters.sort_values(by=['cluster']))

    for i in range(true_k):
        clusters[clusters['cluster'] == i].to_csv(f'./clusters/cluster-{i}.csv')
        
    return

strlst = []
df = pd.read_csv('./datasets/latest3.csv')
for i, row in df.iterrows():
    strlst.append(str(row['claim']))
cluster_text(strlst)