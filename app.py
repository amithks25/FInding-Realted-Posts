
import sklearn.datasets
import scipy as sp

new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks
"""

print("""\
We Have Taken 20NewsGroups dataSet for clustering.
""")

all_data = sklearn.datasets.fetch_20newsgroups(subset="all")
print("Number of total posts: %i" % len(all_data.filenames))

groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset="train",
                                                 categories=groups)
print("Number of training posts in tech groups:", len(train_data.filenames))

labels = train_data.target
num_clusters = 50  # sp.unique(labels).shape[0]

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,stop_words='english',
                 decode_error='ignore')

vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))
# samples: 3529, #features: 4712

from sklearn.cluster import KMeans
ls =[]

km = KMeans(n_clusters=4, n_init=1,init = 'k-means++', verbose=1, random_state=0)
km.fit(vectorized)
ls.append(km.inertia_)
print(ls)

# from sklearn.cluster import KMeans
# ls =[]
# for i in range(1,10):
#     km = KMeans(n_clusters=i, n_init=1,init = 'k-means++', verbose=1, random_state=0)
#     km.fit(vectorized)
#     ls.append(km.inertia_)
# print(ls)

# k = []
# for i in range(1,10):
#     k.append(i)
# print(k)

# import matplotlib.pyplot as plt
# #graph which is used to find the number of clusters
# plt.plot(k, ls,'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()

print("km.labels_=%s" % km.labels_)
# km.labels_=[ 6 34 22 ...,  2 21 26]
print("hi")
print("len of KM_labels",len(km.labels_))
print("km.labels_.shape=%s" % km.labels_.shape)

from sklearn import metrics
print(len(labels))
print("labels.....",labels)
#print("kmlabels_",km.labels_)
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))

new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
print("hey",new_post_label)
# 
similar_indices = (km.labels_ == new_post_label).nonzero()[0]
#print("similar indices",similar_indices)
# print("Completeness After: %0.3f" % metrics.completeness_score(labels, similar_indices))

similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, train_data.data[i]))

similar = sorted(similar)
print("Count similar: %i" % len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
# print(show_at_1)
print()
result = " ".join(str(x) for x in show_at_1)
a = result.replace("/n"," ")
b = a.replace("<"," ")
c = b.replace(">"," ")
print(c)

# print("=== #2 ===")
# print(show_at_2)
# print()

# print("=== #3 ===")
# print()
# result = " ".join(str(x) for x in show_at_3)
# a = result.replace("/n"," ")
# print(a)



print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# Homogeneity: 0.400
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# Completeness: 0.206
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# V-measure: 0.272
print("Adjusted Rand Index: %0.3f" %metrics.adjusted_rand_score(labels, km.labels_))


# import matplotlib.pyplot as plt
# import numpy as np
# x_values = ["Homogeneity","Completeness","V-measure","Adjusted Rand Index"]
# x_count = np.arange(len(x_values))
# value = [0.247,0.401,0.305,0.146]

# plt.bar(x_count, value, align = 'center')
# plt.xticks(x_count, x_values)
# plt.yticks(0,1)
# plt.ylabel('range')
# plt.title('Evaluation K_Measn Model')

# plt.show()