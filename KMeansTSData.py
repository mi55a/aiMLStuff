# This is a program that incorporates K-Means algorithm 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

swift_data = pd.read_csv('taylor_swift_spotify_data.csv')

swift_data.head()
# swift_data.head()
# swift_data.describe()
# swift_data.isnull().sum()

swift_data.drop(['Album', 'Song Name', 'Playlist ID', 'URI'], axis=1, inplace=True)

swift_data['DanceabilityClass'] = swift_data['Danceability'].apply(lambda x: 1 if x >= .400 else 0)

swift_data['LoudnessEnergy'] = swift_data['Loudness'].apply(lambda l: 1 if l >= -4.00 else 0)

swift_data['TempoLabel'] = swift_data['Tempo'].apply(lambda t: 1 if t >= 130.00 else 0)

swift_data['HitClassification'] = swift_data['Energy'].apply(lambda y: 1 if y >= .400 else 0)

X = swift_data[['DanceabilityClass', 'LoudnessEnergy', 'TempoLabel']]

y = swift_data['HitClassification']

scaler = StandardScaler()
X = scaler.fit_transform(X)

k = KMeans(n_clusters=2, init='k-means++')
k.fit(X)
print(k.inertia_)

SSE = []

for cluster in range(2,10):
    k = KMeans(n_clusters=cluster, init='k-means++')
    k.fit(X)
    SSE.append(k.inertia_)



'''
This is where the program starts tweaking

le = LabelEncoder()
X['DanceabilityClass', 'LoudnessEnergy', 'TempoLabel'] = le.fit_transform(X['DanceabilityClass', 'LoudnessEnergy', 'TempoLabel'])
y = le.transform(y)

cols = X.columns
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=[cols])
print(X.head())
'''

