import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix

data_tswift = 'taylor_swift_spotify_data.csv'

df_swift = pd.read_csv(data_tswift, header=0)

df_swift = pd.get_dummies(df_swift, columns=['Album', 'Song Name'], drop_first=True)

df_swift = df_swift.drop(columns= ['Playlist ID', 'URI'], errors='ignore')

df_swift = df_swift.apply(pd.to_numeric, errors="coerce")

df_swift.to_csv('taylor_swift_spotify_data_cleaned.csv', index=False)

df_swift['DanceabilityClass'] = df_swift['Danceability'].apply(lambda x: 1 if x >= .400 else 0)

df_swift['LoudnessEnergy'] = df_swift['Loudness'].apply(lambda l: 1 if l >= -4.00 else 0)

df_swift['HitClassification'] = df_swift['Energy'].apply(lambda y: 1 if y >= .400 else 0)

X = df_swift[['DanceabilityClass', 'LoudnessEnergy']]

y = df_swift['HitClassification']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

accurate = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accurate:.2f}")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

'''
df_swift.info()
RangeIndex: 147 entries, 0 to 146
Columns: 168 entries, Danceability to Song Name_‘tis the damn season
dtypes: bool(155), float64(13)

features_important = rf.feature_importances_
feature_df_swift = pd.DataFrame({"Feature": albumAndSong, 'Importance': features_important})
feature_df_swift = feature_df_swift.sort_values(by='Importance', ascending=False)

print(feature_df_swift)

147, 168 df_swift.shape




'''