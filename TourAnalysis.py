import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

tour_data = 'Taylor_Train.csv'

taylorTours = pd.read_csv(tour_data, encoding='latin1')

taylorTours['Revenue'] = taylorTours['Revenue'].str.replace(',', '')
taylorTours['Revenue'] = taylorTours['Revenue'].str.replace('$', '')
taylorTours['Revenue'] = pd.to_numeric(taylorTours['Revenue'], errors='coerce')
taylorTours[['Tickets Sold', 'Tickets Available']] = taylorTours['Attendance (tickets sold / available)'].str.replace(',', '').str.split(' / ', expand=True)
# taylorTours['Attendance (tickets sold / available)'] = taylorTours['Attendance (tickets sold / available)'].str.split(' / ', expand=True)
taylorTours['Tickets Sold'] = pd.to_numeric(taylorTours['Tickets Sold'], errors='coerce')

taylorTours['Tickets Available'] = pd.to_numeric(taylorTours['Tickets Available'], errors='coerce')

# taylorTours['Tickets Available'].astype(int)
#taylorTours_numerical = pd.get_dummies(taylorTours, columns=['City', 'Country', 'Venue', 'Tour'])
taylorTours = taylorTours.drop(columns='Opening act(s)', errors='ignore')

# target
numerical_features = ['Tickets Sold', 'Tickets Available']

taylorTours['Sold Out Concert'] = pd.to_numeric((taylorTours['Tickets Sold'] == taylorTours['Tickets Available']), errors='coerce')

# (taylorTours['Tickets Sold'] == taylorTours['Tickets Available']).astype(int)

taylorTours = pd.get_dummies(taylorTours, columns=['City', 'Country', 'Venue', 'Tour'])


taylorDataNew = taylorTours.to_csv('Taylor_Train_Modified.csv', index=False)


# print(taylorTours.sort_values(["Revenue"], ascending=False))
#print(taylorTours[taylorTours['Revenue'] > 1000000])
# print(taylorTours[taylorTours['Tour_The_1989_World_Tour'] == 1])
filtered_data = taylorTours[(taylorTours['Revenue'] > 1000000) & (taylorTours['City_Chicago'] == 1)]

# Print the filtered DataFrame
print(filtered_data)




'''

X = pd.concat([taylorTours[numerical_features], LocationClass, RevenueClass], axis=1)

y = taylorTours['Sold Out Concert']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

accurate = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accurate:.2f}")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)




'''
