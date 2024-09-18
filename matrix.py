# libraries 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# data from excel 
actual = np.array(
    [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1])
predicted = np.array(
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
# confusion matrix 
cm = confusion_matrix(actual, predicted)
# plotting the confusion matrix with heatmap
sns.heatmap(cm, annot=True,fmt='g', xticklabels=['Cancerous Cell', 'Not Cancerous Cell'],
            yticklabels=['Cancerous Cell', 'Not Cancerous Cell'])


# converting to np array 
cmnp = np.array(cm)
# how many values are TP, FP, FN, or TN?
TP = cmnp[1,1]
FP = cmnp[0,1]
FN = cmnp[1,0]
TN = cmnp[0,0]
# prints the amount in any of the 4 conditions 
print(f"True Positives: {TP}")
print(f"FalsePositives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Negatives: {TN}")
# report that includes metrics like precision, recall, f1-store, and support 
print(classification_report(actual, predicted))



# plotting confusion matrix!!
plt.ylabel('Actual', fontsize=13)
plt.title("Confusion matrix", pad=20)
plt.gca().xaxis.set_label_position('top')
plt.xlabel("Prediction")
plt.gca().xaxis.tick_top

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5,0.05, "Prediction", ha='center', fontsize=13)
plt.show()
