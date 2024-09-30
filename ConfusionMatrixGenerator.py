# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from random import randint, choice
from sklearn.metrics import classification_report, confusion_matrix

# Welcome to my program, which generates a classification matrix and confusion report from random values between 0 and 1

actual_values = []
predicted_values = []
# what would your confusion matrix be about??
inputFromUser = input("Input your element (ie cancerous cell, dogs, cats, whatever): ")
print(f"{inputFromUser} is your element. Your other element is not {inputFromUser}")

# function for confusion matrix 
def rand_cm(user_elements):
    for x in range(16):
        actual_values.append(randint(0,1))
        predicted_values.append(randint(0,1))
   
        

    print("Array of actual values:", actual_values)
    print("Array of predicted values:", predicted_values)

    randcm = confusion_matrix(actual_values, predicted_values)
    sns.heatmap(randcm, annot=True, xticklabels=[f"not {user_elements}", f"{user_elements}"], yticklabels=[f"not {user_elements}", f"{user_elements}"])

    npCm = np.array(randcm)

    # 1 is one, while 0 is not one 
    # Actual, then predicted 

    TP = npCm[1,1] # True and True
    FP = npCm[0,1] # False and True
    TN = npCm[0,0] # False and False
    FN = npCm[1,0] # True and False

    print(f"True Positive: {TP}")
    print(f"False Positive: {FP}")
    print(f"False Negative: {FN}")
    print(f"True Negative: {TN}")

    print(classification_report(actual_values, predicted_values))

    plt.ylabel("Actual",fontsize=12)
    plt.title(f"Random Confusion Matrix of: {user_elements} and not {user_elements}", pad=20)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel("Predicted", fontsize=12)
    plt.gca().xaxis.tick_top

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5,0.05, "Prediction", ha='center', fontsize=13)
    plt.show()

rand_cm(inputFromUser)
