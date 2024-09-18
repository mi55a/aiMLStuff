# libraries 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# data from excel spreadsheet, divided by model and the actual values

actual_values = [1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400]

model1_values = [1,4,9,26,45,46,69,74,101,110,141,154,189,206,245,266,309,334,381,410]
model2_values = [1, 4, 9, 21, 35, 41, 59, 69, 91, 105, 131, 149, 179, 201, 235, 261, 299, 329, 371, 405]

model3_values = [1, 4, 9, 16, 25, 36, 51, 69, 89, 102, 126, 152, 171, 201, 233, 258, 294, 332, 363, 405]

# mean absolute error calculated and printed 

mae1 = mean_absolute_error(actual_values, model1_values)
mae2 = mean_absolute_error(actual_values, model2_values)
mae3 = mean_absolute_error(actual_values, model3_values)
print("Mean Absolute Error: ", mae1)
print("Mean Absolute Error: ", mae2)
print("Mean Absolute Error: ", mae3)

# array of the mae values

mae_values = [mae1, mae2, mae3]

# converting array to numpy array 

actual_valuesnp = np.array(actual_values)
model1_valuesnp = np.array(model1_values)
model2_valuesnp = np.array(model2_values)
model3_valuesnp = np.array(model3_values)

# root mean squared error calculated 

rmse1 = np.sqrt(mean_squared_error(actual_valuesnp, model1_valuesnp))
rmse2 = np.sqrt(mean_squared_error(actual_valuesnp, model2_valuesnp))
rmse3 = np.sqrt(mean_squared_error(actual_valuesnp, model3_valuesnp))

# values from by-hand calculation by looping 

sum1 = 0
sum2 = 0
sum3 = 0
for (y, x) in zip(actual_values, model1_values):
    sq_diff = ((y - x) ** 2)
    sum1 += sq_diff
for (y, x) in zip(actual_values, model2_values):
    sq_diff2 = ((y - x) **2)
    sum2 += sq_diff2
for (y, x) in zip(actual_values, model3_values):
    sq_diff3 = ((y - x) **2)
    sum3 += sq_diff3

# array of rmse values 

rmseValues = [rmse1, rmse2, rmse3]

# labels for bar graph

models_labels = ['Model 1', 'Model 2', 'Model 3']

# printing 

print(sum1)
print(sum2)
print(sum3)
print("Root Mean Squared Error(RMSE): ", rmse1)
print("Root Mean Squared Error(RMSE): ", rmse2)
print("Root Mean Squared Error(RMSE): ", rmse3)

# Grouped bar graph

fig, ax = plt.subplots(figsize =(12,6))
ml_data = {
    'Root Mean Squared Error:': rmseValues,
    'Mean Absolute Error': mae_values,
}

x = np.arange(len(models_labels))
width = 0.25
multiplier = 0
bar_colors = ['blue', 'tab:pink']


for attribute, value in ml_data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, value, width, label=attribute)
    ax.bar_label(rects, padding=2)
    multiplier+=1

ax.set_ylabel('Value')
ax.set_title('MAE and RMSE by Model')
ax.set_xlabel('Model')

ax.set_xticks(x + width, models_labels)
ax.set_ylim(0,20)

plt.show()

# Can't change bar colors :(



