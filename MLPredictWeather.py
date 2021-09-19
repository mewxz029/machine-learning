import pandas as pd
from scipy import stats 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("Weather.csv")

# train & test set
x = dataset["MinTemp"].values.reshape(-1, 1)
y = dataset["MaxTemp"].values.reshape(-1, 1)

# 80% - 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# training
model = LinearRegression()
model.fit(x_train, y_train)
a = model.intercept_
b = model.coef_

# test
y_pred = model.predict(x_test)

# compare true data & predict data
df = pd.DataFrame({'Actually':y_test.flatten(), 'Predicted':y_pred.flatten()})

def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.
    INPUTS:
    - Single prediction,
    - y_test
    - All test set predictions,
    - Prediction interval threshold (default = .95)
    OUTPUT:
    - Prediction interval for single prediction
    '''
    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
    
    #get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    
    #generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval

    return lower, prediction, upper

z_score = stats.norm.ppf(1 - 0.025)
print(z_score)

## Plot and save confidence interval of linear regression  - 95% 
lower_vet = []
upper_vet = []
for i in y_pred:
    lower, prediction, upper =  get_prediction_interval(i, y_test, y_pred)
    lower_vet.append(lower)
    upper_vet.append(upper)

# lower_vet = np.array(lower_vet)
# upper_vet = np.array(upper_vet)

print("MAE = ", metrics.mean_absolute_error(y_test,y_pred))
print("MSE = ", metrics.mean_squared_error(y_test,y_pred))
print("RMSE = ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Score = ", metrics.r2_score(y_test,y_pred))

plt.scatter(x, y,  color='black')
plt.plot(x_test, y_pred,  label="y = "+ str(b) +"+ "+ str(a) +"x")
plt.plot(x_test, lower_vet, 'r',label="95% Interval")
plt.plot(x_test, upper_vet, 'r')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.legend(loc="upper left")
plt.title("Graph Linear Regression Confidence 95%")

plt.grid()
plt.show()


    

