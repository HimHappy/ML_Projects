import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

file_path = 'D:\Baitha\program\Python\ML\data.csv'

if os.path.exists(file_path):
    housing = pd.read_csv(file_path)
    # print(housing.head())
else:
    print(f"The file '{file_path}' does not exist.")

# print(housing.info())
# print(housing['CHAS'].value_counts())
# print(housing.describe())
# print(housing['RM'].describe())


# # For plotting histogram
import matplotlib.pyplot as plt
# housing.hist(bins=60, figsize=(30, 20))
# plt.show()
# %matplotlib inline #no need to use this outside jupyter

# Train Test Split
# For learning purpose THIS IS ALREADY PRESENT IN SKLEARN
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:] 
#     return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
# THIS IS NOT A GOOD SPLITTING BECAUSE CHAS  HAVE ONLY TWO CATEGORIES BUT IT WILL GIVE ONLY ONE MAY BE 
# SO WE WILL USE STRATIFIED SUFFELING
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_train_set['CHAS'].value_counts())
print(strat_test_set['CHAS'].value_counts())
housing = strat_train_set.copy()

# Correction 
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
# from pandas.plotting import scatter_matrix
# attributes = ["MEDV", "RM", "ZN", "LSTAT"]
# scatter_matrix(housing[attributes], figsize = (12,8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)
# plt.show()

# TRYING DIFFRENT ATTRIBUTE
housing["TAXRM"] = housing['TAX']/housing['RM']
# print(housing.head())
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)
# plt.show()

housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# MISSING ATTRIBUTE
# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)
a = housing.dropna(subset=["RM"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged
housing.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged
median = housing["RM"].median() # Compute median for Option 3
housing["RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged
housing.shape
print(housing['RM'].describe()) # before we started filling missing attributes

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") #getting median for eveery attribute so koi bhi null ho to usme bhara jaye
imputer.fit(housing)
# print(imputer.statistics_)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
print(housing_tr['RM'].describe())#after we filled missing attributes


'''
SCIKIT LEARN DESIGN
Primarily, three types of objects
1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters

2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.

3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.
'''
'''
FEATURE SCALING
Primarily, two types of feature scaling methods:
1. Min-max scaling (Normalization)
    (value - min)/(max - min)
    Sklearn provides a class called MinMaxScaler for this
    
2. Standardization
    (value - mean)/std
    Sklearn provides a class called StandardScaler for this
'''
'''CREATING A PIPELINE'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr.shape

# Selecting a desired model for Real Estates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
print(f"Transformed data shape: {prepared_data.shape}")
model.predict(prepared_data)
print(list(some_labels))

'''Evaluating the model'''
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print(rmse)

'''Using better evaluation technique - Cross Validation'''
# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print(rmse_scores)
def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
print_scores(rmse_scores)

'''Saving the model'''
from joblib import dump, load
dump(model, 'Dragon.joblib') 

''' Testing the model on test data'''
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))
print(final_rmse)
print(prepared_data[0])

''' Using the model  to make predictions for new data '''
model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


