# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import ensemble
# from sklearn.metrics import mean_absolute_error
# import joblib
#
# df = pd.read_csv('E:/archive/MELBOURNE_HOUSE_PRICES_LESS.csv')
# # df.head(n=5)
#  del df['Address']
#  del df['Method']
#  del df['SellerG']
#  del df['Date']
#  del df['Postcode']
#  # del df['Lattitude']
#  # del df['Longtitude']
#  del df['Regionname']
#  del df['Propertycount']
# df.dropna(axis=0,how='any',thresh=None,subset=None,inplace=True)
# features_df = pd.get_dummies(df,columns=['Suburb','CouncilArea','Type'])
# del features_df['Price']
# X = features_df.values
# y = df['Price'].values
# # print(X)
# # print(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# model = ensemble.GradientBoostingRegressor(
#     n_estimators=150,
#     learning_rate=0.1,
#     max_depth=30,
#     min_samples_split=4,
#     min_samples_leaf=6,
#     max_features=0.6,
#     loss='ls'
# )
# model.fit(X_train,y_train)
# joblib.dump(model,'house_trained_model.pkl')
# mse1 = mean_absolute_error(y_train, model.predict(X_train))
# print ("Training Set Mean Absolute Error: %.2f" % mse1)
# mse = mean_absolute_error(y_test, model.predict(X_test))
# print ("Test Set Mean Absolute Error: %.2f" % mse)
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
# Read in data from CSV
df = pd.read_csv('E:/archive/MELBOURNE_HOUSE_PRICES_LESS.csv')
# Delete unneeded columns
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
# del df['Lattitude']
# del df['Longtitude']
del df['Regionname']
del df['Propertycount']
# Remove rows with missing values
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
# Convert non-numerical data using one-hot encoding
features_df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])
# Remove price
del features_df['Price']
# Create X and y arrays from the dataset
X = features_df.values
y = df['Price'].values
# Split data into test/train set (70/30 split) and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Set up algorithm
model = ensemble.GradientBoostingRegressor(
n_estimators=250,
learning_rate=0.1,
max_depth=5,
min_samples_split=4,
min_samples_leaf=6,
max_features=0.6,
loss='ls'
)
# Run model on training data
model.fit(X_train, y_train)
# Save model to file
joblib.dump(model, 'trained_model.pkl')
# Check model accuracy (up to two decimal places)
mse = mean_absolute_error(y_train, model.predict(X_train))
print ("Training Set Mean Absolute Error: %.2f" % mse)
mse = mean_absolute_error(y_test, model.predict(X_test))
print ("Test Set Mean Absolute Error: %.2f" % mse)
