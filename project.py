import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import joblib
df = pd.read_csv('E:/archive/MELBOURNE_HOUSE_PRICES_LESS.csv')
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Regionname']
del df['Propertycount']
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
features_df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])
del features_df['Price']
X = features_df.values
y = df['Price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = ensemble.GradientBoostingRegressor(
n_estimators=250,
learning_rate=0.1,
max_depth=5,
min_samples_split=4,
min_samples_leaf=6,
max_features=0.6,
loss='ls'
)
model.fit(X_train, y_train)
joblib.dump(model, 'trained_model.pkl')
mse = mean_absolute_error(y_train, model.predict(X_train))
print ("Training Set Mean Absolute Error: %.2f" % mse)
mse = mean_absolute_error(y_test, model.predict(X_test))
print ("Test Set Mean Absolute Error: %.2f" % mse)
