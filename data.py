import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.00

df['PCT_change'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.00

df = df[['Adj. Close','PCT_change','HL_PCT','Adj. Volume']]

forecast_col = 'Adj. Close'

df.fillna(-99999,inplace=True)

# print(df.tail())

days = int(math.ceil(0.1*len(df)))

df['label'] = df[forecast_col].shift(-days)

df.dropna(inplace=True)

print(df)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[days:]

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)

clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)


predicted_set = clf.predict(X_lately)
print(predicted_set, accuracy, days)
