import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')
print(df)

df = df[['Adj. High','Adj. Low','Adj. Open','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.00

df['PCT_change'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.00

df = df[['Adj. Close','PCT_change','HL_PCT','Adj. Volume']]

print(df.head())
