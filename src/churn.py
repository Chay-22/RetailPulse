import datetime as dt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def churn(df):
    snapshot = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency','Frequency','Monetary']

    rfm['Churn'] = rfm['Recency'].apply(lambda x: 1 if x > 90 else 0)

    X = rfm[['Recency','Frequency','Monetary']]
    y = rfm['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    rfm['Churn_Prob'] = model.predict_proba(X)[:,1]

    return rfm