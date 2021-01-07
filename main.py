import joblib
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

def AvgMinutes(Count, Duration):
    if Duration == 0:
        output = 0
    elif Duration != 0:
        output = float(Duration)/float(Count)
    return output
    
Columns = ['Administrative', 'Administrative_Duration',
           'Informational', 'Informational_Duration',
           'ProductRelated', 'ProductRelated_Duration']

class CustomTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['AvgAdministrative'] = X.apply(lambda x: AvgMinutes(Count = x['Administrative'], Duration = x['Administrative_Duration']), axis = 1)
        X['AvgInformational'] = X.apply(lambda x: AvgMinutes(Count = x['Informational'], Duration = x['Informational_Duration']), axis = 1)
        X['AvgProductRelated'] = X.apply(lambda x: AvgMinutes(Count = x['ProductRelated'], Duration = x['ProductRelated_Duration']), axis = 1)
        X.drop(['Administrative', 'Administrative_Duration','Informational', 'Informational_Duration','ProductRelated', 'ProductRelated_Duration'], axis = 1, inplace = True)
        return X[['AvgAdministrative', 'AvgInformational', 'AvgProductRelated']]



if __name__ == '__main__':
    clf = joblib.load('./BDSFinalProject-master/model.pkl')
    X = pd.read_csv('./BDSFinalProject-master/ShoppingData.csv')
    X['Weekend'] = X['Weekend'].astype(int)
    X = X.drop('Revenue', axis = 1)
    foo = X.iloc[[138]]
    bar = clf.predict(foo)
    print(bar)