import connexion
import joblib
import pandas as pd
import json
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

Train_Cols = ['Administrative',
            'Administrative_Duration',
            'Informational',
            'Informational_Duration',
            'ProductRelated',
            'ProductRelated_Duration',
            'BounceRates',
            'ExitRates',
            'PageValues',
            'SpecialDay',
            'Month',
            'OperatingSystems',
            'Browser',
            'Region',
            'TrafficType',
            'VisitorType',
            'Weekend']

class CustomTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['AvgAdministrative'] = X.apply(lambda x: AvgMinutes(Count = x['Administrative'], Duration = x['Administrative_Duration']), axis = 1)
        X['AvgInformational'] = X.apply(lambda x: AvgMinutes(Count = x['Informational'], Duration = x['Informational_Duration']), axis = 1)
        X['AvgProductRelated'] = X.apply(lambda x: AvgMinutes(Count = x['ProductRelated'], Duration = x['ProductRelated_Duration']), axis = 1)
        X.drop(['Administrative', 'Administrative_Duration','Informational', 'Informational_Duration','ProductRelated', 'ProductRelated_Duration'], axis = 1, inplace = True)
        return X[['AvgAdministrative', 'AvgInformational', 'AvgProductRelated']]

# inputs
model_file_name = './model.pkl'

# These will be populated at training time
clf = joblib.load(model_file_name)


def predict(payloads):
    if clf:
        try:
            json_ = payloads

            # import pdb; pdb.set_trace()

            query = pd.read_json(json.dumps(json_), orient='records')

            prediction = list(clf.predict(query[Train_Cols]))

            # Converting to int from int64
            return {"prediction": list(map(int, prediction))}

        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc()}
    else:
        print('train first')
        return 'no model here'



app = connexion.App(__name__)
app.add_api('swagger.yml')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 8889

    app.run(host='0.0.0.0', port=port, debug=True)