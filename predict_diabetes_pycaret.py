import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    df = pd.read_excel(filepath, index_col='Patient number')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, data=df)
    predictions.rename({'Label': 'Diabetes'}, axis=1, inplace=True)
    predictions['Diabetes'].replace({1: 'Diabetes', 0: 'No diabetes'},
                                            inplace=True)
    return predictions['Diabetes']


if __name__ == "__main__":
    df = load_data('../data/diabetes_data.xlsx')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
