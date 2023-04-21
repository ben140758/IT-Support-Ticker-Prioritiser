import pandas as pd


class ITSupportPredictionFormat:
    predictions: pd.DataFrame

    def __init__(self):
        prediction_format = {'Description': [], 'PredictedPriority': []}
        self.predictions = pd.DataFrame(prediction_format)

    def load_predictions(self, new_predictions: pd.DataFrame):
        self.predictions = pd.concat([self.predictions, new_predictions])

    def save_predictions_to_file(self, filename: str, filetype: str):
        filetypes = {'csv': self._to_csv,
                     'xlsx': self._to_excel}

        filetypes[filetype](filename)

    def _to_csv(self, filename):
        self.predictions = self.predictions.apply(lambda x: x.replace(",", " "))
        self.predictions.to_csv(filename + '.csv', sep=",", columns=['Description', 'PredictedPriority'])

    def _to_excel(self, filename):
        self.predictions.to_excel(filename + '.xlsx', sep=",", columns=['Description', 'PredictedPriority'])
