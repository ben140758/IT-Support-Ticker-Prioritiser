from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from custom_models.classifiers.DL_classifiers import KerasCNN
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from project_utilities.evaluators import DetailedConfusionMatrix, AccuracyPerClass
from project_utilities.model_interaction import SKLearnModelFileInteraction, KerasModelFileInteraction
from project_utilities import predictionformats
from project_utilities.my_datasets import ITSupportDatasetBuilder
from projectsettings import DefaultConfig

import pandas as pd

'''f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets.csv",
f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets_High_Prio.csv")'''
dataset_file_loc = input("Enter dataset file_location: ")
dataset = ITSupportDatasetBuilder(dataset_file_loc) \
    .with_summaries_and_descriptions_combined() \
    .with_overall_priority_column() \
    .with_pre_processed_descriptions() \
    .build().corpus

TFIDF_model = TFIDF_Model()
TFIDF_model.from_file(
    f'{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/tfidf_model.joblib',
    SKLearnModelFileInteraction())

CNN_model = KerasCNN()
CNN_model.from_file(f'{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/CNN_model.h5',
                    KerasModelFileInteraction())

#X_train_str, X_test_str, y_train, y_test = TFIDF_model.split_dataset(0.1, dataset['Description'].tolist(),
#                                                                     dataset['Priority'].tolist())

# X_train = TFIDF_model.vectorize_descriptions(X_train_str)
X_test = TFIDF_model.vectorize_descriptions(dataset['Description'].tolist())
# vectorised_descriptions = TFIDF_model.vectorize_descriptions(dataset['Description'].tolist())
# X_train, X_test, y_train, y_test = TFIDF_model.split_dataset(0.1, vectorised_descriptions,
# dataset['Priority'].tolist())

# vectorized_desc = TFIDF_Model.vectorize_description(self=TFIDF_model, description="WIFI network has lost connction across the whole campus, this needs fixing ASAP")
encoder = LabelEncoder().fit(['P5', 'P4', 'P3', 'P2', 'P1'])
'''y_train = to_categorical(encoder.transform(y_train))
y_val = to_categorical(encoder.transform(y_test))'''

encoded_predictions = CNN_model.make_predictions(X_test)
decoded_predictions = encoder.inverse_transform(encoded_predictions.argmax(axis=1))

confusion_matrix = DetailedConfusionMatrix(decoded_predictions, dataset['Priority'].tolist(), ['P5', 'P4', 'P3', 'P2', 'P1'])
confusion_matrix.plot_confusion_matrix(fullscreen_requested=True)
try:
    apc = AccuracyPerClass(decoded_predictions, dataset['Priority'].tolist(), ['P5', 'P4', 'P3', 'P2', 'P1'])
    apc.plot_confusion_matrix()
except ZeroDivisionError:
    pass

dict_descriptions_predictions = {'Description': dataset['Description'].tolist(), 'PredictedPriority': decoded_predictions}
formatted_predictions = pd.DataFrame(dict_descriptions_predictions)
prediction_saver = predictionformats.ITSupportPredictionFormat()
prediction_saver.load_predictions(formatted_predictions)
filename = input("Enter filename: ")
prediction_saver.save_predictions_to_file(filename, 'csv')
