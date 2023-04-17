from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from custom_models.classifiers.DL_classifiers import KerasCNN
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from project_utilities.evaluators import DetailedConfusionMatrix, AccuracyPerClass
from project_utilities.model_interaction import SKLearnModelFileInteraction, KerasModelFileInteraction
from project_utilities.my_datasets import ITSupportDatasetBuilder

dataset = ITSupportDatasetBuilder() \
    .with_summaries_and_descriptions_combined() \
    .with_overall_priority_column() \
    .with_pre_processed_descriptions() \
    .build().corpus

TFIDF_model = TFIDF_Model()
TFIDF_model.from_file('custom_models/feature_selection_extraction/tfidf_model.joblib', SKLearnModelFileInteraction())

CNN_model = KerasCNN()
CNN_model.from_file('custom_models/classifiers/CNN_model.h5', KerasModelFileInteraction())

vectorised_descriptions = TFIDF_model.vectorize_descriptions(dataset['Description'].tolist())
X_train, X_test, y_train, y_test = TFIDF_model.split_dataset(0.1, vectorised_descriptions,
                                                             dataset['Priority'].tolist())

#vectorized_desc = TFIDF_Model.vectorize_description(self=TFIDF_model, description="WIFI network has lost connction across the whole campus, this needs fixing ASAP")
encoder = LabelEncoder().fit(['P5', 'P4', 'P3', 'P2', 'P1'])
'''y_train = to_categorical(encoder.transform(y_train))
y_val = to_categorical(encoder.transform(y_test))'''

encoded_predictions = CNN_model.make_predictions(X_test)
decoded_predictions = encoder.inverse_transform(encoded_predictions.argmax(axis=1))

confusion_matrix = DetailedConfusionMatrix(decoded_predictions, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
confusion_matrix.plot_confusion_matrix(fullscreen_requested=True)

apc = AccuracyPerClass(decoded_predictions, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
apc.plot_confusion_matrix()

