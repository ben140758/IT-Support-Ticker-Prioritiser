from sklearn.preprocessing import LabelEncoder
from custom_models.classifiers.DL_classifiers import KerasCNN
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from project_utilities.model_interaction import SKLearnModelFileInteraction, KerasModelFileInteraction
from projectsettings import DefaultConfig
from threading import Timer
import sys

# Load Pre-configured TF-IDF
TFIDF_model = TFIDF_Model()
TFIDF_model.from_file(
    f'{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/tfidf_model.joblib',
    SKLearnModelFileInteraction())

# Load Pre-configured Keras CNN
CNN_model = KerasCNN()
CNN_model.from_file(f'{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/CNN_model.h5',
                    KerasModelFileInteraction())

# Convert P1-5 into categories the model understands
encoder = LabelEncoder().fit(['P5', 'P4', 'P3', 'P2', 'P1'])

timer_finished = False
def amend_timer():
    global timer_finished
    timer_finished = True


while True:
    timer_finished = False
    print("Paste IT issue here: ")
    contents = []
    t = Timer(1, amend_timer)
    t.start()
    while not timer_finished:
        line = input()
        contents.append(line)
    t.cancel()
    print("...")
    description = ' '.join(contents)

    # Convert the Descriptions to Sparse Matrices, representative of text
    vectorized_desc = TFIDF_model.vectorize_description(description=repr(description))
    # Make prediction
    encoded_prediction = CNN_model.make_predictions(vectorized_desc)
    # Convert prediction back to P5-P1
    decoded_prediction = encoder.inverse_transform(encoded_prediction.argmax(axis=1))
    print(decoded_prediction)
