#import ITSupportTicketPrioritisers.FromCSV_TFIDF_KerasCNN_ToCSV
#import ITSupportTicketPrioritisers.DefaultDatasets_TFIDF_KerasCNN_ToCSV
#import ITSupportTicketPrioritisers.NoDataset_TFIDF_KerasCNN
#import ITSupportTicketPrioritisers.DefaultDatasets_TFIDF_SKLearnLogReg_ToCSV

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from custom_models.classifiers.DL_classifiers import KerasCNN
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from project_utilities.evaluators import DetailedConfusionMatrix, AccuracyPerClass
from project_utilities.model_interaction import SKLearnModelFileInteraction, KerasModelFileInteraction
from project_utilities import predictionformats
from project_utilities.my_datasets import ITSupportDatasetBuilder
from projectsettings import DefaultConfig
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Conv1D, MaxPooling1D, Dropout, LSTM
from project_utilities import my_datasets
from keras.callbacks import EarlyStopping
from project_utilities import evaluators
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2

# Load Dataset
dataset = ITSupportDatasetBuilder(f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets.csv",
                                  f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets_High_Prio.csv") \
    .with_summaries_and_descriptions_combined() \
    .with_overall_priority_column() \
    .with_pre_processed_descriptions() \
    .build().corpus

# Load Pre-configured TF-IDF
TFIDF_model = TFIDF_Model()
TFIDF_model.from_file(
    f'{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/tfidf_model.joblib',
    SKLearnModelFileInteraction())

# Split dataset into test and train
X_train_str, X_test_str, y_train, y_test = TFIDF_model.split_dataset(0.1, dataset['Description'].tolist(),
                                                                     dataset['Priority'].tolist())

X_train_tfidf = TFIDF_model.vectorize_descriptions(X_train_str)
X_test_tfidf = TFIDF_model.vectorize_descriptions(X_test_str)

# Encode class labels
encoder = LabelEncoder()
encoder.fit(['P5', 'P4', 'P3', 'P2', 'P1'])

y_train = encoder.transform(y_train)
y_val = encoder.transform(y_test)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

input_dim = X_train_tfidf.shape[1]

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.60))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(5, activation='softmax'))

# Compile model
opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

# Train model
from keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train_tfidf, y_train, epochs=50, batch_size=50, validation_data=(X_test_tfidf, y_val), callbacks=[early_stopping])
model.save('CNN_model_larger_regularised.h5')
print("finished")

'''# Make predictions
encoded_predictions = CNN_model.make_predictions(X_test)
decoded_predictions = encoder.inverse_transform(encoded_predictions.argmax(axis=1))

# Represent accuracies
confusion_matrix = DetailedConfusionMatrix(decoded_predictions, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
confusion_matrix.plot_confusion_matrix(fullscreen_requested=True)

apc = AccuracyPerClass(decoded_predictions, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
apc.plot_confusion_matrix()'''