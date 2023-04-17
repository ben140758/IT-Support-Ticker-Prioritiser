from project_utilities.ModelTemplates import KerasDeepLearningModel
from project_utilities.my_datasets import ITSupportDatasetBuilder
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from project_utilities.model_interaction import KerasModelFileInteraction, SKLearnModelFileInteraction
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

class KerasCNN(KerasDeepLearningModel):
    def __init__(self, model=None):
        super().__init__()

    def train_model(self, vectors, labels, test_vectors, test_labels, epochs=50, batch_size=50, callbacks=None):
        if not callbacks:
            self.model.fit(vectors, labels, epochs=50, batch_size=50, validation_data=(test_vectors, test_labels), )
        self.model.fit(vectors, labels, epochs=50, batch_size=50, validation_data=(test_vectors, test_labels),
                       callbacks=[callbacks])

    def make_predictions(self, vectors):
        return self.model.predict(vectors)


if __name__ == "__main__":
    dataset = ITSupportDatasetBuilder() \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    TFIDF_model = TFIDF_Model()
    TFIDF_model.from_file('/custom_models/feature_selection_extraction/tfidf_model.joblib',
                          SKLearnModelFileInteraction())

    CNN_model = KerasCNN()
    CNN_model.from_file('custom_models/classifiers/CNN_model.h5', KerasModelFileInteraction())

    vectorised_descriptions = TFIDF_model.vectorize_descriptions(dataset['Description'].tolist())
    X_train, X_test, y_train, y_test = TFIDF_model.split_dataset(0.1, vectorised_descriptions,
                                                                             dataset['Priority'].tolist())
    encoder = LabelEncoder().fit(['P5', 'P4', 'P3', 'P2', 'P1'])
    y_train = to_categorical(encoder.transform(y_train))
    y_val = to_categorical(encoder.transform(y_test))

    encoded_predictions = CNN_model.make_predictions(X_test)
    decoded_predictions = encoder.inverse_transform(encoded_predictions.argmax(axis=1))





