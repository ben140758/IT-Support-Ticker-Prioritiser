import numpy
from tqdm import tqdm
from project_utilities import my_datasets, preprocessing_functionality
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from project_utilities import evaluators
import pandas
import numba
from custom_models.classifiers import ML_classifiers



@numba.jit(forceobj=1)
def preprocess_corpus(corpus: pandas.DataFrame, *columns):
    for column in columns:
        corpus[column] = corpus[column].apply(preprocessing_functionality.clean_text)
    return corpus


class ITSupportTFIDFImplementation:
    vectorizer = TfidfVectorizer
    dataset = pandas.DataFrame
    vectorized_descriptions: list
    training_descriptions = \
        testing_descriptions = \
        training_labels = \
        testing_labels = numpy.ndarray

    def __init__(self, dataset: pandas.DataFrame):
        tqdm.pandas(desc="progress-bar")
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.dataset = dataset

    def vectorize_descriptions(self):
        self.vectorized_descriptions = self.vectorizer.fit_transform(self.dataset['Description'].values).toarray()

    def split_dataset(self, percentage_testing: float):
        self.training_descriptions, self.testing_descriptions, self.training_labels, self.testing_labels = \
            train_test_split(self.vectorized_descriptions, self.dataset['Priority'].values,
                             test_size=percentage_testing, random_state=1000)


def Main():

    # Get Dataset
    dataset = my_datasets.ITSupportDatasetBuilder() \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    tfidf = ITSupportTFIDFImplementation(dataset)
    tfidf.vectorize_descriptions()
    tfidf.split_dataset(0.1)

    logreg = ML_classifiers.ITMultinomialLogisticRegression(vectors=tfidf.training_descriptions,
                                                            labels=tfidf.training_labels,
                                                            cores_allocated=-1,
                                                            inverse_regularisation_strength=1e5)
    print('Training Model')
    logreg.train_model()
    label_predictions = logreg.make_predictions(tfidf.testing_descriptions)

    print('Made Predictions') #classification_report(tfidf.testing_labels, label_predictions))
    labels = ['P5', 'P4', 'P3', 'P2', 'P1']
    cm = evaluators.ITSupportPriorityConfusionMatrixEvaluator(label_predictions, tfidf.testing_labels, labels)
    cm.plot_confusion_matrix(fullscreen_requested=True)

if __name__ == '__main__':
    Main()
