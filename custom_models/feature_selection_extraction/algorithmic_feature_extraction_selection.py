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
import joblib
from project_utilities import model_interaction


@numba.jit(forceobj=1)
def preprocess_corpus(corpus: pandas.DataFrame, *columns):
    for column in columns:
        corpus[column] = corpus[column].apply(preprocessing_functionality.clean_text)
    return corpus


class TFIDF_Model:
    vectorizer = TfidfVectorizer

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def fit_to_corpus(self, texts):
        self.vectorizer.fit(texts)

    def from_file(self, filename, model_loader: model_interaction.SKLearnModelFileInteraction):
        self.vectorizer = model_loader.load_from_file(filename)

    def to_file(self, filename):
        joblib.dump(self.vectorizer, filename)

    def vectorize_description(self, description):
        return self.vectorizer.transform([description]).toarray()

    def vectorize_descriptions(self, descriptions):
        return self.vectorizer.transform(descriptions).toarray()

    @staticmethod
    def split_dataset(percentage_testing: float, X, y):
        return train_test_split(X, y, test_size=percentage_testing, random_state=1000)


'''def Main():
    # Get Dataset
    dataset = my_datasets.ITSupportDatasetBuilder() \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    tfidf = ITSupportTFIDFImplementation()
    tfidf.fit_to_corpus(dataset['Description'].tolist())
    vectorised_descriptions = tfidf.vectorize_descriptions(dataset['Description'].tolist())
    # print(tfidf.vectorized_descriptions[0].shape)
    X_train, X_test, y_train, y_test = tfidf.split_dataset(0.1, vectorised_descriptions, dataset['Priority'].tolist())

    logreg = ML_classifiers.ITMultinomialLogisticRegression()
    logreg.use_preconfigured_model('logreg_model.joblib', model_interaction.SKLearnModelFileInteraction())
    # logreg.use_preconfigured_model('logreg_model.joblib', model_interaction.SKLearnModelFileInteraction())
     print('Training Model')
    logreg.train_model()
    joblib.dump(logreg, "logreg_model.joblib")
    print("finished!")
    # print(X_train, X_test)
    # logreg.train_model(X_train, y_train)
    # logreg.save_model('logreg_model.joblib', model_interaction.SKLearnModelFileInteraction())
    label_predictions = logreg.make_predictions(X_test)

    # print('Made Predictions') #classification_report(tfidf.testing_labels, label_predictions))
    labels = ['P5', 'P4', 'P3', 'P2', 'P1']
    from sklearn import metrics

    # print(metrics.classification_report(y_test, label_predictions))
    cm = evaluators.ITSupportPriorityConfusionMatrixEvaluator(label_predictions, y_test, labels)
    cm.plot_confusion_matrix(fullscreen_requested=True)
    # user_issue = input("Enter ticket desc: ")'''


if __name__ == '__main__':
    # Main()
    # Get Dataset
    '''dataset = my_datasets.ITSupportDatasetBuilder() \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    tfidf = ITSupportTFIDFImplementation()
    tfidf.fit_to_corpus(dataset['Description'].tolist())
    tfidf.to_file('tfidf_model.joblib')'''
    # Get Dataset
    '''dataset = my_datasets.ITSupportDatasetBuilder() \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    tfidf = ITSupportTFIDFImplementation(dataset)
    tfidf.vectorize_descriptions()
    logreg = joblib.load("logreg_model.joblib")
    IT_issue = input("Enter IT issue to be prioritised: ")
    preprocessed_input = tfidf.vectorize_description(IT_issue)
    label_predictions = logreg.make_predictions(preprocessed_input)
    print(label_predictions)'''
