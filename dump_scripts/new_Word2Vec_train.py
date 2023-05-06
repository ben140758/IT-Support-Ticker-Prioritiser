from gensim.models import Word2Vec
from project_utilities import my_datasets
from projectsettings import DefaultConfig
import numpy as np
from custom_models.classifiers.ML_classifiers import ITMultinomialLogisticRegression
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from project_utilities.evaluators import DetailedConfusionMatrix, AccuracyPerClass
# Load Dataset
dataset = my_datasets.ITSupportDatasetBuilder(
    f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets.csv",
    f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets_High_Prio.csv",
    f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/synonym_IT_tickets.csv") \
    .with_summaries_and_descriptions_combined() \
    .with_overall_priority_column() \
    .with_pre_processed_descriptions() \
    .build().corpus
dataset['Description'] = dataset['Description'].apply(lambda x: x.split(' '))
# Split dataset into test and train
X_train_str, X_test_str, y_train, y_test = TFIDF_Model.split_dataset(0.1, dataset['Description'].tolist(),
                                                                     dataset['Priority'].tolist())



# Create and train the Word2Vec model
model = Word2Vec(sentences=X_train_str, vector_size=250, window=5, min_count=3, workers=16)

# Save the model
#model.save("word2vec.model")

def get_vectors(texts):
    X_vectors = []
    for sentence in texts:
        sentence_vectors = []
        for word in sentence:
            if word in model.wv:
                sentence_vectors.append(model.wv[word])
            else:
                # Handle words not in the vocabulary
                sentence_vectors.append(np.zeros(model.vector_size))
        X_vectors.append(np.mean(sentence_vectors, axis=0))
    X_vectors = np.array(X_vectors)
    return X_vectors

X_vectors = get_vectors(X_train_str)
X_test_vectors = get_vectors(X_test_str)

logreg = ITMultinomialLogisticRegression()
logreg.train_model(X_vectors, y_train)

pred = logreg.make_predictions(X_test_vectors)
# Represent accuracies
confusion_matrix = DetailedConfusionMatrix(pred, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
confusion_matrix.plot_confusion_matrix(fullscreen_requested=True)

apc = AccuracyPerClass(pred, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
apc.plot_confusion_matrix()