import keras
import numpy
import pandas as pd

import custom_models.feature_selection_extraction.ML_DL_feature_extraction_selection
import custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection
from project_utilities import my_datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
import tensorflow as tf
from keras import metrics
from pandas import DataFrame


class PresetSoftmaxClassifier:
    vectorized_dataset = DataFrame
    classes = list

    def __init__(self, vectorized_dataset, classes: list):
        self.vectorized_dataset = vectorized_dataset
        self.classes = classes



if __name__ == '__main__':
    dataset = my_datasets.ITSupportDatasetBuilder() \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    doc2vec_IT = custom_models.feature_selection_extraction.ML_DL_feature_extraction_selection.ITSupportDoc2VecImplementation(
        dataset=dataset, model_type=custom_models.feature_selection_extraction.ML_DL_feature_extraction_selection.Doc2VecModels.DBOW)
    # doc2vec_IT.pre_process_texts()
    doc2vec_IT.tag_documents()
    doc2vec_IT.create_model()
    doc2vec_IT.build_vocabulary()
    doc2vec_IT.train_model(dataset_shuffles=1, epochs=10)  # dataset_shuffles=10, epochs=30)
    print("Got here 0.5")
    doc2vec_IT.generate_vectors()

    Z = tf.keras.utils.to_categorical(dataset.Priority, num_classes=5)
    print(Z)

    '''descriptions_train, descriptions_test, tfidf.training_labels, tfidf.testing_labels = train_test_split(
        dataset.Descriptions, Z, test_size=0.3,
        random_state=1000)

    vectorizer.fit(descriptions_train)
    tfidf.training_descriptions = vectorizer.transform(descriptions_train)
    tfidf.testing_descriptions = vectorizer.transform(descriptions_test)'''
    # tfidf.training_labels = tf.keras.utils.to_categorical(tfidf.training_labels, num_classes=5)
    print(dataset.train_labels)
    # vectorizer.fit(tfidf.training_labels)
    input_dim = dataset.training_descriptions.shape[1]#tfidf.training_descriptions.shape[1]

    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.Recall()])

    # model.summary()

    history = model.fit(tfidf.training_descriptions, tfidf.training_labels,
                        epochs=100,
                        verbose=False,
                        validation_data=(tfidf.testing_descriptions, tfidf.testing_labels),
                        batch_size=5)

    loss, accuracy = model.evaluate(tfidf.testing_descriptions, tfidf.testing_labels, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # matrix = metrics.confusion_matrix(tfidf.testing_labels.argmax(axis=1), y_prediction.argmax(axis=1))
    y_prediction = model.predict(tfidf.testing_descriptions)
    y_prediction = numpy.argmax(y_prediction, axis=1)
    tfidf.testing_labels = numpy.argmax(tfidf.testing_labels, axis=1)
    print(keras.metrics.categorical_accuracy(tfidf.testing_labels, y_prediction))
    # tf.keras.metrics.confusion_matrix(tfidf.testing_labels.argmax(axis=1), y_prediction.argmax(axis=1))

    # cm = ITSupportPriorityConfusionMatrixEvaluator(predictions=y_prediction, actual_values=tfidf.testing_labels, labels=['P1', 'P2', 'P3', 'P4', 'P5'])
    # clear_session()

    # keras.metrics.confusion_matrix(tfidf.testing_labels, y_prediction)
    '''from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression()
    classifier.fit(tfidf.training_descriptions, tfidf.training_labels)
    score = classifier.score(tfidf.testing_descriptions, tfidf.testing_labels)'''

    from scikitplot.metrics import plot_confusion_matrix
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    num_to_pnum = ['P5', 'P4', 'P3', 'P2', 'P1']
    tfidf.testing_labels_lab = [num_to_pnum[x] for x in tfidf.testing_labels]
    y_pred_lab = [num_to_pnum[x] for x in y_prediction]
    # print(tfidf.testing_labels_lab, type(tfidf.testing_labels))
    # plot_confusion_matrix(tfidf.testing_labels_lab, y_pred_lab, ax=ax, labels=['P1', 'P2', 'P3', 'P4', 'P5'])
    # plt.show()
    from project_utilities.evaluators import ITSupportPriorityConfusionMatrixEvaluator

    cm = ITSupportPriorityConfusionMatrixEvaluator(
        predictions=y_pred_lab,
        actual_values=tfidf.testing_labels_lab,
        labels=['P1', 'P2', 'P3', 'P4', 'P5'])
    cm.plot_confusion_matrix(fullscreen_requested=True)
