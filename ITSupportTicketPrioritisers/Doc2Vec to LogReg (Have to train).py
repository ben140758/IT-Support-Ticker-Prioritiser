from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import custom_models.classifiers.ML_classifiers
from custom_models.classifiers.DL_classifiers import KerasCNN
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from custom_models.feature_selection_extraction.ML_DL_feature_extraction_selection import ITSupportDoc2VecImplementation, Doc2VecModels
from project_utilities.evaluators import DetailedConfusionMatrix, AccuracyPerClass
from project_utilities.model_interaction import SKLearnModelFileInteraction, KerasModelFileInteraction, GensimWordEmbeddingModelFileInteraction
from project_utilities import predictionformats
from project_utilities.my_datasets import ITSupportDatasetBuilder
from projectsettings import DefaultConfig
import pandas as pd


# Load Dataset
dataset = ITSupportDatasetBuilder(f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets.csv",
                                  f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets_High_Prio.csv",
                                  f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/synonym_IT_tickets.csv") \
    .with_summaries_and_descriptions_combined() \
    .with_overall_priority_column() \
    .with_pre_processed_descriptions() \
    .build().corpus

# Split dataset into test and train
X_train_str, X_test_str, y_train, y_test = TFIDF_Model.split_dataset(0.1, dataset['Description'].tolist(),
                                                                     dataset['Priority'].tolist())

# Get pre-configured doc2vec model
doc2vec_model = ITSupportDoc2VecImplementation(Doc2VecModels.DBOW)
'''doc2vec_model.from_file(
                f"{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/doc2vec_model.model",
                GensimWordEmbeddingModelFileInteraction())'''
tagged_training_documents = doc2vec_model.tag_documents(pd.DataFrame({'Description': X_train_str, 'Priority': y_train}))
tagged_testing_documents = doc2vec_model.tag_documents(pd.DataFrame({'Description': X_test_str, 'Priority': y_test}))
doc2vec_model.build_vocabulary(tagged_training_documents)
doc2vec_model.train_model(tagged_training_documents, dataset_shuffles=10, epochs=10)
#doc2vec_model.to_file("doc2vec_model.model", model_interaction.GensimWordEmbeddingModelFileInteraction())
#tagged_descriptions = doc2vec_model.tag_documents(X_test_str)
X_train = doc2vec_model.vectorize_documents(X_train_str)
X_test = doc2vec_model.vectorize_documents(X_test_str)

# Load Logistic Regression model
logreg_model = custom_models.classifiers.ML_classifiers.ITMultinomialLogisticRegression(cores_allocated=1)
'''logreg_model.use_preconfigured_model(
    f'{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/doc2vec_to_logreg_model.joblib',
    SKLearnModelFileInteraction())'''
logreg_model.train_model(vectors=X_train, labels=y_train)
#logreg_model.save_model('doc2vec_to_logreg_model.joblib', SKLearnModelFileInteraction())

# Make predictions
predictions = logreg_model.make_predictions(X_test)

# Represent accuracies
confusion_matrix = DetailedConfusionMatrix(predictions, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
confusion_matrix.plot_confusion_matrix(fullscreen_requested=True)

apc = AccuracyPerClass(predictions, y_test, ['P5', 'P4', 'P3', 'P2', 'P1'])
apc.plot_confusion_matrix()

