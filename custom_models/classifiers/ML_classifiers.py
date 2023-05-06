from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from project_utilities.ModelTemplates import SKLearnMachineLearningModel
from custom_models.feature_selection_extraction.algorithmic_feature_extraction_selection import TFIDF_Model
from projectsettings import DefaultConfig
from project_utilities.model_interaction import SKLearnModelFileInteraction
from project_utilities import my_datasets
class ITMultinomialLogisticRegression(SKLearnMachineLearningModel):
    def __init__(self, inverse_regularisation_strength: float = 1e5, cores_allocated: int = 1):
        super().__init__(LogisticRegression(n_jobs=cores_allocated,
                                            C=inverse_regularisation_strength,
                                            multi_class='multinomial',
                                            solver='newton-cg',
                                            verbose=1,
                                            max_iter=10000))


class ITMultinomialNaiveBayes(SKLearnMachineLearningModel):
    def __init__(self):
        super().__init__(MultinomialNB())


class ITSupportVectorClassifier(SKLearnMachineLearningModel):
    def __init__(self):
        super().__init__(LinearSVC())


class ITRandomForestClassifier(SKLearnMachineLearningModel):
    def __init__(self, tree_quantity: int = 200, max_tree_depth: int = 10, randomness: int = 1):
        super().__init__(RandomForestClassifier(n_estimators=tree_quantity, max_depth=max_tree_depth, random_state=randomness))


if __name__ == "__main__":
    # Get Dataset
    dataset = my_datasets.ITSupportDatasetBuilder(
        f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets.csv",
        f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/ITSupport_Tickets_High_Prio.csv",
        f"{DefaultConfig.absolute_project_root_path()}/project_utilities/Datasets/synonym_IT_tickets.csv") \
        .with_summaries_and_descriptions_combined() \
        .with_overall_priority_column() \
        .with_pre_processed_descriptions() \
        .build().corpus

    logreg = ITMultinomialLogisticRegression(1e5, 6)
    tfidf = TFIDF_Model()
    tfidf.from_file(f"{DefaultConfig.absolute_project_root_path()}/custom_models/preconfigured_models/tfidf_larger_model.joblib", SKLearnModelFileInteraction())
    X = dataset['Description'].tolist()
    y = dataset['Priority'].tolist()
    X_train_str, X_test_str, y_train, y_test = tfidf.split_dataset(0.1, X, y)
    X_train = tfidf.vectorize_descriptions(X_train_str)
    X_test = tfidf.vectorize_descriptions(X_test_str)
    logreg.train_model(X_train, y_train)
    logreg.save_model('tfidf_to_logreg_model.joblib', SKLearnModelFileInteraction())

