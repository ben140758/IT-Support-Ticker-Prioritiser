from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from project_utilities.ModelTemplates import SKLearnMachineLearningModel


class ITMultinomialLogisticRegression(SKLearnMachineLearningModel):
    def __init__(self, inverse_regularisation_strength: float = 1e5, cores_allocated: int = 1):
        super().__init__(LogisticRegression(n_jobs=cores_allocated,
                                            C=inverse_regularisation_strength,
                                            multi_class='multinomial',
                                            solver='newton-cg',
                                            verbose=1))


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
    # logreg = ITMultinomialLogisticRegression(6, 1e5)
    pass
