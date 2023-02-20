from enum import Enum
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


class ModelType(Enum):
    MULTINOMIAL_LOGISTIC_REGRESSION = 1
    MULTINOMIAL_NAIVE_BAYES = 2
    LINEAR_SUPPORT_VECTOR_CLASSIFICATION = 3
    RANDOM_FOREST = 4


class ITMachineLearningClassifierImplementation:
    cores_allocated: int

    def __init__(self, vectors, labels, cores_allocated: int = 1) -> None:
        self.model = None
        self.cores_allocated = cores_allocated
        self.vectors = vectors
        self.labels = labels

    def use_preconfigured_model(self, preconfigured_model):
        self.model = preconfigured_model

    def train_model(self):
        self.model.fit(self.vectors, self.labels)

    def make_predictions(self, items):
        return self.model.predict(items)


class ITMultinomialLogisticRegression(ITMachineLearningClassifierImplementation):
    def __init__(self, vectors, labels, inverse_regularisation_strength: float, cores_allocated: int = 1):
        super().__init__(vectors=vectors, labels=labels, cores_allocated=cores_allocated)
        self.model = LogisticRegression(n_jobs=self.cores_allocated,
                                        C=inverse_regularisation_strength,
                                        multi_class='multinomial',
                                        solver='newton-cg',
                                        verbose=1)


class ITMultinomialNaiveBayes(ITMachineLearningClassifierImplementation):
    def __init__(self, vectors, labels):
        super().__init__(vectors, labels)
        self.model = MultinomialNB()


class ITSupportVectorClassifier(ITMachineLearningClassifierImplementation):
    def __init__(self, vectors, labels):
        super().__init__(vectors, labels)
        self.model = LinearSVC()


class ITRandomForestClassifier(ITMachineLearningClassifierImplementation):
    def __init__(self, vectors, labels, tree_quantity: int = 200, max_tree_depth: int = 10, randomness: int = 1):
        super().__init__(vectors, labels)
        RandomForestClassifier(n_estimators=tree_quantity, max_depth=max_tree_depth, random_state=randomness)


if __name__ == "__main__":
    # logreg = ITMultinomialLogisticRegression(6, 1e5)
    pass
