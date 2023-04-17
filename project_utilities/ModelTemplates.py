from abc import ABC, abstractmethod
from project_utilities import model_interaction
from keras.models import load_model


class SKLearnMachineLearningModel(ABC):
    cores_allocated: int

    @abstractmethod
    def __init__(self, model=None):
        self.model = model

    def use_preconfigured_model(self, filename, model_loader: model_interaction.SKLearnModelFileInteraction):
        self.model = model_loader.load_from_file(filename)

    def save_model(self, filename, model_loader: model_interaction.SKLearnModelFileInteraction):
        model_loader.load_to_file(self.model, filename)

    def train_model(self, vectors, labels):
        self.model.fit(vectors, labels)

    def make_predictions(self, items):
        return self.model.predict(items)


class KerasDeepLearningModel(ABC):

    @abstractmethod
    def __init__(self, model=None):
        self.model = model

    def from_file(self, filename, model_loader: model_interaction.KerasModelFileInteraction):
        self.model = model_loader.load_from_file(filename)

    def to_file(self, filename, model_loader: model_interaction.KerasModelFileInteraction):
        model_loader.load_to_file(self.model, filename)

    def add_model_config(self, layer):
        self.model.add(layer)

    def compile_model(self, loss_function, optimizer, *metrics):
        self.model.compile(loss=loss_function, metrics=[*metrics, ], optimizer=optimizer)

    @abstractmethod
    def train_model(self, vectors, labels, test_vectors, test_labels, epochs, batch_size):
        pass

    @abstractmethod
    def make_predictions(self, vectors):
        pass
