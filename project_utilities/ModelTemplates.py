from abc import ABC, abstractmethod
from project_utilities import model_interaction
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec


class SKLearnMachineLearningModel(ABC):
    cores_allocated: int

    @abstractmethod
    def __init__(self, model=None):
        self.model = model

    # Load model from file
    def use_preconfigured_model(self, filename, model_loader: model_interaction.FileInteraction):
        self.model = model_loader.load_from_file(filename)

    # Load model to file
    def save_model(self, filename, model_loader: model_interaction.FileInteraction):
        model_loader.load_to_file(self.model, filename)

    # Train model
    def train_model(self, vectors, labels):
        self.model.fit(vectors, labels)

    # Given items, predict priority
    def make_predictions(self, items):
        return self.model.predict(items)


class KerasDeepLearningModel(ABC):

    @abstractmethod
    def __init__(self, model=None):
        self.model = model

    def from_file(self, filename, model_loader: model_interaction.FileInteraction):
        self.model = model_loader.load_from_file(filename)

    def to_file(self, filename, model_loader: model_interaction.FileInteraction):
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


class GensimWordEmbeddingModel(ABC):
    model: Doc2Vec or Word2Vec

    def __init__(self, model=None):
        self.model = model

    def from_file(self, filename, model_loader: model_interaction.GensimWordEmbeddingModelFileInteraction):
        self.model = model_loader.load_from_file(filename)
        print(self.model)

    def to_file(self, filename, model_loader: model_interaction.GensimWordEmbeddingModelFileInteraction):
        model_loader.load_to_file(self.model, filename)
