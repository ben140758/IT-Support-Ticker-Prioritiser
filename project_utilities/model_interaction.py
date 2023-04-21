from abc import ABC, abstractmethod
import joblib
from keras.models import load_model, save_model


class FileInteraction(ABC):
    @staticmethod
    @abstractmethod
    def load_from_file(filename):
        pass

    @staticmethod
    @abstractmethod
    def load_to_file(model, filename):
        pass


class SKLearnModelFileInteraction(FileInteraction):

    @staticmethod
    def load_from_file(filename):
        return joblib.load(filename)

    @staticmethod
    def load_to_file(model, filename):
        joblib.dump(model, filename)


class KerasModelFileInteraction(FileInteraction):
    @staticmethod
    def load_from_file(filename):
        return load_model(filename)

    @staticmethod
    def load_to_file(model, filename):
        save_model(model, filename)
