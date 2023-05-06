import multiprocessing
import time
from enum import Enum

import gensim.models
import gensim.models.doc2vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from pandas import DataFrame
from sklearn import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from project_utilities import model_interaction
from project_utilities.ModelTemplates import GensimWordEmbeddingModel
from gensim.models.doc2vec import Doc2Vec
# from project_utilities import preprocessing_functionality, my_datasets
from project_utilities import my_datasets
import preprocessing_functionality


class Doc2VecModels(Enum):
    DBOW = 1
    DM = 2


class ITSupportDoc2VecImplementation(GensimWordEmbeddingModel):
    model_type = Doc2VecModels

    def __init__(self, model_type, alpha_change=-.002):
        self.model_type = model_type
        model = self.create_model()
        self.alpha_change = alpha_change
        tqdm.pandas(desc="progress-bar")
        super().__init__(model)

    @staticmethod
    def split_texts(dataset):
        training_data, testing_data = train_test_split(dataset, test_size=0.1, random_state=1000)
        return training_data, testing_data

    def tag_documents(self, documents) -> DataFrame:
        tagged_documents = documents.apply(
            lambda docs: gensim.models.doc2vec.TaggedDocument(
                words=preprocessing_functionality.tokenize_text(docs.Description),
                tags=[docs.Priority]),
            axis=1)

        return tagged_documents

    def create_model(self):
        cores = multiprocessing.cpu_count()
        match self.model_type:
            case Doc2VecModels.DBOW:
                return self._create_dbow_model(cores)
            case Doc2VecModels.DM:
                return self._create_dm_model(cores)
            case _:
                raise TypeError("Must be a Doc2Vec model type (DBOW, DM, COMBINED)")

    def _create_dbow_model(self, cores):
        model = Doc2Vec(
            dm=0, vector_size=1000, negative=5, hs=0, min_count=2, sample=0, workers=cores)
        self.alpha_change = 0.0002
        return model

    def _create_dm_model(self, cores):
        model = Doc2Vec(
            dm=1, dm_mean=1, vector_size=300, window=10, negative=5,
            min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
        self.alpha_change = -0.002
        return model

    def build_vocabulary(self, tagged_training_documents):
        vocabulary = [x for x in tqdm(tagged_training_documents.values)]
        self.model.build_vocab(vocabulary)

    def train_model(self, tagged_training_documents, dataset_shuffles: int = 1, epochs: int = 1):
        for training_round in range(dataset_shuffles):
            # shuffle training data
            shuffled_training_data = utils.shuffle([x for x in tqdm(tagged_training_documents.values)])
            dataset_size = len(tagged_training_documents)
            self.model.train(shuffled_training_data, total_examples=dataset_size,epochs=epochs)
            self.model.alpha += self.alpha_change
            self.model.min_alpha = self.model.alpha

    # @numba.jit(forceobj=True)
    def vectorize_tagged_documents(self, tagged_documents):
        sentences = tagged_documents.values
        targets, regressors = zip(*[(doc.tags[0], self.model.infer_vector(doc.words)) for doc in sentences])
        return targets, regressors

    def generate_training_vectors(self, tagged_documents):
        labels, descriptions = self.vectorize_tagged_documents(tagged_documents)
        return labels, descriptions

    def vectorize_documents(self, documents):
        documents = [document.split(' ') for document in documents]
        return [self.model.infer_vector(document) for document in documents]



if __name__ == '__main__':
    dataset = my_datasets.ITSupportDatasetBuilder()\
        .with_overall_priority_column()\
        .with_summaries_and_descriptions_combined()\
        .with_pre_processed_descriptions()\
        .build().corpus
    doc2vec_IT = ITSupportDoc2VecImplementation(model_type=Doc2VecModels.DBOW, alpha_change=-0.002)
    training_documents, testing_documents = doc2vec_IT.split_texts(dataset)
    tagged_training_documents = doc2vec_IT.tag_documents(training_documents)
    tagged_testing_documents = doc2vec_IT.tag_documents(testing_documents)
    doc2vec_IT.build_vocabulary(tagged_training_documents)
    doc2vec_IT.train_model(tagged_training_documents, dataset_shuffles=3, epochs=10)
    doc2vec_IT.to_file("doc2vec_model.model", model_interaction.GensimWordEmbeddingModelFileInteraction())
    #doc2vec_IT.generate_vectors()
    #print(doc2vec_IT.X_test)
