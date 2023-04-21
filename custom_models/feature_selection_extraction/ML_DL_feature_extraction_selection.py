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

# from project_utilities import preprocessing_functionality, my_datasets
from project_utilities import my_datasets
import preprocessing_functionality

class Doc2VecModels(Enum):
    DBOW = 1
    DM = 2
    COMBINED = 3


class ITSupportDoc2VecImplementation:
    dataset = DataFrame
    tagged_training_documents = DataFrame
    tagged_testing_documents = DataFrame
    model_type = Doc2VecModels
    model = gensim.models.Doc2Vec
    train_descriptions = \
        test_descriptions = \
        train_labels = \
        test_labels = tuple

    def __init__(self, dataset, model_type):
        self.dataset = dataset
        self.model_type = model_type
        self.alpha_change = None
        tqdm.pandas(desc="progress-bar")

    def split_texts(self):
        training_data, testing_data = train_test_split(self.dataset, test_size=0.1, random_state=1000)
        return training_data, testing_data

    def tag_documents(self):
        training_documents, testing_documents = self.split_texts()
        self.tagged_training_documents = training_documents.apply(
            lambda docs: gensim.models.doc2vec.TaggedDocument(
                words=preprocessing_functionality.tokenize_text(docs.Description),
                tags=[docs.Priority]),
            axis=1)
        self.tagged_testing_documents = testing_documents.apply(
            lambda docs: gensim.models.doc2vec.TaggedDocument(
                words=preprocessing_functionality.tokenize_text(docs.Description),
                tags=[docs.Priority]),
            axis=1)

    def create_model(self):
        cores = multiprocessing.cpu_count()
        match self.model_type:
            case Doc2VecModels.DBOW:
                self._create_dbow_model(cores)
            case Doc2VecModels.DM:
                self._create_dm_model(cores)
            case Doc2VecModels.COMBINED:
                self._create_combined_model(cores)
            case _:
                raise TypeError("Must be a Doc2Vec model type (DBOW, DM, COMBINED)")

    def _create_dbow_model(self, cores):
        self.model = gensim.models.Doc2Vec(
            dm=0, vector_size=1000, negative=5, hs=0, min_count=2, sample=0, workers=cores)
        self.alpha_change = 0.0002

    def _create_dm_model(self, cores):
        self.model = gensim.models.Doc2Vec(
            dm=1, dm_mean=1, vector_size=300, window=10, negative=5,
            min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
        self.alpha_change = -0.002

    def _create_combined_model(self, cores):
        dbow_model = gensim.models.Doc2Vec(
            dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample=0, workers=cores)
        dm_model = gensim.models.Doc2Vec(
            dm=1, dm_mean=1, vector_size=300, window=10, negative=5,
            min_count=1, workers=cores, alpha=0.065, min_alpha=0.065)
        self.model = ConcatenatedDoc2Vec([dbow_model, dm_model])

    def build_vocabulary(self):
        vocabulary = [x for x in tqdm(self.tagged_training_documents.values)]
        self.model.build_vocab(vocabulary)

    def train_model(self, dataset_shuffles: int = 1, epochs: int = 1):
        for training_round in range(dataset_shuffles):
            shuffled_training_data = utils.shuffle([x for x in tqdm(self.tagged_training_documents.values)])
            datapoint_quantity = len(self.tagged_training_documents)
            self.model.train(shuffled_training_data, total_examples=datapoint_quantity,
                             epochs=epochs)
            self.model.alpha += self.alpha_change
            self.model.min_alpha = self.model.alpha

    #@numba.jit(forceobj=True)
    def vectorize_tagged_documents(self, tagged_documents):
        sentences = tagged_documents.values
        targets, regressors = zip(*[(doc.tags[0], self.model.infer_vector(doc.words)) for doc in sentences])
        return targets, regressors

    def generate_vectors(self):
        self.train_labels, self.train_descriptions = self.vectorize_tagged_documents(self.tagged_training_documents)
        self.test_labels, self.test_descriptions = self.vectorize_tagged_documents(self.tagged_testing_documents)


if __name__ == '__main__':
    '''dataset = my_datasets.ITSupportDatasetBuilder()\
        .with_overall_priority_column()\
        .with_summaries_and_descriptions_combined()\
        .with_pre_processed_descriptions()\
        .build()
    doc2vec_IT = ITSupportDoc2VecImplementation(dataset=dataset.corpus, model_type=Doc2VecModels.DM)
    #doc2vec_IT.pre_process_texts()
    doc2vec_IT.tag_documents()
    doc2vec_IT.create_model()
    t1 = time.perf_counter()
    doc2vec_IT.build_vocabulary()
    doc2vec_IT.train_model(dataset_shuffles=1, epochs=1)
    print("time: " + str(time.perf_counter() - t1))
    doc2vec_IT.generate_vectors()
    print(doc2vec_IT.tagged_training_documents[50])
    #print(doc2vec_IT.X_test)'''

