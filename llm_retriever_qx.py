import json
import re
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet
from transformers import pipeline
import torch

class QXRetriever:
    """
    QXRetriever is a class designed to initialize and use the LLaMA model for text generation,
    expand queries using synonyms from WordNet, and retrieve documents that match the expanded query.

    Attributes:
        MODEL_ID (str): The identifier for the LLaMA model to be used.
    """

    MODEL_ID = "meta-llama/Llama-3.2-1B"
    SYSTEM_PROMPT = "You are a linux researcher with an intimate knowlege of linux systems.\
        You are an expert at phrasing precise questions and creating specificly worded answers."

    def __init__(self):
        """
        Initializes the QXRetriever instance and sets up the LLaMA model pipeline.
        """
        self.model = self.initialize_llama()

    def initialize_llama(self):
        """
        Initialize the LLaMA model pipeline.

        Returns:
            pipeline: The initialized LLaMA model pipeline.
        """
        pipe = pipeline(
            "text-generation", 
            model=self.MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            max_length=200  # Set max_length instead of max_new_tokens
        )
        return pipe

    def expand_query(self, query):
        """
        Expand the query using synonyms from WordNet.

        Args:
            query (str): The input query string.

        Returns:
            str: The expanded query string with synonyms.
        """
        expanded_query = set(query.split())
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_query.add(lemma.name())
        return ' '.join(expanded_query)
import json
import re
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

class QXRetriever:
    """
    QXRetriever is a class designed to initialize and use the LLaMA model for text generation,
    expand queries using synonyms from WordNet, and retrieve documents that match the expanded query.

    Attributes:
        MODEL_ID (str): The identifier for the LLaMA model to be used.
    """

    MODEL_ID = "meta-llama/Llama-3.2-1B"

    def __init__(self, bi_encoder_model='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        """
        Initializes the QXRetriever instance and sets up the LLaMA model pipeline and bi-encoder.
        """
        self.model = self.initialize_llama()
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.document_embeddings = {}

    def initialize_llama(self):
        """
        Initialize the LLaMA model pipeline.

        Returns:
            pipeline: The initialized LLaMA model pipeline.
        """
        pipe = pipeline(
            "text-generation", 
            model=self.MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            max_length=200  # Set max_length instead of max_new_tokens
        )
        return pipe

    def expand_query(self, query):
        """
        Expand the query using synonyms from WordNet.

        Args:
            query (str): The input query string.

        Returns:
            str: The expanded query string with synonyms.
        """
        expanded_query = set(query.split())
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_query.add(lemma.name())
        return ' '.join(expanded_query)

    def encode_documents(self, documents):
        """
        Encode the documents using the bi-encoder and store their embeddings.

        Args:
            documents (dict): A dictionary where keys are document IDs and values are document texts.
        """
        for doc_id, text in documents.items():
            self.document_embeddings[doc_id] = self.bi_encoder.encode(text, convert_to_tensor=True)

    def retrieve_documents(self, expanded_query, documents):
        """
        Retrieve documents that match the expanded query using bi-encoder.

        Args:
            expanded_query (str): The query string with expanded terms.
            documents (dict): A dictionary where keys are document IDs and values are document texts.

        Returns:
            list: A list of dictionaries, each containing 'Id' and 'Text' of documents that match the expanded query.
        """
        query_embedding = self.bi_encoder.encode(expanded_query, convert_to_tensor=True)
        results = []
        for doc_id, doc_embedding in self.document_embeddings.items():
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            results.append({'Id': doc_id, 'Text': documents[doc_id], 'Score': score})
        results = sorted(results, key=lambda x: x['Score'], reverse=True)
        return results

    def gen_with_llama(self, text_input):
        """
        Expand text input using LLaMA 3.2 1B.

        This method takes a text input and generates expanded text using the LLaMA model.
        If a ValueError occurs during text generation, the original text input is returned.

        Args:
            text_input (str): The input text to be expanded.

        Returns:
            str: The expanded text generated by the LLaMA model, or the original text input if an error occurs.
        """
        try:
            gen_text = self.model(text_input, max_length=200)[0]['generated_text']
        except ValueError as e:
            print(f"ValueError: {e}")
            return text_input  # Return the original text if a ValueError occurs
        return gen_text

    def expand_query_title(self, topic):
        """
        Expand the title field of the topics JSON and return a copy of the JSON exactly identical to the original but with an expanded title.

        Args:
            topic (dict): The JSON formatted dictionary containing queries with Id, Title, and Body fields.

        Returns:
            dict: The updated topic with an expanded title.
        """
        title = topic['Title']
        text_input = f"{title}"
        expanded_title = self.gen_with_llama(text_input)
        topic['Title'] = expanded_title
        return topic

    @staticmethod
    def load_queries(filepath):
        """
        Load queries from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing queries.

        Returns:
            list: A list of queries loaded from the JSON file.
        """
        with open(filepath, 'r') as file:
            queries = json.load(file)
        return queries

    @staticmethod
    def load_documents(filepath):
        """
        Load documents from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing documents.

        Returns:
            dict: A dictionary of documents loaded from the JSON file.
        """
        with open(filepath, 'r') as file:
            documents = json.load(file)
        return documents

    @staticmethod
    def load_qrels(filepath):
        """
        Load qrels from a TSV file.

        Args:
            filepath (str): The path to the TSV file containing qrels.

        Returns:
            DataFrame: A pandas DataFrame containing the qrels.
        """
        qrels = pd.read_csv(filepath, sep='\t', header=None, names=['query_id', 'doc_id', 'relevance'])
        return qrels

    @staticmethod
    def remove_html_tags(text):
        """
        Remove HTML tags from a string.

        Args:
            text (str): The input string containing HTML tags.

        Returns:
            str: The cleaned string with HTML tags removed.
        """
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    def gen_with_llama(self, text_input):
        """
        Expand text input using LLaMA 3.2 1B.

        This method takes a text input and generates expanded text using the LLaMA model.
        If a ValueError occurs during text generation, the original text input is returned.

        Args:
            text_input (str): The input text to be expanded.

        Returns:
            str: The expanded text generated by the LLaMA model, or the original text input if an error occurs.
        """
        try:
            gen_text = self.model(text_input, max_length=200)[0]['generated_text']
        except ValueError as e:
            print(f"ValueError: {e}")
            return text_input  # Return the original text if a ValueError occurs
        return gen_text

    def expand_query_title(self, topic):
        """
        Expand the title field of the topics JSON and return a copy of the JSON exactly identical to the original but with an expanded title.

        Args:
            topic (dict): The JSON formatted dictionary containing queries with Id, Title, and Body fields.

        Returns:
            dict: The updated topic with an expanded title.
        """
        title = topic['Title']
        text_input = f"{title}"
        expanded_title = self.gen_with_llama(text_input)
        topic['Title'] = expanded_title
        return topic

    @staticmethod
    def load_queries(filepath):
        """
        Load queries from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing queries.

        Returns:
            list: A list of queries loaded from the JSON file.
        """
        with open(filepath, 'r') as file:
            queries = json.load(file)
        return queries

    @staticmethod
    def load_documents(filepath):
        """
        Load documents from a JSON file.

        Args:
            filepath (str): The path to the JSON file containing documents.

        Returns:
            dict: A dictionary of documents loaded from the JSON file.
        """
        with open(filepath, 'r') as file:
            documents = json.load(file)
        return documents

    @staticmethod
    def load_qrels(filepath):
        """
        Load qrels from a TSV file.

        Args:
            filepath (str): The path to the TSV file containing qrels.

        Returns:
            DataFrame: A pandas DataFrame containing the qrels.
        """
        qrels = pd.read_csv(filepath, sep='\t', header=None, names=['query_id', 'doc_id', 'relevance'])
        return qrels

    @staticmethod
    def remove_html_tags(text):
        """
        Remove HTML tags from a string.

        Args:
            text (str): The input string containing HTML tags.

        Returns:
            str: The cleaned string with HTML tags removed.
        """
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)