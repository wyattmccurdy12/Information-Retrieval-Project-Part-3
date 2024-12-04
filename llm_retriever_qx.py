import json
import re
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet
from transformers import pipeline
import torch

class QXRetriever:
    def __init__(self):
        self.model = self.initialize_llama()

    def initialize_llama(self):
        """Initialize the LLaMA model pipeline."""
        model_id = "meta-llama/Llama-3.2-1B"
        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            max_length=200  # Set max_length instead of max_new_tokens
        )
        return pipe

    def expand_query(self, query):
        """Expand the query using synonyms from WordNet."""
        expanded_query = set(query.split())
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_query.add(lemma.name())
        return ' '.join(expanded_query)

    def retrieve_documents(self, expanded_query, documents):
        """Retrieve documents that match the expanded query."""
        results = []
        for doc_id, text in documents.items():
            if any(word in text for word in expanded_query.split()):
                results.append({'Id': doc_id, 'Text': text})
        return results

    def gen_with_llama(self, text_input):
        """Expand text input using LLaMA 3.2 1B."""
        try:
            gen_text = self.model(text_input, max_length=200)[0]['generated_text']
        except ValueError as e:
            print(f"ValueError: {e}")
            return text_input  # Return the original text if a ValueError occurs
        return gen_text

    def expand_query_title(self, topic):
        """Expand the title field of the topics JSON and return a copy of the JSON exactly identical to the original but with an expanded title."""
        title = topic['Title']
        text_input = f"{title}"
        expanded_title = self.gen_with_llama(text_input)
        topic['Title'] = expanded_title
        return topic

    @staticmethod
    def load_queries(filepath):
        """Load queries from a JSON file."""
        with open(filepath, 'r') as file:
            queries = json.load(file)
        return queries

    @staticmethod
    def load_documents(filepath):
        """Load documents from a JSON file."""
        with open(filepath, 'r') as file:
            documents = json.load(file)
        return documents

    @staticmethod
    def load_qrels(filepath):
        """Load qrels from a TSV file."""
        qrels = pd.read_csv(filepath, sep='\t', header=None, names=['query_id', 'doc_id', 'relevance'])
        return qrels

    @staticmethod
    def remove_html_tags(text):
        """Remove HTML tags from a string."""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)