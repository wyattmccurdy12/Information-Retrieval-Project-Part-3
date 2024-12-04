import json
import llama
from nltk.corpus import wordnet
import pandas as pd

class QXRetriever:
    def __init__(self):
        self.model = llama.LlamaModel(version="3.2")

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
        for doc in documents:
            if any(word in doc['text'] for word in expanded_query.split()):
                results.append(doc)
        return results

def load_queries(filepath):
    """Load queries from a JSON file."""
    with open(filepath, 'r') as file:
        queries = json.load(file)
    return queries

def load_documents(filepath):
    """Load documents from a JSON file."""
    with open(filepath, 'r') as file:
        documents = json.load(file)
    return documents

def load_qrels(filepath):
    """Load qrels from a TSV file."""
    qrels = pd.read_csv(filepath, sep='\t', header=None, names=['query_id', 'doc_id', 'relevance'])
    return qrels

def main():
    # Load data
    queries = load_queries('data/inputs/topics_1.json')
    documents = load_documents('data/inputs/Answers.json')
    qrels = load_qrels('data/inputs/qrel_1.tsv')
    
    # Initialize QXRetriever
    retriever = QXRetriever()
    
    for query in queries:
        query_id = query['query_id']
        query_text = query['query']
        
        # Expand the query
        expanded_query = retriever.expand_query(query_text)
        print(f"Expanded Query for {query_id}: {expanded_query}")
        
        # Retrieve documents
        results = retriever.retrieve_documents(expanded_query, documents)
        print(f"Retrieved Documents for {query_id}:")
        for result in results:
            print(result)

if __name__ == "__main__":
    main()