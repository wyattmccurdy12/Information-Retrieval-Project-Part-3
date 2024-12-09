'''
Wyatt McCurdy
Information Retrieval Assignment 3
October 28, 2024
Dr. Behrooz Mansouri

This assignment compares fine-tuned and pre-trained versions of a re-ranking system.
The re-ranking system is made from two models - a bi-encoder which searches over all
documents and a cross-encoder which re-ranks the top 100 results from the 
bi-encoder. 

The query-document pairs will be split into train/test/validation sets
with a 80/10/10 split.

Data in the data directory is in trec format. 
documents: Answers.json
queries: topics_1.json, topics_2.json
qrels:   qrel_1.tsv (qrel_2.tsv is reserved by the instructor)

models used: 
sentence-transformers/all-MiniLM-L6-v2 for the bi-encoder
cross-encoder/ms-marco-MiniLM-L-6-v2
'''

import re
from bs4 import BeautifulSoup
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
import json
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from tqdm import tqdm

class Retriever:
    """
    A class to represent a retriever model which can be either a bi-encoder or a cross-encoder.

    Attributes:
    model_type (str): The type of the model ('bi-encoder' or 'cross-encoder').
    model (SentenceTransformer or CrossEncoder): The loaded model.
    """
    def __init__(self, model_type, model_name):
        """
        Initializes the Retriever with the specified model type and name.

        Parameters:
        model_type (str): The type of the model ('bi-encoder' or 'cross-encoder').
        model_name (str): The name of the model to load.
        """
        self.model_type = model_type
        if model_type == 'bi-encoder':
            self.model = SentenceTransformer(model_name)
        elif model_type == 'cross-encoder':
            self.model = CrossEncoder(model_name)
        else:
            raise ValueError("Invalid model type. Choose 'bi-encoder' or 'cross-encoder'.")

    def encode(self, texts):
        """
        Encodes a list of texts using the bi-encoder model.

        Parameters:
        texts (list of str): The texts to encode.

        Returns:
        list: The encoded texts.
        """
        if self.model_type != 'bi-encoder':
            raise ValueError("Encode method is only available for bi-encoder.")
        return self.model.encode(texts)

    def predict(self, query, documents):
        """
        Predicts scores for query-document pairs using the cross-encoder model.

        Parameters:
        query (str): The query text.
        documents (list of str): The document texts.

        Returns:
        list: The predicted scores.
        """
        if self.model_type != 'cross-encoder':
            raise ValueError("Predict method is only available for cross-encoder.")
        pairs = [[query, doc] for doc in documents]
        return self.model.predict(pairs)

def load_data(queries_path, documents_path):
    """
    Loads the documents and queries from the specified files.

    Parameters:
    queries_path (str): The path to the queries file.
    documents_path (str): The path to the documents file.

    Returns:
    tuple: A tuple containing the documents and queries.
    """
    with open(documents_path, 'r') as f:
        documents = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    return documents, queries

def write_results_to_tsv(results, output_filename, model_type, model_status):
    """
    Writes the results to a TSV file in TREC format.

    Parameters:
    results (dict): The results to write.
    output_filename (str): The name of the output file.
    model_type (str): The type of the model ('simple' or 'reranked').
    model_status (str): The status of the model ('pretrained' or 'finetuned').
    """
    with open(output_filename, 'w') as f:
        for query_id, doc_scores in results.items():
            for rank, (doc_id, score) in enumerate(doc_scores[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {model_type}_{model_status}\n")

def remove_html_tags(text):
    """
    Remove HTML tags from the input text using BeautifulSoup.

    Parameters:
    text (str): The text from which to remove HTML tags.

    Returns:
    str: The text with HTML tags removed.
    """
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"\\", "", text)
    return text

def main():
    """
    Main function to run the ranking and re-ranking system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Ranking and Re-Ranking System')
    parser.add_argument('-r', '--rerank', action='store_true', help='Perform re-ranking')
    parser.add_argument('-q', '--queries', required=True, help='Path to the queries file')
    parser.add_argument('-d', '--documents', required=True, help='Path to the documents file')
    parser.add_argument('-be', '--bi_encoder', required=True, help='Bi-encoder model string')
    parser.add_argument('-ce', '--cross_encoder', required=True, help='Cross-encoder model string')
    parser.add_argument('-ft', '--finetuned', action='store_true', help='Indicate if the model is fine-tuned')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    documents, queries = load_data(args.queries, args.documents)
    print("Data loaded successfully.")

    # Instantiate retrievers
    bi_encoder_retriever = Retriever('bi-encoder', args.bi_encoder)
    cross_encoder_retriever = Retriever('cross-encoder', args.cross_encoder)
    
    # Process and encode queries using bi-encoder
    print("Processing and encoding queries...")
    processed_queries = []
    for query in tqdm(queries, desc="Processing queries"):
        query_id = query['Id']
        title = remove_html_tags(query['Title'])
        body = remove_html_tags(query['Body'])
        tags = ' '.join(query['Tags'])
        merged_query = f"{title} {body} {tags}"
        processed_queries.append((query_id, merged_query))
    
    encoded_queries = bi_encoder_retriever.encode([q[1] for q in processed_queries])
    print("Queries processed and encoded successfully.")
    
    # Process and encode documents using bi-encoder
    print("Processing and encoding documents...")
    processed_documents = {}
    for doc in tqdm(documents, desc="Processing documents"):
        doc_id = doc['Id']
        text = remove_html_tags(doc['Text'])
        processed_documents[doc_id] = text
    
    encoded_documents = bi_encoder_retriever.encode(list(processed_documents.values()))
    print("Documents processed and encoded successfully.")
    
    # Initial ranking using bi-encoder
    print("Performing initial ranking using bi-encoder...")
    initial_rankings = {}  # Dictionary to store initial rankings
    for query_id, query_text in tqdm(processed_queries, desc="Ranking queries"):
        query_embedding = bi_encoder_retriever.encode([query_text])[0]
        # Compute similarity scores and rank documents
        scores = np.dot(encoded_documents, query_embedding)
        ranked_doc_indices = np.argsort(scores)[::-1][:100]
        initial_rankings[query_id] = [(list(processed_documents.keys())[doc_id], scores[doc_id]) for doc_id in ranked_doc_indices]
    print("Initial ranking completed.")
    
    # Extract topic number from queries file name
    topic_number = args.queries.split('_')[-1].split('.')[0]
    
    if args.rerank:
        # Re-ranking using cross-encoder
        print("Performing re-ranking using cross-encoder...")
        re_rankings = {}  # Dictionary to store re-rankings
        for query_id, doc_scores in tqdm(initial_rankings.items(), desc="Re-ranking queries"):
            query_text = next(q[1] for q in processed_queries if q[0] == query_id)
            doc_texts = [processed_documents[doc_id] for doc_id, _ in doc_scores]
            scores = cross_encoder_retriever.predict(query_text, doc_texts)
            re_rankings[query_id] = sorted(zip([doc_id for doc_id, _ in doc_scores], scores), key=lambda x: x[1], reverse=True)
    
        # Determine output filename
        output_filename = f"result_ce{'_ft' if args.finetuned else ''}_{topic_number}.tsv"
        # Write re-ranked results to TSV
        write_results_to_tsv(re_rankings, output_filename, model_type='reranked', model_status='finetuned' if args.finetuned else 'pretrained')
        print(f"Re-ranked results have been computed and saved to {output_filename}.")
    else:
        # Determine output filename
        output_filename = f"result_bi{'_ft' if args.finetuned else ''}_{topic_number}.tsv"
        # Write initial rankings to TSV
        write_results_to_tsv(initial_rankings, output_filename, model_type='simple', model_status='finetuned' if args.finetuned else 'pretrained')
        print(f"Initial rankings have been computed and saved to {output_filename}.")

if __name__ == "__main__":
    main()