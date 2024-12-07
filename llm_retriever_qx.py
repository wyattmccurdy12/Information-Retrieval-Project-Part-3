import argparse
import json
import re
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging

# Set up logging
logging.basicConfig(filename='qx_retriever.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class QXRetriever:

    MODEL_ID = "meta-llama/Llama-3.2-1B"

    def __init__(self, bi_encoder_model='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        """
        Initializes the QXRetriever instance and sets up the LLaMA model pipeline and bi-encoder.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.initialize_llama()
        self.bi_encoder = SentenceTransformer(bi_encoder_model).to(self.device)
        self.document_embeddings = {}
        logging.info("Initialized QXRetriever with bi-encoder model: %s", bi_encoder_model)

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
        logging.info("Initialized LLaMA model pipeline with model ID: %s", self.MODEL_ID)
        return pipe

    def encode_documents(self, documents):
        """
        Encode the documents using the bi-encoder and store their embeddings.

        Args:
            documents (dict): A dictionary where keys are document IDs and values are document texts.
        """
        for doc_id, text in tqdm(documents.items(), desc="Encoding documents"):
            text_tensor = self.bi_encoder.encode(text, convert_to_tensor=True, device=self.device)
            self.document_embeddings[doc_id] = text_tensor
        logging.info("Encoded %d documents", len(documents))

    def retrieve_documents(self, expanded_query, documents, top_k=100):
        """
        Retrieve documents that match the expanded query using bi-encoder.

        Args:
            expanded_query (str): The query string with expanded terms.
            documents (dict): A dictionary where keys are document IDs and values are document texts.
            top_k (int): The number of top documents to retrieve.

        Returns:
            list: A list of dictionaries, each containing 'Id', 'Text', and 'Score' of documents that match the expanded query.
        """
        query_embedding = self.bi_encoder.encode(expanded_query, convert_to_tensor=True, device=self.device)
        results = []
        for doc_id, doc_embedding in self.document_embeddings.items():
            score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
            results.append({'Id': doc_id, 'Text': documents[doc_id], 'Score': score})
        results = sorted(results, key=lambda x: x['Score'], reverse=True)[:top_k]
        logging.info("Retrieved top %d documents for query: %s", top_k, expanded_query)
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
            logging.error("ValueError during text generation: %s", e)
            return text_input  # Return the original text if a ValueError occurs
        logging.info("Generated expanded text for input: %s", text_input)
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
        logging.info("Expanded query title for topic ID: %s", topic['Id'])
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
        logging.info("Loaded queries from file: %s", filepath)
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
        logging.info("Loaded documents from file: %s", filepath)
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
        logging.info("Loaded qrels from file: %s", filepath)
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
    

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-q', '--queries_file', default='topics_1.json', help='Path to the queries JSON file')
    parser.add_argument('-d', '--documents_file', default='Answers.json', help='Path to the documents JSON file')
    parser.add_argument('-be', '--bi_encoder', default='sentence-transformers/multi-qa-mpnet-base-dot-v1', help='Bi-encoder model string')
    parser.add_argument('-o', '--output_dir', default='data/outputs/', help='Output directory for TREC-formatted results')
    parser.add_argument('-r', '--results_name', default='trec_results.txt', help='Name of the results file')
    parser.add_argument('--write-expanded', action='store_true', default=True, help='Write expanded queries to a JSON file')
    args = parser.parse_args()

    try:
        # Load data
        print("Loading queries...")
        queries = QXRetriever.load_queries(f"data/inputs/{args.queries_file}")
        print("Loading documents...")
        documents = QXRetriever.load_documents(f"data/inputs/{args.documents_file}")
        print("Loading qrels...")
        qrels = QXRetriever.load_qrels('data/inputs/qrel_1.tsv')
        
        # Initialize QXRetriever
        print("Initializing QXRetriever...")
        retriever = QXRetriever(bi_encoder_model=args.bi_encoder)
        
        # Process and encode documents
        print("Processing and encoding documents...")
        processed_documents = {}
        for doc in tqdm(documents, desc="Processing documents"):
            doc_id = doc['Id']
            text = QXRetriever.remove_html_tags(doc['Text'])
            processed_documents[doc_id] = text
        
        # Encode documents
        retriever.encode_documents(processed_documents)
        
        # Process queries
        print("Processing queries...")
        processed_queries = []
        expanded_queries = []
        expanded_queries_file = os.path.join(args.output_dir, 'expanded_queries.json')

        if not args.write_expanded and os.path.exists(expanded_queries_file):
            print(f"Loading expanded queries from {expanded_queries_file}...")
            with open(expanded_queries_file, 'r') as f:
                expanded_queries = json.load(f)
            processed_queries = [(query['Id'], query['ExpandedTitle']) for query in expanded_queries]
        else:
            for query in tqdm(queries, desc="Processing queries"):
                query_id = query['Id']
                title = QXRetriever.remove_html_tags(query['Title'])
                body = QXRetriever.remove_html_tags(query['Body'])
                tags = ' '.join(query['Tags'])
                merged_query = f"{title} {body} {tags}"
                expanded_query = retriever.expand_query_title({'Title': merged_query})
                processed_queries.append((query_id, expanded_query['Title']))
                expanded_queries.append({'Id': query_id, 'ExpandedTitle': expanded_query['Title']})
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, args.results_name)

        # Write TREC-formatted results to file
        with open(output_file, 'w') as f:
            for query_id, query_text in processed_queries:
                # Expand the query
                expanded_query = retriever.expand_query(query_text)
                print(f"Expanded Query for {query_id}: {expanded_query}")
                
                # Retrieve documents
                print(f"Retrieving documents for query {query_id}...")
                results = retriever.retrieve_documents(expanded_query, processed_documents)
                print(f"Retrieved Documents for {query_id}:")
                for rank, result in enumerate(results, start=1):
                    f.write(f"{query_id} Q0 {result['Id']} {rank} {result['Score']} STANDARD\n")
        
        # Write expanded queries to a JSON file if the flag is set
        if args.write_expanded:
            with open(expanded_queries_file, 'w') as f:
                json.dump(expanded_queries, f, indent=4)
            print(f"Expanded queries written to {expanded_queries_file}")

        logging.info("Process completed successfully")

    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()