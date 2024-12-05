import argparse
import json
import re
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import os

class RRRetriever:

    MODEL_ID = "meta-llama/Llama-3.2-1B"

    def __init__(self, bi_encoder_model='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        """
        Initializes the RRRetriever instance and sets up the LLaMA model pipeline and bi-encoder.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.initialize_llama()
        self.bi_encoder = SentenceTransformer(bi_encoder_model).to(self.device)
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

    def encode_documents(self, documents):
        """
        Encode the documents using the bi-encoder and store their embeddings.

        Args:
            documents (dict): A dictionary where keys are document IDs and values are document texts.
        """
        for doc_id, text in tqdm(documents.items(), desc="Encoding documents"):
            text_tensor = self.bi_encoder.encode(text, convert_to_tensor=True, device=self.device)
            self.document_embeddings[doc_id] = text_tensor

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
        return results

    def re_rank_documents(self, query, documents):
        """
        Re-rank the retrieved documents using the LLaMA model.

        Args:
            query (str): The query string.
            documents (list): A list of dictionaries, each containing 'Id', 'Text', and 'Score' of documents.

        Returns:
            list: A re-ranked list of dictionaries, each containing 'Id', 'Text', and 'Score' of documents.
        """
        # Prepare the system prompt
        system_prompt = f"Re-rank the following documents based on their relevance to the query: '{query}'.\n\n"
        for i, doc in enumerate(documents):
            system_prompt += f"Document {i+1}: {doc['Text']}\n\n"
        system_prompt += "Provide the re-ranked list of document numbers based on their relevance to the query."

        # Generate the re-ranked list using LLaMA
        generated_text = self.model(system_prompt, max_length=500)[0]['generated_text']
        re_ranked_indices = [int(s) for s in generated_text.split() if s.isdigit()]

        # Re-rank the documents based on the generated indices
        re_ranked_results = [documents[i-1] for i in re_ranked_indices if 1 <= i <= len(documents)]
        return re_ranked_results

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
    parser.add_argument('-be', '--bi_encoder', default='sentence-transformers/multi-qa-mpnet-base-dot-v1', help='Bi-encoder model string')
    parser.add_argument('-o', '--output_dir', default='data/outputs/', help='Output directory for TREC-formatted results')
    args = parser.parse_args()

    # Load data
    print("Loading queries...")
    queries = RRRetriever.load_queries('data/inputs/topics_1.json')
    print("Loading documents...")
    documents = RRRetriever.load_documents('data/inputs/Answers.json')
    
    # Initialize RRRetriever
    print("Initializing RRRetriever...")
    retriever = RRRetriever(bi_encoder_model=args.bi_encoder)
    
    # Process and encode documents
    print("Processing and encoding documents...")
    processed_documents = {}
    for doc in tqdm(documents, desc="Processing documents"):
        doc_id = doc['Id']
        text = RRRetriever.remove_html_tags(doc['Text'])
        processed_documents[doc_id] = text
    
    # Encode documents
    retriever.encode_documents(processed_documents)
    
    # Process queries
    print("Processing queries...")
    processed_queries = []
    for query in tqdm(queries, desc="Processing queries"):
        query_id = query['Id']
        title = RRRetriever.remove_html_tags(query['Title'])
        body = RRRetriever.remove_html_tags(query['Body'])
        tags = ' '.join(query['Tags'])
        merged_query = f"{title} {body} {tags}"
        processed_queries.append((query_id, merged_query))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'trec_results.txt')

    # Write TREC-formatted results to file
    with open(output_file, 'w') as f:
        for query_id, query_text in processed_queries:
            # Retrieve top 100 documents
            print(f"Retrieving documents for query {query_id}...")
            results = retriever.retrieve_documents(query_text, processed_documents, top_k=100)
            print(f"Retrieved Documents for {query_id}:")
            
            # Re-rank top 10 documents using LLaMA
            top_10_results = results[:10]
            re_ranked_results = retriever.re_rank_documents(query_text, top_10_results)
            for rank, result in enumerate(re_ranked_results, start=1):
                f.write(f"{query_id} Q0 {result['Id']} {rank} {result['Score']} STANDARD\n")

if __name__ == "__main__":
    main()