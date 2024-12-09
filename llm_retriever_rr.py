import argparse
import json
import re
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import os
import logging

# Set up logging
logging.basicConfig(filename='rr_retriever.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class RRRetriever:

    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    def __init__(self, bi_encoder_model='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
        """
        Initializes the RRRetriever instance and sets up the LLaMA model pipeline and bi-encoder.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.initialize_llama()
        self.bi_encoder = SentenceTransformer(bi_encoder_model).to(self.device)
        self.document_embeddings = {}
        logging.info("Initialized RRRetriever with bi-encoder model: %s", bi_encoder_model)

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
        system_prompt = [
            {"role": "system", "content": "You excel at creating clean, well-ordered lists. In this task, please output WITH NO EXTRA TEXT a re-ordered version of the provided list. The focus in re-ranking should be relevance to the provided query. The query/document information will be provided in the format 'Query: ...query text..., Documents: ...the ten documents to be re-ranked text...'."},
            {"role": "user", "content": f"Query: {query}, Documents: " + "\n\n".join([f"Document {i+1}: {doc['Text']}" for i, doc in enumerate(documents)])}
        ]

        # Generate the re-ranked list using LLaMA
        generated_text = self.model(system_prompt, max_new_tokens=500)[0]['generated_text']
        re_ranked_indices = [int(s) for s in generated_text.split() if s.isdigit()]

        # Re-rank the documents based on the generated indices
        re_ranked_results = [documents[i-1] for i in re_ranked_indices if 1 <= i <= len(documents)]
        logging.info("Re-ranked documents for query: %s", query)
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
    def load_results(filepath):
        """
        Load results from a TSV file.

        Args:
            filepath (str): The path to the TSV file containing results.

        Returns:
            dict: A dictionary containing the results.
        """
        results = {}
        with open(filepath, 'r') as file:
            for line in file:
                qid, _, docid, rank, score, _ = line.strip().split()
                if qid not in results:
                    results[qid] = []
                results[qid].append((docid, int(rank), float(score)))
        logging.info("Loaded results from file: %s", filepath)
        return results

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
    parser.add_argument('-r', '--results_file', default='bi_encoder_results.tsv', help='Path to the results TSV file')
    parser.add_argument('-be', '--bi_encoder', default='sentence-transformers/multi-qa-mpnet-base-dot-v1', help='Bi-encoder model string')
    parser.add_argument('-o', '--output_dir', default='data/outputs/', help='Output directory for TREC-formatted results')
    parser.add_argument('-rn', '--results_name', default='trec_results.txt', help='Name of the results file')
    args = parser.parse_args()

    try:
        print("Loading queries...")
        queries = RRRetriever.load_queries(f"data/inputs/{args.queries_file}")
        print("Loading documents...")
        documents = RRRetriever.load_documents(f"data/inputs/{args.documents_file}")
        print("Loading results...")
        results = RRRetriever.load_results(f"data/outputs/{args.results_file}")
        
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
        output_file = os.path.join(args.output_dir, args.results_name)

        # Write TREC-formatted results to file
        with open(output_file, 'w') as f:
            for query_id, query_text in processed_queries:
                if query_id not in results:
                    continue
                # Retrieve top 10 documents from the results
                top_10_results = sorted(results[query_id], key=lambda x: x[1])[:10]
                top_10_docs = [{'Id': doc_id, 'Text': processed_documents[doc_id], 'Score': score} for doc_id, rank, score in top_10_results]
                
                # Re-rank top 10 documents using LLaMA
                re_ranked_results = retriever.re_rank_documents(query_text, top_10_docs)
                for rank, result in enumerate(re_ranked_results, start=1):
                    f.write(f"{query_id} Q0 {result['Id']} {rank} {result['Score']} STANDARD\n")

        logging.info("Process completed successfully")

    except Exception as e:
        logging.error("An error occurred: %s", e)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()