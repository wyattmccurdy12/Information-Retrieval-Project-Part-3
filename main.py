import argparse
from llm_retriever_qx import QXRetriever
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-be', '--bi_encoder', default='sentence-transformers/multi-qa-mpnet-base-dot-v1', help='Bi-encoder model string')
    args = parser.parse_args()

    # Load data
    queries = QXRetriever.load_queries('data/inputs/topics_1.json')
    documents = QXRetriever.load_documents('data/inputs/Answers.json')
    qrels = QXRetriever.load_qrels('data/inputs/qrel_1.tsv')
    
    # Initialize QXRetriever
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
    processed_queries = []
    for query in tqdm(queries, desc="Processing queries"):
        query_id = query['Id']
        title = QXRetriever.remove_html_tags(query['Title'])
        body = QXRetriever.remove_html_tags(query['Body'])
        tags = ' '.join(query['Tags'])
        merged_query = f"{title} {body} {tags}"
        expanded_query = retriever.expand_query_title({'Title': merged_query})
        processed_queries.append((query_id, expanded_query['Title']))
    
    for query_id, query_text in processed_queries:
        # Expand the query
        expanded_query = retriever.expand_query(query_text)
        print(f"Expanded Query for {query_id}: {expanded_query}")
        
        # Retrieve documents
        results = retriever.retrieve_documents(expanded_query, processed_documents)
        print(f"Retrieved Documents for {query_id}:")
        for result in results:
            print(result)

if __name__ == "__main__":
    main()