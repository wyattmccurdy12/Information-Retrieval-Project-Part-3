# Information-Retrieval-Project-Part-3
Project part 3 for IR. Evaluate two separate LLM techniques for document ranking.

## Description of the project
Ask Ubuntu is a community support site for "answers to common questions about installing, troubleshooting and optimising Ubuntu."
The task is simple: given a sample query from the site, provide the top 100 answers, in order, from a pool of answers from the same site.
The data will be a topics json files (queries) and an answers json file (documents). A qrel file will be given for the first topics file, and the second qrel will be reserved for instructor testing. 

### Data
1. `topics_1.json`
2. `topics_2.json`
The above two files are json file containing queries with keys "Id": str (representing int), "Title": str, "Body": str (html formatted), "Tags": str (representing list).
3. `Answers.json`: A json file containing keys "Id": str (representing int), "Text": str (html formatted), "Score": str (representing int - number of AskUbuntu community votes)
4. `qrel_1.tsv`: A trec-formatted qrel file with query ids from the topics files and document ids from the answers file. 

## Description of techniques
The techniques in this project will be dedicated to application of large language models for information retrieval. The specific techniques are described below.

### Query expansion
Query expansion involves broadening a search query by adding related terms or synonyms to the original query. This can help capture more relevant documents that might not have been retrieved by the original query alone. For example, if a user searches for "dog," query expansion might add related terms like "puppy," "canine," or "pet" to the search query, potentially retrieving more relevant results like articles about different dog breeds or tips for caring for puppies.


### Query re-ranking
Query re-ranking involves reassessing the ranking of search results to ensure that the most relevant results are presented first. This can be done using various techniques, such as considering the semantic similarity between the query and the documents, analyzing the context of the query, or incorporating user feedback. For example, if a user searches for "best pizza in New York," query re-ranking might prioritize results from highly-rated restaurants with positive reviews and photos of delicious pizzas, even if they are not ranked highly based on traditional keyword matching.