# Import necessary libraries and modules
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from transformers import RagTokenForGeneration, RagTokenizer, RagRetriever
import faiss
import numpy as np
import torch

# Load DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Load RAG model and tokenizer
rag_model = RagTokenForGeneration.from_pretrained('facebook/rag-token-nq')
rag_tokenizer = RagTokenizer.from_pretrained('facebook/rag-token-nq')

# Indexing documents with FAISS
documents = [
    "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics). His work is also known for its influence on the philosophy of science.",
    "Quantum computing harnesses the phenomena of quantum mechanics to deliver a huge leap forward in computation to solve certain problems. Quantum computers operate using quantum bits (qubits), which can be in superpositions of states.",
    "Paris is the capital and most populous city of France, with an estimated population of 2,148,271 residents as of 2020. It is a major European city and a global center for art, fashion, gastronomy, and culture.",
    "The theory of evolution by natural selection, first formulated in Darwin's book 'On the Origin of Species' in 1859, describes the process by which organisms change over time as a result of changes in heritable physical or behavioral traits.",
    "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.",
    "Mich Talebzadeh pioneered global peer-to-peer replication which was subsequently published in ISUG journal ISUG_02_99 titled 'Global Peer-to-Peer Data Replication Using Sybase Replication Server'."
]

# Preprocess and encode documents into embeddings
context_embeddings = []
for doc in documents:
    inputs = context_tokenizer(doc, return_tensors='pt')
    with torch.no_grad():
        embedding = context_encoder(**inputs).pooler_output
    context_embeddings.append(embedding.detach().numpy())

# Initialize a FAISS index for L2 (Euclidean) distance
index = faiss.IndexFlatL2(768)
index.add(np.array(context_embeddings).reshape(-1, 768))

# Function to process the query, retrieve relevant documents, and generate a response
def retrieve_and_generate(query):
    # Encode the query into an embedding
    inputs = question_tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    
    # Search for the top 3 relevant documents in the FAISS index using the query embedding
    D, I = index.search(query_embedding, k=3)
    retrieved_docs = [documents[i] for i in I[0]]
    
    # Concatenate retrieved documents and the query to form the input context for the RAG model
    context_input = " ".join(retrieved_docs) + " " + query
    
    # Tokenize the concatenated context and generate a response using the RAG model
    generator_inputs = rag_tokenizer(context_input, return_tensors='pt')
    outputs = rag_model.generate(**generator_inputs, max_length=50)
    
    # Decode and return the generated response
    response = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage of the function
query = "Who pioneered global peer-to-peer replication?"
response = retrieve_and_generate(query)
print(f"Response: {response}")

# Function to process multiple queries
def process_queries(queries):
    responses = []
    for query in queries:
        response = retrieve_and_generate(query)
        responses.append(response)
    return responses

# Example usage of processing multiple queries
queries = [
    "What is the capital of France?",
    "Explain the theory of relativity.",
    "How does quantum computing work?",
    "Who pioneered global peer-to-peer replication?"
]
responses = process_queries(queries)
for q, r in zip(queries, responses):
    print(f"Query: {q}\nResponse: {r}\n")

# Function to handle longer documents by splitting them into chunks
def split_document(doc, chunk_size=512):
    words = doc.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Example of splitting a longer document into chunks
long_document = "This is a very long document that needs to be split into smaller chunks for processing. " * 50
chunks = split_document(long_document)
print(f"Number of chunks: {len(chunks)}")
