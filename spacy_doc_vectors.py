import spacy
import numpy as np

def get_document_vector(doc_path):
    nlp = spacy.load("en_core_web_lg")
    with open(doc_path, 'r', encoding='utf-8') as file:
        text = file.read()
    doc = nlp(text)
    # Return the average of word vectors as a NumPy array
    return np.mean([word.vector for word in doc], axis=0)

def compare_documents(doc_vector1, doc_vector2):
    # Calculate cosine similarity between document vectors
    similarity = np.dot(doc_vector1, doc_vector2) / (np.linalg.norm(doc_vector1) * np.linalg.norm(doc_vector2))
    return similarity
    

# Example usage
document1_path = 'document1.txt'
document2_path = 'document2.txt'

# Get vectors for each document
vector1 = get_document_vector(document1_path)
vector2 = get_document_vector(document2_path)

# Compare the documents
similarity_score = compare_documents(vector1, vector2)

# Print the similarity score
print(f"Similarity between {document1_path} and {document2_path}: {similarity_score}")
