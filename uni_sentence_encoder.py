import tensorflow as tf
import tensorflow_hub as hub


# From online hub
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# To download the model url + ?tf-hub-format=compressed

# model = hub.load(module_url)
# print("module %s loaded" % module_url)

# From modal available locally
model_path = 'universal-sentence-encoder_4'
model = hub.load(model_path)


def embed(input):
    return model(input)

# Load target document
target_document = open('document1.txt').read()

# Load collection of word documents along with their names
document_data = [
    ('document1.txt', open('document1.txt').read()),
    ('document2.txt', open('document2.txt').read()),
    ('document3.txt', open('document3.txt').read()),
    ('document4.txt', open('document4.txt').read()),
    ('document5.txt', open('document5.txt').read()),
]

# Represent documents as vectors
# Embed documents
document_embeddings = embed([doc_content for _, doc_content in document_data])

# Embed target document
target_document_embedding = embed([target_document])[0]

print(target_document_embedding)

# Calculate cosine similarity between target document and each document in the collection
cos_scores = []
for i, (doc_name, doc_embedding) in enumerate(zip(document_data, document_embeddings)):
    cos_similarity = tf.keras.losses.cosine_similarity(target_document_embedding, doc_embedding)
    cos_scores.append((doc_name[0], cos_similarity.numpy()))

# Print document name and percentage similarity
for doc_name, similarity_score in cos_scores:
    percentage_similarity = abs(similarity_score) * 100
    print(f"Document: {doc_name}, Similarity: {percentage_similarity:.2f}%")
