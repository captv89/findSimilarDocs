import tensorflow as tf
import tensorflow_hub as hub
from pymilvus import Milvus, DataType

# Connect to Milvus server
milvus = Milvus(host='localhost', port='19530')

# Create a collection (analogous to a table in a traditional database)
collection_name = 'document_embeddings'
milvus.create_collection(collection_name, {'dimension': 512}, data_type=DataType.FLOAT_VECTOR)



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

def embed_and_add_to_collection(input, collection_name):
    embeddings = embed(input)
    vectors = [embedding.numpy().tolist() for embedding in embeddings]
    milvus.insert(collection_name=collection_name, records=vectors)

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

# Embed each document in the collection
embed_and_add_to_collection([document for _, document in document_data], collection_name)

# Embed target document
target_document_embedding = embed([target_document])[0]

# print(type(target_document_embedding))
# <class 'tensorflow.python.framework.ops.EagerTensor'>

# Calculate cosine similarity between target document and each document in the collection
cos_scores = []
# for i, (doc_name, doc_embedding) in enumerate(zip(document_data, document_embeddings)):
#     cos_similarity = tf.keras.losses.cosine_similarity(target_document_embedding, doc_embedding)
#     cos_scores.append((doc_name[0], cos_similarity.numpy()))

# Check similarity between target document and each document in the collection
target_document_embedding_numpy = target_document_embedding.numpy()

# Search for similar documents
top_k = 5
search_param = {
    'nprobe': 16
}
status, results = milvus.search(collection_name=collection_name, query_records=[target_document_embedding_numpy], top_k=top_k, params=search_param)

# Print results
for result in results[0]:
    print(result.id, result.distance)


# # Print document name and percentage similarity
# for doc_name, similarity_score in cos_scores:
#     percentage_similarity = abs(similarity_score) * 100
#     print(f"Document: {doc_name}, Similarity: {percentage_similarity:.2f}%")
