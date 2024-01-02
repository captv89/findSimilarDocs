import tensorflow_hub as hub
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility, connections

# Connect to Milvus server
connections.connect(
  alias="default",
  user='username',
  password='password',
  host='localhost',
  port='19530'
)

# Define PrimaryKey field
primary_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)

# Define field schema for document vectors
document_name_field = FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=200,
  # The default value will be used if this field is left empty during data inserts or upserts.
  # The data type of `default_value` must be the same as that specified in `dtype`.
  default_value="Unknown")
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512)

# Create a collection schema
schema = CollectionSchema(fields=[primary_field, document_name_field, embedding_field], description="Collection for storing document vectors")

# Specify collection parameters
collection_name = "document_embeddings"
shards_num = 2

# Create a collection
collection = Collection(name=collection_name, schema=schema, using='default', shards_num=shards_num)


# Load model
# From modal available locally
model_path = 'universal-sentence-encoder_4'
model = hub.load(model_path)


def embed(input):
    return model(input)

def embed_and_add_to_collection(document_data, collection_name):
    collection = Collection(collection_name)
    embeddings = embed([content for _, content in document_data])
    
    # Convert embeddings to list
    vectors = [embedding.numpy().tolist() for embedding in embeddings]
    
    # Get document names
    doc_names = [doc_name for doc_name, _ in document_data]
    
    data = [
        doc_names,
        vectors
    ]

    status = collection.insert(data)
    print(status)
    

# Load target document
target_document = open('document3.txt').read()

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
embed_and_add_to_collection(document_data, collection_name)


# Build Index
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}

# Get an existing collection.
collection = Collection(collection_name)      
collection.create_index(
  field_name="embedding", 
  index_params=index_params
)

utility.index_building_progress(collection_name)

# Embed target document
target_document_embedding = embed([target_document])[0]

# Convert to numpy array
target_document_list = [target_document_embedding.numpy().tolist()]

# Search for similar documents
top_k = 5
search_param = {
    'nprobe': 16
}

# Search in collection
collection = Collection(collection_name)
collection.load()

# Prepare search parameters
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 16}
}

results = collection.search(data=target_document_list, anns_field="embedding", param=search_params, limit=top_k, expr=None, output_fields=['document_name'], consistency_level="Strong")

hit = results[0]
print(hit)


