import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub


# From modal available locally
model_path = 'universal-sentence-encoder_4'
model = hub.load(model_path)

st.title('Document Similarity Checker')
st.subheader('This is a simple app used to check the similarity of the uploaded document with the collection of documents')

def embed(input):
    return model(input)


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


#Divider
st.divider()

st.text('Please upload a document to check for similar documents')

# Load target document
target_document = ''
uploaded_file_1 = st.file_uploader("Choose a file", type=['txt'])
if uploaded_file_1 is not None:
    uploaded_file_1.seek(0)
    target_document = uploaded_file_1.read()
    

if st.button('Check Similarity'):
    # Calculate
    with st.spinner('Calculating Similarity...'):
    
        # Represent documents as vectors
        # Embed target document
        target_document_embedding = embed([target_document])[0]

        # Calculate cosine similarity between target document and each document in the collection
        cos_scores = []
        for i, (doc_name, doc_embedding) in enumerate(zip(document_data, document_embeddings)):
            cos_similarity = tf.keras.losses.cosine_similarity(target_document_embedding, doc_embedding)
            cos_scores.append((doc_name[0], abs(cos_similarity.numpy()) * 100))

        # Display results using gauge lines
        for doc_name, similarity_score in cos_scores:
            # convert similarity score to between 0 and 1 for progress bar
            similarity_score_progress = round(similarity_score / 100, 1)
            st.progress(similarity_score_progress, f"Document: {doc_name} | Similarity: {similarity_score:.2f} %")

        st.balloons()