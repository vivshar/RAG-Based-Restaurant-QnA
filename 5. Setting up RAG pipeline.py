from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

text_field = 'review'
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)