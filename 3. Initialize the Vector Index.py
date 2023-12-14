## Initialize the Vector Index
from datasets import Dataset
# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or '<Insert PINECONE_API_KEY>',
    environment=os.environ.get('PINECONE_ENVIRONMENT') or 'gcp-starter'
)

index_name = 'llama-2-rag'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
pinecone.describe_index(index_name)
index = pinecone.Index(index_name)
index.describe_index_stats()

new_review_df = pd.read_csv('sampled_dataset.csv')
# data = Dataset.from_pandas(new_review_df)
data = Dataset.from_pandas(new_review_df)
data
data = data.to_pandas()

batch_size = 256

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['combined_id']}" for i, x in batch.iterrows()]
    texts = [x['review'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {
         'review': x['review'],
         'city': x['city'],
         'state': x['state'],
         'name': x['name'],
        } for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds,metadata))