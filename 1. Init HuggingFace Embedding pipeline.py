!pip install -qU \
  transformers==4.31.0 \
  sentence-transformers==2.2.2 \
  pinecone-client==2.2.2 \
  datasets==2.14.0 \
  accelerate==0.21.0 \
  einops==0.6.1 \
  langchain==0.0.240 \
  xformers==0.0.20 \
  bitsandbytes==0.41.0

import os
import pinecone
import time
import numpy as np
import pandas as pd
from torch import cuda
import pandas as pd
import json
import random
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

## Initializing the Hugging Face Embedding Pipeline

if torch.cuda.is_available():
    device = torch.device("cuda")
    num_devices = torch.cuda.device_count()
    print(f"Using {num_devices} GPU(s)")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")
embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)