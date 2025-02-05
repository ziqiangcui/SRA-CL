## Introduction
This is the code for our paper: **Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation**.

<!-- ## Environment Dependencies
You can refer to `requirements.txt` for the experimental environment we set to use. -->

## Running SRA-CL

Follow the steps below to run the Semantic Retrieval Augmented Contrastive Learning (SRA-CL):

### 1. Build Datasets and Generate Prompts

Navigate to the `build_datasets&prompts` directory and run the appropriate Jupyter notebook for your dataset:

```bash
cd build_datasets&prompts
# Replace <dataset> with the name of your dataset
jupyter notebook <dataset>.ipynb 
```

### 2. Use LLM API to generate text descriptions and then obtain semantic embeddings.
Navigate to the get_llmResponse&semanticEmb directory and run the following scripts to generate text descriptions and obtain semantic embeddings:
```bash
cd get_llmResponse&semanticEmb

# Obtain LLM's description for items
python obtain_response_item.py

# Obtain LLM's description for users
python obtain_response_user.py

# Transform items' textual descriptions into embeddings
python obtain_text_emb_item.py

# Transform users' textual descriptions into embeddings
python obtain_text_emb_user.py
```

### 3. Train recommender models.
```bash
cd recommender_code
sh train.sh
```