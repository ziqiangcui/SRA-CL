from transformers import AutoTokenizer, AutoModel
import torch
import json
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "./data/Beauty/"

with open(file_path+'responses_user_preference.json', 'r') as file:
    user_txt_dic = json.load(file)
    
user_num = len(set(user_txt_dic.keys()))
print("seq_num: ", user_num)

keys_to_int = {key: i for i, key in enumerate(user_txt_dic.keys())}

with open(file_path+'keys_to_int.pkl', 'wb') as f:
    pickle.dump(keys_to_int, f)

user_txt_dic_new = {}
for key in user_txt_dic.keys():
    int_key = keys_to_int[key]
    user_txt_dic_new[int_key] = user_txt_dic[key]

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

texts = [user_txt_dic_new[i] for i in range(user_num)]
print("len(texts list)", len(texts))

batch_size = 256
all_embeddings = []
model.to(device)

for i in range(0, len(texts), batch_size):
    print(i)
    batch_texts = texts[i:i + batch_size]
    
    inputs_simcse = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    
    inputs_simcse = {key: value.to(device) for key, value in inputs_simcse.items()}
    
    with torch.no_grad():
        embeddings_simcse = model(**inputs_simcse, output_hidden_states=True, return_dict=True).pooler_output
    
    all_embeddings.append(embeddings_simcse.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)

torch.save(all_embeddings, file_path+'beauty_user_semantic_embeddings.pt')
print("inference over", all_embeddings.shape)
