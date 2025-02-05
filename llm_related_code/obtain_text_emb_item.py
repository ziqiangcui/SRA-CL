from transformers import AutoTokenizer, AutoModel
import torch
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = './data/Beauty/'

with open(file_path+'responses_item_summary.json', 'r') as file:
    item_txt_dic = json.load(file)
    
item_num = len(set(item_txt_dic.keys()))


tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
item_txt_dic['0'] = "None"
texts = [item_txt_dic[str(i)] for i in range(item_num+1)]
print("len(texts)", len(texts))

batch_size = 64
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

torch.save(all_embeddings, file_path+'beauty_item_semantic_embeddings.pt')
print("inference over", all_embeddings.shape)
