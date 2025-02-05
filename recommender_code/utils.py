import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import ChainMap
import math 
from torch.utils.data import DataLoader, TensorDataset

def calculate_valid_score(valid_result, valid_metric=None):
    r"""return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result["Recall@10"]

def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')


class SequenceDataset(Dataset):
    def __init__(self, args, uid_list, item_list, target_list, item_list_length, seq_unique_key_list, maxlen):
        self.item_list = item_list # list of lists
        self.target_list = target_list # list
        self.item_list_length = item_list_length # list
        self.uid_list = uid_list
        self.maxlen = maxlen # int
        self.mlm_probability = 0.2
        self.neighbor_matrix = args.sorted_indices_numpy
        self.user_neighbor_matrix = args.user_sorted_indices_numpy
        self.seq_unique_key_list = seq_unique_key_list
        self.seq_int_to_keys = args.seq_int_to_keys

    def __len__(self):
        return len(self.item_list)
    
    def recall_similar_seqs(self, neighbor_matrix, seq_unique_id):
        neighbors = neighbor_matrix[seq_unique_id]
  
        neighbor_seq_list = []
        for neighbor_seq_id in neighbors:
            neighbor_seq_key = self.seq_int_to_keys[neighbor_seq_id]
            neighbor_seq_key = neighbor_seq_key.split(":")
            neighbor_seq_key = list(map(int, neighbor_seq_key))
            neighbor_seq = neighbor_seq_key[1:]
            neighbor_seq = self.padding_and_truncation(neighbor_seq)
            neighbor_seq_list.append(neighbor_seq)
        
        return np.array(neighbor_seq_list)
        
    
    def replace_input_ids(self, input_ids, p, neighbor_matrix):
        mask_indices = []
        replaced_input_ids = []

        for token in input_ids:
            if token == 0:
                mask_indices.append(0)  
                replaced_input_ids.append(token)
            else:
                if random.random() < p:
                    mask_indices.append(1)
                
                    neighbors = neighbor_matrix[token]
                    if len(neighbors) > 0:
                        replaced_token = random.choice(neighbors)
                    else:
                        print("====debug no neighbors=========")
                        replaced_token = token 
                    replaced_input_ids.append(replaced_token)
                else:
                    mask_indices.append(0)
                    replaced_input_ids.append(token)

        return replaced_input_ids

    def padding_and_truncation(self, seq):
        length = len(seq)
        if length < self.maxlen:
            padded_seq = np.zeros(self.maxlen, dtype=np.int32)
            padded_seq[-length:] = seq[:]
        else:
            padded_seq = seq
        
        padded_seq = padded_seq[-self.maxlen:]
        return padded_seq
        
    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.item_list[idx]
        target = self.target_list[idx]
        length = self.item_list_length[idx]
        seq_unique_id = self.seq_unique_key_list[idx]
       
        padded_seq = self.padding_and_truncation(seq)
        
        similar_seqs = self.recall_similar_seqs(self.user_neighbor_matrix, seq_unique_id)
        aug_seq1 = self.replace_input_ids(padded_seq, self.mlm_probability, self.neighbor_matrix)
        aug_seq2 = self.replace_input_ids(padded_seq, self.mlm_probability, self.neighbor_matrix)
        

        return torch.tensor(seq_unique_id, dtype=torch.long), torch.tensor(padded_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long), \
               torch.tensor(length, dtype=torch.long), torch.tensor(aug_seq1, dtype=torch.long),torch.tensor(aug_seq2, dtype=torch.long), torch.tensor(similar_seqs, dtype=torch.long),


class TestDataset(Dataset):
    def __init__(self, item_list, max_seq_len):
        
        self.uid_list, self.item_list, self.target_list, self.item_list_length = self.data_process(item_list)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.item_list[idx]
        target = self.target_list[idx]
        length = self.item_list_length[idx]

        # Padding sequence to maxlen
        if length < self.max_seq_len:
            padded_seq = np.zeros(self.max_seq_len, dtype=np.int32)
            padded_seq[-length:] = seq[:]
        else:
            padded_seq = seq
        padded_seq = padded_seq[-self.max_seq_len:]
        # print("debug padded_seq", padded_seq)
        return torch.tensor(uid, dtype=torch.long), torch.tensor(padded_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def data_process(self, data_dict):
    
        uid_list, item_list, target_list, item_list_length = [], [], [], []

        for uid, item_id_seq in data_dict.items():
            # print("debug item_id_seq", item_id_seq)
            if len(item_id_seq)>1:
                uid_list.append(uid)
                item_list.append(item_id_seq[:-1])  
                target_list.append(item_id_seq[-1]) 
                item_list_length.append(len(item_id_seq[:-1]))
        # print("debug uid_list", uid_list)
        return uid_list, item_list, target_list, item_list_length


def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))



def data_augmentation(data_dict, max_seq_len, seq_keys_to_int):
    
    uid_list, item_list, target_list, item_list_length = [], [], [], []
    seq_unique_key_list = []

    for uid, item_id_seq in data_dict.items():
        seq_start = 0
        for i in range(1, len(item_id_seq)):
            if i - seq_start > max_seq_len:
                seq_start += 1
            uid_list.append(uid)
            item_list.append(item_id_seq[seq_start:i])  
            target_list.append(item_id_seq[i]) 
            item_list_length.append(i - seq_start)
            
            seq_unique_key = ":".join(map(str, [uid] + item_id_seq[seq_start:i]))
            seq_unique_key_list.append(seq_keys_to_int[seq_unique_key])
            
    print("debug sample num: ", len(item_list))
    print("debug seq_unique_key_list num: ", len(seq_unique_key_list))
    print("debug seq_unique_key_list", seq_unique_key_list[:10])
    return uid_list, item_list, target_list, item_list_length, seq_unique_key_list

    
    
# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    # f = open('data/%s.txt' % fname, 'r')
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
           
            user_valid[user] = (User[user][:-1])
            
            user_test[user] = (User[user][:])
    return [user_train, user_valid, user_test, usernum, itemnum]



def evaluate(args, model, eval_data, load_best_model=True, model_file=None, show_progress=False):
   
    if not eval_data:
        # print("debug eval data==", eval_data)
        print("====debug no eval data===")
        return

    if load_best_model:
        if model_file:
            checkpoint_file = model_file
        else:
            checkpoint_file = args.saved_model_file
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
        print(message_output)
        
    model.eval()

    prog_iter = tqdm(eval_data, leave=False)

    scores = []
    labels = []
    # torch.tensor(uid, dtype=torch.long), torch.tensor(padded_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor(length, dtype=torch.long)
    
    for batch in prog_iter:
        uid, item_seq, pos_item, item_seq_len = \
                                            batch[0].to(args.device), \
                                            batch[1].to(args.device), \
                                            batch[2].to(args.device), \
                                            batch[3].to(args.device)
                                                           
        # print("debug item_seq", item_seq.shape)     
        bs_scores = model.full_sort_predict(item_seq).detach().cpu()
        # print("debug bs_scores", bs_scores.shape) 
        batch_user_index = uid.cpu().numpy()
      
        # if not test:
        #     bs_scores[self.generator.valid_rating_matrix[batch_user_index].toarray() > 0] = -100
        # else:
        #     bs_scores[self.generator.test_rating_matrix[batch_user_index].toarray() > 0] = -100
        
        bs_labels = (pos_item-1).reshape(-1,1).cpu()
        scores.append(bs_scores)
        labels.append(bs_labels)
        
    scores = torch.cat(scores, axis=0).numpy()
    partitioned_indices = np.argpartition(-scores, 20, axis=1)[:, :20]
    pred_list = partitioned_indices[np.arange(scores.shape[0])[:, None], np.argsort(-scores[np.arange(scores.shape[0])[:, None], partitioned_indices], axis=1)].tolist()
    labels = torch.cat(labels, axis=0).numpy().tolist()

    result = get_full_sort_score(labels, pred_list)

    return result


def full_sort_batch_eval(args, model, batched_data):
    
    uid, item_seq, target, item_seq_len = batched_data
    
    scores = model.full_sort_predict(item_seq.to(args.device))
    
    scores = scores.view(-1, args.item_num)
    scores[:, 0] = -np.inf
    
    # if history_index is not None:
    #     scores[history_index] = -np.inf

    swap_row = swap_row.to(args.device)
    swap_col_after = swap_col_after.to(args.device)
    swap_col_before = swap_col_before.to(args.device)
    scores[swap_row, swap_col_after] = scores[swap_row, swap_col_before]

    return item_seq, scores


def train_epoch(args, model, train_data, optimizer, epoch_idx, loss_func=None, show_progress=True, device=None):
     
    model.train()
    
    total_rec_loss = None
    total_item_cl_loss = None
    total_user_cl_loss = None
    total_loss = None
    
    iter_data = (
        tqdm(
            enumerate(train_data),
            total=len(train_data),
            desc=f"Train {epoch_idx:>5}",
        ) if show_progress else enumerate(train_data)
    )
    for batch_idx, batch_data in iter_data:
        # print("debug interaction: ", batch_data)
        
        seq_unique_id, item_seq, pos_items, item_seq_len, aug_seq1, aug_seq2, neighbor_seqs = batch_data
        item_seq = item_seq.to(args.device)
        pos_items = pos_items.to(args.device)
        # print('debug pos_items: ', min(pos_items))
        item_seq_len = item_seq_len.to(args.device)
        aug_seq1 = aug_seq1.to(args.device)
        aug_seq2 = aug_seq2.to(args.device)
        neighbor_seqs = neighbor_seqs.to(args.device)
       
        
        optimizer.zero_grad()
        # print("debug labels", pos_items)
        rec_loss = model.calculate_loss(item_seq, pos_items)
        
        item_cl_loss = model.calculate_item_cl_loss(aug_seq1, aug_seq2, item_seq_len) 
        user_cl_loss = model.calculate_user_cl_loss(item_seq, neighbor_seqs, seq_unique_id)
        # losses = rec_loss +  0.1 * item_cl_loss + 0.1 * user_cl_loss
        losses = rec_loss + args.alpha * user_cl_loss + args.beta * item_cl_loss
       
        
        total_rec_loss = rec_loss.item() if total_rec_loss is None else total_rec_loss + rec_loss.item()
        total_item_cl_loss = item_cl_loss.item() if total_item_cl_loss is None else total_item_cl_loss + item_cl_loss.item()
        total_user_cl_loss = user_cl_loss.item() if total_user_cl_loss is None else total_user_cl_loss + user_cl_loss.item()
        total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        # print(loss.item())
        check_nan(losses)
        
        losses.backward()
        optimizer.step()

    return (total_rec_loss, total_item_cl_loss, total_user_cl_loss)

def valid_epoch(args, model, valid_data, show_progress=False):
    r"""Valid the model with valid data

    Args:
        valid_data (DataLoader): the valid data.
        show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

    Returns:
        float: valid score
        dict: valid result
    """
    
    valid_result = evaluate(args, model, valid_data, load_best_model=False)
    valid_score = calculate_valid_score(valid_result, args.valid_metric)
    # print("im here!!!!!!!!!!!")
    return valid_score, valid_result


def save_checkpoint(epoch, model, saved_model_file):
    r"""Store the model parameters information and training information.

    Args:
        epoch (int): the current epoch id

    """
    state = {'state_dict': model.state_dict()}
    torch.save(state, saved_model_file)
    
    
def generate_train_loss_output(epoch_idx, s_time, e_time, losses):
    des = 4
    train_loss_output = (('epoch %d training') + ' [' + ('time') +
                            ': %.2fs, ') % (epoch_idx, e_time - s_time)
    if isinstance(losses, tuple):
        des = (('train_loss%d') + ': %.' + str(des) + 'f')
        train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
    else:
        des = '%.' + str(des) + 'f'
        train_loss_output += ('train loss') + ': ' + des % losses
    return train_loss_output + ']'



def get_full_sort_score(answers, pred_list):
        recall, ndcg, mrr = [], [], []
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
            mrr.append(mrr_at_k(answers, pred_list, k))
        post_fix = {
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2])
        }
        # print(post_fix)
        # result_dic = {
        #     "HIT@5": recall[0], "NDCG@5": ndcg[0], "MRR@5": mrr[0],
        #     "HIT@10": (recall[1]), "NDCG@10": (ndcg[1]), "MRR@10": mrr[1],
        #     "HIT@20": (recall[2]), "NDCG@20": (ndcg[2]), "MRR@20": mrr[2],
        # }
        result_dic = {
        "HIT@5": round(recall[0], 4), "NDCG@5": round(ndcg[0], 4), "MRR@5": round(mrr[0], 4),
        "HIT@10": round(recall[1], 4), "NDCG@10": round(ndcg[1], 4), "MRR@10": round(mrr[1], 4),
        "HIT@20": round(recall[2], 4), "NDCG@20": round(ndcg[2], 4), "MRR@20": round(mrr[2], 4),
    }

        
        return result_dic
        
        
def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    # print("actual", len(actual))
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
    
def mrr_at_k(actual, predicted, topk):
   
    sum_mrr = 0.0
    num_users = len(predicted)
    
    for i in range(num_users):
        act_set = set(actual[i])
        for rank, item in enumerate(predicted[i][:topk], start=1):
            if item in act_set:
                sum_mrr += 1.0 / rank
                break
    
    return sum_mrr / num_users


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model
    
    
import torch.nn as nn

def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
        # print("debug z",z.shape )
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    
        mask = mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def info_nce_single(seq, seq_similar, temp, batch_size, sim='dot'):
    """
    For each sample in seq, the positive sample is the corresponding sample in seq_similar.
    The other samples in seq_similar are treated as negative samples.
    """
    # Compute similarity
    if sim == 'cos':
        sim = nn.functional.cosine_similarity(seq.unsqueeze(1), seq_similar.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(seq, seq_similar.T) / temp
    
    # Extract positive samples
    positive_samples = torch.diag(sim).reshape(batch_size, 1)
    
    # Create mask to exclude positive samples
    mask = mask_correlated_samples_single(batch_size)
    negative_samples = sim[mask].reshape(batch_size, -1)
    
    # Create labels
    labels = torch.zeros(batch_size).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels

def mask_correlated_samples_single(batch_size):
    mask = torch.ones((batch_size, batch_size), dtype=bool)
    mask = mask.fill_diagonal_(0)  
    return mask


def compute_cosine_similarity_batch(user_semantic_emb, top_k=100, batch_size=1024):
    num_users = user_semantic_emb.size(0)
    user_sorted_indices = np.zeros((num_users, top_k), dtype=np.int32)

    # Normalize the embeddings
    user_semantic_emb = user_semantic_emb / user_semantic_emb.norm(dim=1, keepdim=True)

    # Create a DataLoader for batching
    dataset = TensorDataset(user_semantic_emb)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in enumerate(dataloader):
        batch_emb = batch[0].cuda()  # Move batch to GPU
        batch_emb = batch_emb / batch_emb.norm(dim=1, keepdim=True)  # Normalize the batch embeddings
        batch_cosine_sim = torch.mm(batch_emb, user_semantic_emb.t().cuda())  # Compute cosine similarity
        _, batch_topk_indices = torch.topk(batch_cosine_sim, top_k, dim=1, largest=True, sorted=True)  # Get top k indices
        batch_topk_indices = batch_topk_indices.cpu().numpy()  # Move back to CPU

        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_users)
        user_sorted_indices[start_idx:end_idx, :] = batch_topk_indices

    return user_sorted_indices
