import os
from time import time
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from model import TransRec
from utils import *
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str
from logging import getLogger
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--k_num', default=10, type=int)
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--train_batch_size', default=256, type=int)
parser.add_argument('--test_batch_size', default=1024, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--inner_size', default=256, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--n_heads', default=2, type=int)
parser.add_argument('--n_layers', default=2, type=int)
# parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--hidden_dropout_prob', default=0.5, type=float)
parser.add_argument('--attn_dropout_prob', default=0.5, type=float)
# parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--initializer_range', default=0.02, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--hidden_act', default='gelu', type=str)
parser.add_argument('--loss_type', default='CE', type=str)
parser.add_argument('--valid_metric', default='MRR@10', type=str)
parser.add_argument('--stopping_step', default=10, type=int)
parser.add_argument('--eval_step', default=1, type=int)
parser.add_argument('--valid_metric_bigger', default=True, type=bool)
parser.add_argument('--log_dir', default='./log/', type=str)
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--item_semantic_emb_file', default='./data/Beauty/beauty_item_semantic_embeddings.pt', type=str)
parser.add_argument('--user_semantic_emb_file', default='./data/Beauty/beauty_user_semantic_embeddings.pt', type=str)
parser.add_argument('--seq_keys_to_int_file', default='./data/Beauty/beauty_keys_to_int.pkl', type=str)
parser.add_argument('--user_sorted_indices_file', default='./data/Beauty/beauty_user_sorted_indices.npy', type=str)
parser.add_argument('--dataset_file', default='./data/Beauty/Beauty.txt', type=str)
parser.add_argument('--temperature', default=1.0, type=float)


args = parser.parse_args()
set_seed(args.seed)
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
args.saved_model_file = os.path.join(args.log_dir, args.dataset, f"model_{current_time}.pth")
args.saved_result_file = f"{args.log_dir}{args.dataset}/results_alpha_{args.alpha:.3f}_beta_{args.beta:.3f}_k_{args.k_num}.txt"
print("debug args.saved_result_file", args.saved_result_file)
print("debug args.saved_model_file", args.saved_model_file)

with open(args.seq_keys_to_int_file, 'rb') as f:
    seq_keys_to_int = pickle.load(f)
seq_int_to_keys = {value: key for key, value in seq_keys_to_int.items()}
args.seq_int_to_keys = seq_int_to_keys
args.seq_keys_to_int = seq_keys_to_int


logger = getLogger()
if __name__ == '__main__':
    item_semantic_emb = torch.load(args.item_semantic_emb_file)
    cosine_sim_matrix = cosine_similarity(item_semantic_emb.numpy())
    sorted_indices = np.argsort(-cosine_sim_matrix, axis=1)
    sorted_indices = sorted_indices[:, 1:args.k_num+1] # does not include itself
    args.sorted_indices = torch.tensor(sorted_indices).to(args.device)
    args.item_semantic_emb = item_semantic_emb.to(args.device)
    

    user_sorted_indices_file = args.user_sorted_indices_file
    user_semantic_emb = torch.load(args.user_semantic_emb_file).cuda() 
    user_sorted_indices = compute_cosine_similarity_batch(user_semantic_emb)
    np.save(user_sorted_indices_file, user_sorted_indices)
        
    
    user_sorted_indices = user_sorted_indices[:, 1:args.k_num+1] # does not include itself
    args.user_sorted_indices = torch.tensor(user_sorted_indices).long().to(args.device)
    args.user_semantic_emb = user_semantic_emb.to(args.device)
    
    print("computing semantic similarity over...")
        
    args.sorted_indices_numpy = sorted_indices
    args.user_sorted_indices_numpy = user_sorted_indices
    
    # global dataset
    dataset = data_partition(args.dataset_file)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    uid_list, item_list, target_list, item_list_length, seq_unique_key_list = data_augmentation(user_train, args.maxlen, seq_keys_to_int)
    
    args.item_num = itemnum
    
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    num_batch = (len(item_list) - 1) // args.train_batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    # sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
   
    TrainData = SequenceDataset(args, uid_list, item_list, target_list, item_list_length, seq_unique_key_list, args.maxlen)
    TrainDataLoader = DataLoader(TrainData, batch_size=args.train_batch_size, shuffle=True)
    
    num_samples = len(TrainData)
    print(f"Number of samples in TrainDataLoader: {num_samples}")

    # sample = next(iter(TrainDataLoader))
    # print("One sample from TrainDataLoader:")
    # print(sample[1])
    # print(sample[2])
    
    ValData = TestDataset(user_valid, args.maxlen)
    ValDataLoader = DataLoader(ValData, batch_size=args.test_batch_size, shuffle=False)
    TestData = TestDataset(user_test, args.maxlen)
    TestDataLoader = DataLoader(TestData, batch_size=args.test_batch_size, shuffle=False)
    
    num_samples = len(ValData)
    print(f"Number of samples in ValDataLoader: {num_samples}")
    
    
    model = TransRec(args).to(args.device) # no ReLU activation in original SASRec implementation?
    print("-------model initialized-----------")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
 
    # =============== train ===================
    start_epoch = 0
    verbose = True
    saved = True
    train_loss_dict = dict()
    best_valid_score = -np.inf if args.valid_metric_bigger else np.inf
    cur_step = 0
    if saved and start_epoch >= args.epochs:
        save_checkpoint(-1, model, args.saved_model_file)
    
    for epoch_idx in range(start_epoch, args.epochs):
        # train
        training_start_time = time()
        train_loss = train_epoch(args, model, TrainDataLoader, optimizer, epoch_idx)
        # print("debug train_loss", train_loss)
        # train_epoch(model, train_data, optimizer, epoch_idx, loss_func=None, show_progress=False, device=None)
        
        train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
        
        training_end_time = time()
        train_loss_output = \
            generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
        if verbose:
            logger.info(train_loss_output)
        print(train_loss_output)
        
        # eval
        if args.eval_step <= 0:
            if saved:
                save_checkpoint(epoch_idx, model, args.saved_model_file)
                update_output = ('Saving current') + ': %s' % args.saved_model_file
                if verbose:
                    logger.info(update_output)
            continue
        
        if (epoch_idx) % args.eval_step == 0:
            valid_start_time = time()
            valid_score, valid_result = valid_epoch(args, model, ValDataLoader, show_progress=False)
            best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
                valid_score,
                best_valid_score,
                cur_step,
                max_step=args.stopping_step,
                bigger=args.valid_metric_bigger
            )
            valid_end_time = time()
            valid_score_output = (("epoch %d evaluating") + " [" + ("time")
                                + ": %.2fs, " + ("valid_score") + ": %f]") % \
                                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
            valid_result_output = ('valid result') + ': \n' + dict2str(valid_result)
            # print(valid_score_output)
            print(valid_result_output)
            if verbose:
                logger.info(valid_score_output)
                logger.info(valid_result_output)
            if update_flag:
                if saved:
                    save_checkpoint(epoch_idx, model, args.saved_model_file)
                    update_output = ('Saving current best') + ': %s' % args.saved_model_file
                    if verbose:
                        logger.info(update_output)
                    print(update_output)
                best_valid_result = valid_result

            if stop_flag:
                stop_output = 'Finished training, best eval result in epoch %d' % \
                                (epoch_idx - cur_step * args.eval_step)
                if verbose:
                    logger.info(stop_output)
                break
    
    # test
    test_result = evaluate(args, model, TestDataLoader, load_best_model=True)
    df = pd.DataFrame([test_result])
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    df.to_csv(args.saved_result_file, index=False, sep='\t')
    print(('best valid ') + f': {best_valid_result}')
    print(('test result') + f': {test_result}')


