import time
import sys
import copy
import heapq

import numpy as np
import pandas as pd
import scipy
from scipy.stats import mode
from argparse import ArgumentParser
from numba import jit
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

import eval_metrics
from dataset import loadData
from model import AutoEncoder
from utils import clone_module, update_module


if torch.cuda.is_available():
    import torch.cuda as T
else:
    import torch as T


parser = ArgumentParser(description="PREMERE")
# Recommender model related argument
parser.add_argument('-e', '--epoch', type=int, default=100, help='number of epochs for Recommender model')
parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size for training')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('-att', '--num_attention', type=int, default=20, help='the number of dimension of attention')
parser.add_argument('--inner_layers', nargs='+', type=int, default=[200, 50, 200], help='the number of latent factors')
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5, help='the dropout probability')

# Meta model related argument
parser.add_argument('-mw', '--meta_weighting', type=bool, default=False, help='whether to do meta weighting')
parser.add_argument('-ev', '--eval_every_epoch', type=bool, default=True, help='whether to do evaluation every epoch')
parser.add_argument('-q', '--queue', type=int, default=10, help='warm up period and queue size')
parser.add_argument('-ema', '--exponential_moving_average', type=float, default=0.95, help='Moving Average Factor to make mean teacher')
#parser.add_argument('-hk', '--hour_k', type=str, default='0', help='Adjacent hour info of the current model')

# Directory related argument
parser.add_argument('-d', '--dataset', type=str, default='Yelp', help='The name of the dataset')
parser.add_argument('-f', '--folder', type=str, default='./data/', help='The time similarity method')
parser.add_argument('-poi', '--poi_filename', type=str, default='poi_coos.txt', help='The poi filename')
parser.add_argument('-chk', '--checkin_filename', type=str, default='checkins.txt', help='The chekin filename')
args = parser.parse_args()

class POIDataset(Dataset):
    def __init__(self, train_matrix, directory_path, meta_weighting, debugging=False):
        self.meta_weighting = meta_weighting
        if self.meta_weighting: # For recommender update
            self.checkin_counts = train_matrix.toarray()
            self.place_correlation = np.load(directory_path + 'user_geo_context2.npy')
            self.user_entropy = np.load(directory_path + 'user_entropy.npy')
            self.time_correlation  = np.load(directory_path + 'user_time_context2.npy')
            self.poi_popularity = np.load(directory_path + 'poi_popularity.npy')
        train_matrix[train_matrix > 0] = 1.0
        self.label = train_matrix.toarray().astype(np.float16)
        self.valid_indices = self.label.astype(np.int8)
        self.stacking = 1 if not debugging else 2
        self.shape = train_matrix.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # derive train data
        checkin = Variable(torch.from_numpy(self.label[idx].astype(np.uint8)).type(torch.FloatTensor), requires_grad=False)
        label = Variable(torch.from_numpy(self.label[idx]).type(torch.FloatTensor), requires_grad=False)
        valid_indices = Variable(torch.from_numpy(self.valid_indices[idx]).type(torch.FloatTensor), requires_grad=False)
        # derive meta reweight data
        if self.meta_weighting:
            geo_context = Variable(torch.from_numpy(self.place_correlation[idx]).type(torch.FloatTensor), requires_grad=False)
            time_context = Variable(torch.from_numpy(self.time_correlation[idx]).type(torch.FloatTensor), requires_grad=False)
            checkin_cnt = Variable(torch.from_numpy(self.checkin_counts[idx]).type(torch.FloatTensor), requires_grad=False)
            checkin_entropy = Variable(torch.from_numpy(np.tile(self.user_entropy[idx], self.shape[1])).type(torch.FloatTensor), requires_grad=False)
            poi_popularity = Variable(torch.from_numpy(self.poi_popularity).type(torch.FloatTensor), requires_grad=False)
            meta_batch = torch.stack((geo_context, time_context, checkin, checkin_entropy, poi_popularity, checkin_cnt), self.stacking)
            return checkin, label, meta_batch, valid_indices
        else:
            return checkin, label, valid_indices

@jit(cache=True, forceobj=True, parallel=True)
def get_var_reduce(arr, threshold=0.25):
    mu = np.mean(arr, axis=2, dtype=np.float16)
    index = (np.var(arr, axis=2, dtype=np.float16) <= threshold*(mu**2))
    return mu, index


def get_predictions(model, dataloader, shape, place_corr=None, t=None, save=False, save_user_cnt=1000):
    if torch.cuda.is_available():
        model.cuda()
    pred = np.zeros(shape, dtype=np.float16)
    batch_size = args.batch_size

    for idx, _ in enumerate(dataloader):
        checkin = _[0]
        start = idx * batch_size
        end = start + batch_size

        model_input = [list(np.nonzero(_)[0]) for _ in checkin.numpy()]
        pred[start:end] = model(model_input, place_corr).cpu().data.numpy().astype(np.float16)

    if save:
        np.random.seed(0)
        save_user_indices = np.random.randint(0, shape[0], save_user_cnt, dtype=np.uint16)
        np.save("./tmp_pred/" + save + "_" + str(t) + ".npy", pred[save_user_indices])
        np.save("./tmp_pred/" + save + "_" + str(t) + "_all.npy", pred)

    return pred


def train_recommender(train_matrix, test_set, directory_path):
    num_users, num_items = train_matrix.shape
    train_dataset = POIDataset(train_matrix, directory_path, meta_weighting=True)
    if args.meta_weighting:
        meta_dataset = POIDataset(train_matrix, directory_path, meta_weighting=False)
        del train_matrix
        initial_valids = len(meta_dataset.valid_indices.nonzero()[0])
    batch_size = args.batch_size
    teacher_model = None

    precision_max = [0]
    precision = np.zeros((args.epoch, 5), dtype=np.float16)
    recall = np.zeros((args.epoch, 5), dtype=np.float16)
    MAP = np.zeros((args.epoch, 5), dtype=np.float16)
    model = AutoEncoder(num_items, args.inner_layers, num_items, da=args.num_attention, dropout_rate=args.dropout_rate)
    place_corr = scipy.sparse.load_npz(directory_path + 'place_correlation_gamma60.npz')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        model.cuda()
    if args.meta_weighting:
        print("meta weighting during update")
        meta_model = MNet(7)
        optimizer_meta = torch.optim.Adam(meta_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        pred_history = np.full((num_users, num_items, args.queue), 1, dtype=np.float16)
        if torch.cuda.is_available(): meta_model.cuda()
    else: print("default weighting")

    criterion = torch.nn.MSELoss(size_average=False, reduce=False)

    for t in range(0, args.epoch):
        precision_tmp = np.zeros(5, dtype=np.float16)
        recall_tmp = np.zeros(5, dtype=np.float16)
        MAP_tmp  = np.zeros(5, dtype=np.float16)

        model.train()
        print("epoch:{}".format(t))
        avg_cost = 0.
        train_loader = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True)
        if args.meta_weighting:
            meta_loader = DataLoader(meta_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
            meta_iterator = iter(meta_loader)
        for idx, (checkin, label, meta_batch, valid_indices_train) in enumerate(train_loader):
            if args.meta_weighting:
                meta_model.train()
                # Get clone prediction
                model_clone = clone_module(model)
                if torch.cuda.is_available():
                    model_clone.cuda()
                model_input = [list(np.nonzero(_)[0]) for _ in checkin.numpy()]
                clone_pred = model_clone(model_input, place_corr)

                loss_default = criterion(clone_pred, label.cuda())
                # Get meta reweight
                meta_input = torch.cat((loss_default.detach().unsqueeze(2), meta_batch.cuda()), 2)
                reweight = meta_model(meta_input)
                # Update recommender to make a fake step
                loss = (reweight * loss_default).sum() / len(checkin)

                grad = torch.autograd.grad(loss, model_clone.parameters(), create_graph=True)
                updates = [-args.learning_rate * g for g in grad]
                model_clone = update_module(model_clone, updates=updates)

                # Sample users from another random mini-batch in meta-dataset
                # Get another loss using the updated recommender
                meta_checkin, meta_label, valid_indices = next(meta_iterator)

                model_input = [list(np.nonzero(_)[0]) for _ in meta_checkin.numpy()]
                clone_pred = model_clone(model_input, place_corr)

                meta_loss = (valid_indices.cuda() * criterion(clone_pred, meta_label.cuda())).sum() / len(meta_label)

                # Update meta model
                optimizer_meta.zero_grad()
                meta_loss.backward()
                optimizer_meta.step()

                del meta_checkin, valid_indices, meta_label

                batch_x_weight = meta_model(meta_input)

                del loss_default
                del model_clone

            else: batch_x_weight = 1

            model_input = [list(np.nonzero(_)[0]) for _ in checkin.numpy()]
            y_pred = model(model_input, place_corr)

            # Compute and print loss
            loss = (batch_x_weight * criterion(y_pred, checkin.cuda())).sum() / len(checkin)

            if args.meta_weighting:  print(idx, loss.item(), meta_loss.item(), "epoch:{}".format(t))
            else:                    print(idx, loss.data, "epoch:{}".format(t))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_cost += loss / num_users * len(checkin)



        print("Avg loss:{}".format(avg_cost))

        del checkin, label, meta_batch, batch_x_weight, y_pred

        if args.meta_weighting:
            print("making teacher model")
            meta_loader = DataLoader(meta_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)

            # Generate teacher model
            tmp_model = copy.deepcopy(model)
            params = tmp_model.named_parameters()
            dict_params = dict(params)
            # Add parameters of models in the queue
            if teacher_model is None: teacher_model = tmp_model
            else:
                tmp_params2 = teacher_model.named_parameters()
                for tmp_name2, tmp_param2 in tmp_params2:
                    if tmp_name2 in dict_params:
                        dict_params[tmp_name2].data.copy_(dict_params[tmp_name2].data*(1-args.exponential_moving_average) + tmp_param2.data*args.exponential_moving_average)
                teacher_model = tmp_model

            # Predict and get consistent user preference
            tmp_pred = get_predictions(tmp_model,
                                       meta_loader,
                                       (num_users, num_items),
                                       place_corr=place_corr,
                                       t=t)
            del tmp_model

            #tmp_pred = np.int8(tmp_pred*10) # quantize prediction result
            pred_history[:,:,t % 10] = tmp_pred # record the present prediction in the queue of history
            del tmp_pred

            if (t >= args.queue-1) and (t < args.epoch-1):
                refurbish_label, refurbish_index = get_var_reduce(pred_history) # Identify refurbishable samples
                meta_dataset.label[(meta_dataset.label!=1)&refurbish_index] = refurbish_label[(meta_dataset.label!=1)&refurbish_index]
                meta_dataset.valid_indices = ((meta_dataset.label).astype(np.int8)|(refurbish_index).astype(np.int8))
                del refurbish_label, refurbish_index

        train_loader = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=False)

        if (args.eval_every_epoch) or (args.epoch-1 == t):
            model.eval()
            topk = 50

            pred_rating_matrix = get_predictions(model, train_loader, (num_users, num_items), place_corr=place_corr, t=t)
            pred_rating_matrix[train_dataset.label>0] = 0
            recommended_list = np.zeros((num_users, topk), dtype=np.uint16)
            for user in range(num_users):
                recommended_list[user] = np.argsort(-pred_rating_matrix[user])[:topk]
            del pred_rating_matrix

            for top_idx, top_items in enumerate([5, 10, 15, 20, 50]):
                precision_tmp[top_idx] = eval_metrics.precision_at_k(test_set, recommended_list, top_items)
                recall_tmp[top_idx] = eval_metrics.recall_at_k(test_set, recommended_list, top_items)
                MAP_tmp[top_idx] = eval_metrics.mapk(test_set, recommended_list, top_items)

            # Append to history
            precision[t] = precision_tmp
            recall[t] = recall_tmp
            MAP[t] = MAP_tmp
            print(precision_tmp)

# Meta Network
import torch.nn as nn
import torch.nn.functional as F
class MNet(nn.Module):
    def __init__(self, input_dim, hidden=50):
        super(MNet, self).__init__()
        self.linear_latent_factor = nn.Linear(input_dim-1, hidden)
        self.linear_unvisited = nn.Linear(hidden, 1)
        self.linear_visited = nn.Linear(hidden, 1, bias=False)
        # check-in dependent
        self.linear_checkin_activator = nn.Linear(1, hidden, bias=False) # bias should be false to cover zero check-in

        #self.bn1 = torch.nn.BatchNorm1d(hidden)

    def forward(self, x):
        #x = F.relu(self.conv1(x).squeeze(1))
        poi_info = x[:, :, :-1]
        checkins = x[:, :, -1:]
        latent_similarity = F.relu(self.linear_latent_factor(poi_info))
        unv_factor = F.sigmoid(self.linear_unvisited(latent_similarity))+ 1
        unv_factor[checkins>=1] = 0 # Switch: make weights of visited POIs to zero

        checkin_factor = F.relu(self.linear_checkin_activator(checkins))
        v_factor = torch.log((1+F.relu(self.linear_visited(checkin_factor*latent_similarity))/0.0001))
        v_factor[checkins<1] = 0  # Switch: make weights of unvisited POIs to zero
        return (unv_factor + v_factor).squeeze(-1)

def main():
    directory_path = args.folder + args.dataset + '/'
    train_matrix, test_set, raw_matrix = loadData(args.dataset, directory_path, args.checkin_filename, args.poi_filename)
    train_recommender(train_matrix, test_set, directory_path)

if __name__ == '__main__':
    main()
