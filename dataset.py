import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from random import randint, choice
import  re
import datetime
import pandas as pd
import json
from tqdm import tqdm

class movielens_1m(object):
    def __init__(self):
        self.user_data, self.item_data, self.score_data = self.load()

    def load(self):
        path = "movielens/ml-1m"
        profile_data_path = "{}/users.dat".format(path)
        score_data_path = "{}/ratings.dat".format(path)
        item_data_path = "{}/movies_extrainfos.dat".format(path)

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'], 
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'], 
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
        self.n_users, self.n_items = score_data.user_id.max()+1, score_data.movie_id.max()+1
        score_data = score_data.drop(["timestamp"], axis=1)
        self.item_ids =  score_data.movie_id.unique()
        return profile_data, item_data, score_data
    
def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list): 
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


class Metamovie(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metamovie, self).__init__()
        #self.dataset_path = args.data_root
        self.partition = partition
        #self.pretrain = pretrain
        
        self.dataset_path = args.data_root
        dataset_path = self.dataset_path
        rate_list = load_list("{}/m_rate.txt".format(dataset_path))
        genre_list = load_list("{}/m_genre.txt".format(dataset_path))
        actor_list = load_list("{}/m_actor.txt".format(dataset_path))
        director_list = load_list("{}/m_director.txt".format(dataset_path))
        gender_list = load_list("{}/m_gender.txt".format(dataset_path))
        age_list = load_list("{}/m_age.txt".format(dataset_path))
        occupation_list = load_list("{}/m_occupation.txt".format(dataset_path))
        zipcode_list = load_list("{}/m_zipcode.txt".format(dataset_path))

        self.user_clicked_set = pickle.load(open("{}/user_clicked_set.pkl".format(dataset_path), 'rb'))

        self.dataset = movielens_1m()
        
        master_path = self.dataset_path
        if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
            self.movie_dict = {}
            for idx, row in self.dataset.item_data.iterrows():
                m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
                self.movie_dict[row['movie_id']] = m_info
            pickle.dump(self.movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
        else:
            self.movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))
        # hashmap for user profile
        if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
            self.user_dict = {}
            for idx, row in self.dataset.user_data.iterrows():
                u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
                self.user_dict[row['user_id']] = u_info
            pickle.dump(self.user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
        else:
            self.user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))
        if partition == 'train' or partition == 'valid':
            self.state = 'warm_state'
            self.training = True
        else:
            if test_way is not None:
                self.training = False
                if test_way == 'old':
                    self.state = 'warm_state'
                elif test_way == 'old_user':
                    self.state = 'user_old_state'
                elif test_way == 'new_user':
                    self.state = 'user_cold_state'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())
        with open("{}/global_popularity.json".format(dataset_path), encoding="utf-8") as f:
            pop_dict = json.loads(f.read())
            if type(next(iter(pop_dict.keys()))) == str:
                pop_dict = {int(key): pop_dict[key] for key in pop_dict}
            self.dataset_split_popularity = pop_dict


        length = len(self.dataset_split.keys())

        self.final_index = list(self.dataset_split.keys())
        '''
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])

            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
        '''

        self.test_buffer = torch.empty([1, 3704, 10246], dtype=torch.long)
        for idx, item_id in enumerate(self.dataset.item_ids):
            self.test_buffer[0, idx, 0:10242] = self.movie_dict[item_id]

        self.test_buffer_idx = torch.tensor(self.dataset.item_ids).view(1, -1).long()

        self.test_popularity = torch.empty([1, 3704, 1], dtype=torch.long)
        for idx, item_id in enumerate(self.dataset.item_ids):
            self.test_popularity[0, idx, 0] = pop_dict[item_id]

    ''' 
    # function for explicit feedback (original)
    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])
        
        # (# of user-item interactions, feature_size)
        support_x_app = None
        for m_id in tmp_x[indices[:-10]]:
            m_id = int(m_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
        query_x_app = None
        for m_id in tmp_x[indices[-10:]]:
            m_id = int(m_id)
            u_id = int(user_id)
            tmp_x_converted = torch.cat((self.movie_dict[m_id], self.user_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1)
    ''' 

    # function for implicit feedback
    # retrieve user-wise interaction history and a random negative item
    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        # random.shuffle(indices) # do not disturb chronological order
        tmp_x = np.array(self.dataset_split[str(u_id)])
        support_x_app = None

        support_x_app = torch.empty([len(tmp_x[indices[:-10]]), 2, 10246], dtype=torch.long)
        support_x_app[:, :, 10242:10246] = self.user_dict[u_id] # user info
        support_x_app_pop = torch.empty([len(tmp_x[indices[:-10]]), 2, 1], dtype=torch.double)
        # 0~10241 items, 10242~10245 users
        for m_idx, m_id in enumerate(tmp_x[indices[:-10]]):
            m_id_negs = self._sample_neg_items(u_id, num_neg=1, support=True)
            m_id = int(m_id)
            support_x_app[m_idx, 0, 0:10242] = self.movie_dict[m_id] # positive item
            support_x_app_pop[m_idx, 0] = self.dataset_split_popularity[m_id] # positive item
            for idx, m_id_neg in enumerate(m_id_negs):
                support_x_app[m_idx, idx+1, 0:10242] = self.movie_dict[m_id_neg.item()] # negative item
                #support_x_app[m_idx, idx+1, 10246] = self.dataset_split_popularity[m_id_neg.item()]
                try:
                    support_x_app_pop[m_idx, idx+1] = self.dataset_split_popularity[m_id_neg.item()] # positive item
                except Exception as e:
                    support_x_app_pop[m_idx, idx+1] = 0.000291 # lowest item popularity

        num_neg = 1 if (self.partition == 'train') else self.dataset.n_items
        #query_x_app = torch.empty([1, num_neg+1, 10247], dtype=torch.long)
        #query_x_app_idx = torch.empty([1, num_neg+1], dtype=torch.long)




        if self.training:
            query_x_app = torch.empty([10, num_neg+1, 10246], dtype=torch.long)
            query_x_app[:, :, 10242:10246] = self.user_dict[u_id] # user info
            query_x_app_idx = torch.empty([10, num_neg+1], dtype=torch.long)
            query_x_app_pop = torch.empty([10, num_neg+1,1], dtype=torch.double)

            for m_idx, m_id in enumerate(tmp_x[indices[-10:]]):
                m_id_negs = self._sample_neg_items(u_id, num_neg=num_neg, support=True)
                m_id = int(m_id)
                query_x_app[m_idx, 0, 0:10242] = self.movie_dict[m_id] # positive item
                query_x_app_pop[m_idx, 0] = self.dataset_split_popularity[m_id] # positive item
                query_x_app_idx[m_idx, 0] = m_id
                query_x_app_idx[m_idx, 1:num_neg+1] = m_id_negs
                for idx, m_id_neg in enumerate(m_id_negs):
                    query_x_app[m_idx, idx+1, 0:10242] = self.movie_dict[m_id_neg.item()]
                    try:
                        query_x_app_pop[m_idx, idx+1] = self.dataset_split_popularity[m_id_neg.item()]
                    except Exception as e:
                        query_x_app_pop[m_idx, idx+1] = 0.000291


        else:
            query_x_app = self.test_buffer
            query_x_app_idx = self.test_buffer_idx
            query_x_app_pop = self.test_popularity
        # total prediction values

        # (# of user-item interactions, pos_len+neg_len, feature_size)
        return support_x_app, query_x_app, query_x_app_idx, tmp_x[indices[:-10]], support_x_app_pop, query_x_app_pop


    def __len__(self):
        return len(self.final_index)

    def _sample_neg_items(self, u_id, num_neg=1, support=True):
        r"""Sample positive (for meta_batch) and negative items (for meta and current batch).
        For training: randomly sample 1 item.
        For testing: randomly sample 99 item.
        """
        neg_items = torch.empty(num_neg, dtype=torch.int64)
        user_clicked_set = self.user_clicked_set[u_id]
        for neg in range(num_neg):
            neg_item = self._randint_w_exclude(user_clicked_set)
            neg_items[neg] = neg_item

        return neg_items

    def _randint_w_exclude(self, clicked_set):
        randItem = randint(1, self.dataset.n_items-1)
        return self._randint_w_exclude(clicked_set) if (randItem in clicked_set) or (randItem not in self.movie_dict.keys()) else randItem

