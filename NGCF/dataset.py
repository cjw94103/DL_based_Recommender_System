import numpy as np
import random as rd

class SampleGenerator(object):
    def __init__(self, trainset_path, valset_path, user_item_info_path, batch_size):
        self.train_items = np.load(trainset_path, allow_pickle=True).item()
        self.valset = np.load(valset_path, allow_pickle=True).item()

        user_item_info = np.load(user_item_info_path, allow_pickle=True).item()
        self.n_users, self.n_items = user_item_info['n_users'], user_item_info['n_items']
        self.n_train, self.n_val = user_item_info['n_train'], user_item_info['n_val']
        self.exist_users = user_item_info['exist_users']

        self.batch_size = batch_size

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in self.valset[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items