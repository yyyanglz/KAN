from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans, proc_fact, get_top_obj

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npy')

        # if __C.EVAL_EVERY_EPOCH and __C.RUN_MODE in ['train']:
        #     self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz')

        # else:
        #     self.img_feat_path_list = \
        #         glob.glob(__C.IMG_FEAT_PATH['train'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['test'] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        self.react_obj_rel = json.load(open(__C.QUESTION_PATH['concept'], 'r'))
        self.img_obj_train = json.load(open(__C.IMG_OBJ_PATH['train'], 'r'))
        self.img_obj_val = json.load(open(__C.IMG_OBJ_PATH['val'], 'r'))
        self.img_obj_test = json.load(open(__C.IMG_OBJ_PATH['test'], 'r'))

        # Loading answer word list
        # self.stat_ans_list = \
        #     json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
        #     json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)

        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        if self.__C.PRELOAD:
            print('==== Pre-Loading features ...')
            time_start = time.time()
            self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
            time_end = time.time()
            print('==== Finished in {}s'.format(int(time_end - time_start)))
        else:
            self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        # self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.react_rel = list(self.react_obj_rel.values())
        self.token_to_ix, self.pretrained_emb = tokenize(self.react_rel, self.stat_ques_list, __C.USE_GLOVE)

        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        # self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')

    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                # img_feat_x = img_feat['x'].transpose((1, 0))
                img_feat_x = img_feat
            img_feat_iter = img_feat_x  # proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

            # get 36 objects from img_obj_train.json
            objects = self.img_obj_train[str(ans['image_id'])]  # ['person', 'windows', ..., 'apple']

            # get top 10 objects
            top_objects = get_top_obj(objects, top=self.__C.TOP_OBJ)
            while len(top_objects) < self.__C.TOP_OBJ:
                top_objects.append('pad_obj')
                # print('image id:', str(ans['image_id']), 'top_objects:', top_objects)

            # query object from react_obj_rel, and only get top 5 sentences
            top_fact = []
            pad_rel = self.react_obj_rel['pad_obj'][0]
            for obj in top_objects:
                relations = self.react_obj_rel[obj]
                curr_top_fact = relations[:self.__C.TOP_REL]
                while len(curr_top_fact) < self.__C.TOP_REL:
                    curr_top_fact.append(pad_rel)
                top_fact.extend(curr_top_fact)

            # we get about 50 fact sentence index
            num_token = self.__C.FACT_TOKEN*self.__C.TOP_OBJ*self.__C.TOP_REL
            fact_idx_iter = proc_fact(top_fact, self.token_to_ix, num_token, self.__C.FACT_TOKEN)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            # # Process image feature from (.npz) file
            # img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            # img_feat_x = img_feat['x'].transpose((1, 0))
            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
                # img_feat_x = img_feat['x'].transpose((1, 0))
                img_feat_x = img_feat
            img_feat_iter = img_feat_x  # proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # get 36 objects from img_obj_train.json
            objects = self.img_obj_val[str(ques['image_id'])]  # ['person', 'windows', ...]

            # get top 10 objects
            top_objects = get_top_obj(objects, top=self.__C.TOP_OBJ)
            while len(top_objects) < self.__C.TOP_OBJ:
                top_objects.append('pad_obj')
                # print('image id:', str(ans['image_id']), 'top_objects:', top_objects)

            # query object from react_obj_rel, and only get top 5 sentences
            top_fact = []
            pad_rel = self.react_obj_rel['pad_obj'][0]
            for obj in top_objects:
                relations = self.react_obj_rel[obj]
                curr_top_fact = relations[:self.__C.TOP_REL]
                while len(curr_top_fact) < self.__C.TOP_REL:
                    curr_top_fact.append(pad_rel)
                top_fact.extend(curr_top_fact)

            # we get about 50 fact sentence index
            num_token = self.__C.FACT_TOKEN*self.__C.TOP_OBJ*self.__C.TOP_REL
            fact_idx_iter = proc_fact(top_fact, self.token_to_ix, num_token, self.__C.FACT_TOKEN)

        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter), \
               torch.from_numpy(fact_idx_iter)

    def __len__(self):
        return self.data_size
