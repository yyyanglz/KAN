import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.DATASET_PATH = '/data/Feature/vqa/'

        # bottom up features root path
        self.FEATURE_PATH = '/data/Feature/vqa/'

        # image object root path
        self.OBJECT_PATH = '/data/Feature/short_id/'

        self.init_path()

    def init_path(self):

        self.IMG_FEAT_PATH = {
            'train': self.FEATURE_PATH + 'train2014/',
            'val': self.FEATURE_PATH + 'val2014/',
            'test': self.FEATURE_PATH + 'test2015/',
        }

        self.IMG_OBJ_PATH = {
            'train': self.OBJECT_PATH + 'img_obj_train.json',
            'val': self.OBJECT_PATH + 'img_obj_val.json',
            'test': self.OBJECT_PATH + 'img_obj_test.json',
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'val': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.DATASET_PATH + 'v2_OpenEnded_mscoco_test2015_questions.json',
            'vg': self.DATASET_PATH + 'VG_questions.json',
            'concept': self.DATASET_PATH + 'obj_rel.json',
        }

        self.ANSWER_PATH = {
            'train': self.DATASET_PATH + 'v2_mscoco_train2014_annotations.json',
            'val': self.DATASET_PATH + 'v2_mscoco_val2014_annotations.json',
            'vg': self.DATASET_PATH + 'VG_annotations.json',
        }

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.CACHE_PATH = './results/cache/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')

    def check_path(self):
        print('Checking dataset ...')

        for mode in self.IMG_FEAT_PATH:
            if not os.path.exists(self.IMG_FEAT_PATH[mode]):
                print(self.IMG_FEAT_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                exit(-1)

        print('Finished')
        print('')

