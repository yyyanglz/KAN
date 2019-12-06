from core.data.ans_punct import prep_ans
import numpy as np
import en_vectors_web_lg, random, re, json
from collections import Counter


def shuffle_list(ans_list):
    random.shuffle(ans_list)


# ------------------------------
# ---- Initialization Utils ----
# ------------------------------

def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


def img_feat_load(path_list):
    iid_to_feat = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img_feat = np.load(path)
        img_feat_x = img_feat['x'].transpose((1, 0))
        iid_to_feat[iid] = img_feat_x
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')

    return iid_to_feat


def ques_load(ques_list):
    qid_to_ques = {}

    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques

    return qid_to_ques


# def tokenize(stat_ques_list, use_glove):
#     token_to_ix = {
#         'PAD': 0,
#         'UNK': 1,
#     }
#
#     spacy_tool = None
#     pretrained_emb = []
#     if use_glove:
#         spacy_tool = en_vectors_web_lg.load()
#         pretrained_emb.append(spacy_tool('PAD').vector)
#         pretrained_emb.append(spacy_tool('UNK').vector)
#
#     for ques in stat_ques_list:
#         words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             ques['question'].lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
#         for word in words:
#             if word not in token_to_ix:
#                 token_to_ix[word] = len(token_to_ix)
#                 if use_glove:
#                     pretrained_emb.append(spacy_tool(word).vector)
#
#     pretrained_emb = np.array(pretrained_emb)
#
#     return token_to_ix, pretrained_emb


def tokenize(fact_list, stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    for facts in fact_list:
        for fact in facts:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                fact['text'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb

# def ans_stat(stat_ans_list, ans_freq):
#     ans_to_ix = {}
#     ix_to_ans = {}
#     ans_freq_dict = {}
#
#     for ans in stat_ans_list:
#         ans_proc = prep_ans(ans['multiple_choice_answer'])
#         if ans_proc not in ans_freq_dict:
#             ans_freq_dict[ans_proc] = 1
#         else:
#             ans_freq_dict[ans_proc] += 1
#
#     ans_freq_filter = ans_freq_dict.copy()
#     for ans in ans_freq_dict:
#         if ans_freq_dict[ans] <= ans_freq:
#             ans_freq_filter.pop(ans)
#
#     for ans in ans_freq_filter:
#         ix_to_ans[ans_to_ix.__len__()] = ans
#         ans_to_ix[ans] = ans_to_ix.__len__()
#
#     return ans_to_ix, ix_to_ans


def ans_stat(json_file):
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

    return ans_to_ix, ix_to_ans


# ------------------------------------
# ---- Real-Time Processing Utils ----
# ------------------------------------

def proc_img_feat(img_feat, img_feat_pad_size):
    img_feat_pad_size = 100
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def proc_fact(top_facts, token_to_ix, max_token, single_token):
    # max_token = 250
    ques_ix = np.zeros(max_token, np.int64)

    # 50 facts in top_facts
    for idx, fact in enumerate(top_facts):

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            fact['text'].lower()
        ).replace('-', ' ').replace('/', ' ').replace('paddd', 'PAD').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[idx*single_token+ix] = token_to_ix[word]
            else:
                ques_ix[idx*single_token+ix] = token_to_ix['UNK']

            if ix + 1 == single_token:
                break

    return ques_ix


# def proc_fact(top_facts, token_to_ix, max_token):
#     fact_str = ''
#     for idx, fact in enumerate(top_facts):
#         # word is str
#         word = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             fact['text'].lower()
#         ).replace('-', ' ').replace('/', ' ')
#         # str to list
#         words = word.split()
#
#         if len(words) > max_token:
#             words_max_len = words[-max_token:]
#             fact_max_len = ' '.join([w for w in words_max_len])
#         else:
#             fact_max_len = word
#         fact_max_len += '. '
#         fact_str += fact_max_len
#
#     return fact_str


def get_top_obj(objects, top=10):
    bad_object = ['headlights', 'pastries', 'bangs', 'rims', 'receipt', 'handlebars', 'moped',
                  'suitcases', 'dvds', 'lunch', 'sandwhich', 'skies', 'tshirt', 'skiis',
                  'kneepad', 'backsplash', 'laptop', 'remotes', 'hoodie', 'wetsuit', 'wipers',
                  'stickers', 'fixtures', 'flops', 'iphone', 'urinals', 'skatepark', 'suspenders',

                  'headlights', 'extinguisher', 'fries', 'pastries', 'eagle', 'shack', 'blazer', 'bangs', 'rims',
                  'receipt', 'jet', 'handlebars', 'mitt', 'motorbike', 'bouquet', 'calf', 'mannequin', 'cheek',
                  'graffiti', 'lettering', 'tarp', 'moped', 'outlet', 'keypad', 'suitcases', 'suv', 'dvds', 'lunch',
                  'sandwhich', 'remote', 'skies', 'tshirt', 'skiis', 'kneepad', 'backsplash', 'laptop', 'seagull',
                  'remotes', 'shorts', 'sneaker', 'hoodie', 'wetsuit', 'trolley', 'storefront', 'vent', 'spots',
                  'lighthouse', 'wipers', 'cub', 'stickers', 'croissant', 'fixtures', 'collar', 'bandana', 'uniform',
                  'flops', 'iphone', 'splash', 'urinals', 'undershirt', 'skatepark', 'station', 'latch', 'jersey',
                  'sprinkles', 'suspenders', 'logo', 'stroller',
                  ]
    objects = [obj for obj in objects if obj not in bad_object]
    count_objects = Counter(objects)
    top_obj_list = count_objects.most_common(top)
    real_top = len(top_obj_list)
    top_obj = []
    for i in range(real_top):
        top_obj.append(top_obj_list[i][0])

    return top_obj


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.


def proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score

