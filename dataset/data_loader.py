import json, torch, os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score


class DataLoader_ERC(Dataset):
    def __init__(self, dataset, d_type='multi', desc='train') -> None:
        self.d_type = d_type
        self.samples = dataset.datas['data'][desc]
        self.batch_cols = dataset.batch_cols
        self.tokenizer_ = dataset.tokenizer_
        self.tokenizer = dataset.tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, output = self.samples[idx], {}
        for col, pad in self.keys.items():
            if col == 'audio':
                wav, _sr = sf.read(sample['audio'])
                output[col] = torch.tensor(wav.astype(np.float32))[0:160000]
            elif col == 'label':
                output[col] = torch.tensor(self.ltoi[sample[col]])
            else:
                output[col] = sample[col]
        return output


class ERCDataset_Multi(Dataset):
    def __init__(self, data_dir, batch_size=None, num_workers=8):
        super().__init__()
        self.name = ['erc', data_dir.split('/')[-2]]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_init() # 初始化容器信息
        self.prepare_data(stages=['train', 'valid', 'test'])
        self.num_classes = len(self.tokenizer_['labels']['ltoi'])
        
    def _score(self, results, stage='train'):
        preds = np.concatenate([rec['preds'].cpu().numpy() for rec in results])
        truthes = np.concatenate([rec['labels'].cpu().numpy() for rec in results])
        losses = [rec['loss'].item() for rec in results]

        score_f1 = round(f1_score(truthes, preds, average='weighted'), 4)
        score_acc = round(accuracy_score(truthes, preds), 4)
        score_loss = round(sum(losses)/len(losses), 3)

        if 'ddg' in self.name: # 去掉 neutral
            neutral_id = self.tokenizer_['labels']['ltoi']['none']
            labels_sel = list(range(self.num_classes))
            labels_sel.pop(labels_sel.index(neutral_id))
            score_f1 = round(f1_score(truthes, preds, labels=labels_sel, average='micro'), 4) # 计算指定标签的 f1_score
            score_acc = round(accuracy_score(truthes, preds), 4)

        return {
            'f1'  : score_f1,
            'acc' : score_acc,
            'loss': score_loss
        }

    def dataset_init(self, only='all'):
        # 初始化数据集要保存的内容
        self.info = {
            'num_conv': {'train': 0, 'valid': 0, 'test': 0},              # number of convs
            'num_conv_speaker': {'train': [], 'valid': [], 'test': []},   # number of speakers in each conv
            'num_conv_utt': {'train': [], 'valid': [], 'test': []},       # number of utts in each conv
            'num_conv_utt_token': {'train': [], 'valid': [], 'test': []}, # number of tokens in each utt

            'num_samp': {'train': 0, 'valid': 0, 'test': 0},            # number of reconstructed samples
            'num_samp_token': {'train': [], 'valid': [], 'test': []},   # number of tokens in each sample
            'emotion_category': {},                                     # n_class
        }
        
        # 映射字典
        path_tokenizer_ = self.data_dir + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}},   # label 字典
                'speakers': { 'stoi': {}, 'itos': {}, 'count': {}}, # speaker 字典
            }
            self.path_tokenizer_ = path_tokenizer_

        # 数据集
        self.datas = {}
        self.loader = {}

        # 评价指标
        self.met, default = 'f1', 0 # 主要评价指标
        self.metrics = {
            'train': { self.met: default, 'loss': default }, 
            'valid': { self.met: default }, 
            'test':  { self.met: default }
            } # 训练指标

    def speaker_label(self, speakers, labels):
        if self.path_tokenizer_ is None: return -1 # 已经加载好了

        # 记录speaker信息
        for speaker in speakers:
            if speaker not in self.tokenizer_['speakers']['stoi']: # 尚未记录
                self.tokenizer_['speakers']['stoi'][speaker] = len(self.tokenizer_['speakers']['stoi'])
                self.tokenizer_['speakers']['itos'][len(self.tokenizer_['speakers']['itos'])] = speaker
                self.tokenizer_['speakers']['count'][speaker] = 0
            self.tokenizer_['speakers']['count'][speaker] += 1

        # 记录label信息
        for label in labels:
            if label is None: continue
            if label not in self.tokenizer_['labels']['ltoi']:
                self.tokenizer_['labels']['ltoi'][label] = len(self.tokenizer_['labels']['ltoi'])
                self.tokenizer_['labels']['itol'][len(self.tokenizer_['labels']['itol'])] = label
                self.tokenizer_['labels']['count'][label] = 0
            self.tokenizer_['labels']['count'][label] += 1

    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path = f'{self.data_dir}/{stage}.raw.json'
            with open(raw_path, 'r', encoding='utf-8') as fp: raw_convs, convs = json.load(fp), []
            self.info['num_conv'][stage] = len(raw_convs)

            for ci, r_conv in enumerate(raw_convs):
                txts, spks, labs = [], [], []
                for utt in r_conv:
                    txt, spk, lab = utt['text'].strip(), utt['speaker'].strip(), utt.get('label')
                    txts.append(txt)
                    spks.append(spk)
                    labs.append(lab)

                    if spk not in self.info['num_conv_speaker'][stage]: self.info['num_conv_speaker'][stage].append(spk)

                assert len(txts) == len(spks) == len(labs)
                self.speaker_label(spks, labs) # tokenizer_ (speakers/labels)
                convs.append({
                    'idx': len(convs),
                    'texts': txts,
                    'speakers': spks,
                    'emotions': labs,
                })
                self.info['num_conv_utt'][stage].append(len(txts))
                self.info['num_conv_utt_token'][stage].append([len(txt.split()) for txt in txts])

                self.info['num_samp'][stage] += len(txts)

            self.datas[stage] = convs

    def get_dataloader(self, batch_size=None):
        if batch_size: self.batch_size = batch_size
        for stage, _ in self.datas.items():
            if stage=='train': self.loader[stage] = self.train_dataloader()
            if stage=='valid': self.loader[stage] = self.val_dataloader()
            if stage=='test':  self.loader[stage] = self.test_dataloader()
        return self.loader

    def train_dataloader(self):
        return DataLoader(
            self.datas['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datas['valid'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, dialogs):
        max_token_num = max([max(dialog['attention_mask'].sum(dim=-1)) for dialog in dialogs])
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'index' in col: 
                temp = torch.tensor([dialog[col] for dialog in dialogs])
            if 'ids' in col or 'mask' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)[:,:,0:max_token_num]
            if 'speakers' in col or 'labels' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
            inputs[col] = temp

        return inputs

class ERCDataset_Multi_Old(Dataset):
    def __init__(self, path, tokenizer=None, lower=False):
        self.path = path
        self.lower = lower
        self.name = ['erc', path.split('/')[-2]]
        self.container_init() # 初始化容器信息
        self.get_dataset()    # 解析数据集
        self.n_class = len(self.tokenizer_['labels']['ltoi']) 
        self.max_seq_len = max_seq_lens[self.name[-1]] # 最大句子长度
        self.type = 'cls' # 分类任务

    def container_init(self, only='all'):
        # 初始化数据集要保存的内容
        self.info = {
            'num_conv': {'train': 0, 'valid': 0, 'test': 0},              # number of convs
            'num_conv_speaker': {'train': [], 'valid': [], 'test': []},   # number of speakers in each conv
            'num_conv_utt': {'train': [], 'valid': [], 'test': []},       # number of utts in each conv
            'num_conv_utt_token': {'train': [], 'valid': [], 'test': []}, # number of tokens in each utt

            'num_samp': {'train': 0, 'valid': 0, 'test': 0},            # number of reconstructed samples
            'num_samp_token': {'train': [], 'valid': [], 'test': []},   # number of tokens in each sample
            'emotion_category': {},                                     # n_class
        }
        
        # 映射字典
        path_tokenizer_ = self.path + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}},   # label 字典
                'speakers': { 'stoi': {}, 'itos': {}, 'count': {}}, # speaker 字典
            }
            self.path_tokenizer_ = path_tokenizer_

        # 数据集
        self.datas = {'data': {}, 'loader': {}}

    def speaker_label(self, speakers, labels):
        if self.path_tokenizer_ is None: return -1 # 已经加载好了

        # 记录speaker信息
        for speaker in speakers:
            if speaker not in self.tokenizer_['speakers']['stoi']: # 尚未记录
                self.tokenizer_['speakers']['stoi'][speaker] = len(self.tokenizer_['speakers']['stoi'])
                self.tokenizer_['speakers']['itos'][len(self.tokenizer_['speakers']['itos'])] = speaker
                self.tokenizer_['speakers']['count'][speaker] = 1
            self.tokenizer_['speakers']['count'][speaker] += 1

        # 记录label信息
        for label in labels:
            if label is None: continue
            if label not in self.tokenizer_['labels']['ltoi']:
                self.tokenizer_['labels']['ltoi'][label] = len(self.tokenizer_['labels']['ltoi'])
                self.tokenizer_['labels']['itol'][len(self.tokenizer_['labels']['itol'])] = label
                self.tokenizer_['labels']['count'][label] = 1
            self.tokenizer_['labels']['count'][label] += 1

    def get_dataset(self):
        for desc in ['train', 'valid', 'test']:
            raw_path = f'{self.path}/{desc}.raw.json'
            with open(raw_path, 'r', encoding='utf-8') as fp:
                raw_convs, convs = json.load(fp), []
            self.info['num_conv'][desc] = len(raw_convs)

            for ci, r_conv in enumerate(raw_convs):
                txts, spks, labs = [], [], []
                for utt in r_conv:
                    txt, spk, lab = utt['text'].strip(), utt['speaker'].strip(), utt.get('label')
                    txts.append(txt)
                    spks.append(spk)
                    labs.append(lab)

                self.speaker_label(spks, labs) # tokenizer_ (speakers/labels)
                convs.append({
                    'idx': len(convs),
                    'texts': txts,
                    'speakers': spks,
                    'emotions': labs,
                })
            self.datas['data'][desc] = convs

    def get_dataloader(self, batch_size, shuffle=None, only=None):
        if shuffle is None:
            shuffle = {'train': True, 'valid': False, 'test': False}

        dataloader = {}
        for desc, data_embed in self.datas['data'].items():
            if only is not None and desc!=only: continue
            dataloader[desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=self.collate_fn)
            
        return dataloader

    def collate_fn(self, dialogs):
        max_token_num = max([max(dialog['attention_mask'].sum(dim=-1)) for dialog in dialogs])
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'index' in col: 
                temp = torch.tensor([dialog[col] for dialog in dialogs])
            if 'ids' in col or 'mask' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)[:,:,0:max_token_num]
            if 'speakers' in col or 'labels' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
            inputs[col] = temp

        return inputs


class ERCDataset_Single(ERCDataset_Multi):
    def get_vector(self, args=None, tokenizer=None, method='tail', only=None):
        speaker_fn, label_fn = self.speakers['ntoi'], self.labels['ltoi']
        if args.anonymity: 
            tokenizer = self.refine_tokenizer(tokenizer) # 更新字典
            speaker_fn = self.speakers['atoi']
            
        self.args, self.tokenizer = args, tokenizer
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                embedding = tokenizer(item['text'], return_tensors='pt')
                input_ids, attention_mask = self.vector_truncate(embedding, method='first')
                speaker, label = speaker_fn[item['speaker']], label_fn[item['label']]
                item_embed = {
                    'index': item['index'],
                    'input_ids': input_ids.squeeze(dim=0),
                    'attention_mask': attention_mask.squeeze(dim=0),
                    'speaker': torch.tensor(speaker),
                    'label': torch.tensor(label),
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def get_specific_dataset(args, d_type='multi'):
    ## 1. 导入数据
    data_path = args.file['data'] + f"{args.train['tasks'][1]}/"
    if d_type == 'multi': 
        dataset = ERCDataset_Multi(data_path, lower=True)
        dataset.batch_cols = {'idx': -1, 'texts': -1, 'speakers': -1, 'labels': -1 }
    else: 
        dataset = ERCDataset_Single(data_path, lower=True)
        dataset.batch_cols = {'idx': -1, 'texts': -1, 'speakers': -1, 'labels': -1 }

    dataset.tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    for desc, data in dataset.datas['data'].items():
        dataset.datas['data'][desc] = DataLoader_ERC(
            dataset,
            d_type=d_type,
            desc=desc
        )
    dataset.task = 'cls'

    return dataset


class ERCDataset_Multi_(Dataset):
    def __init__(self, data_dir, batch_size=None, num_workers=8):
        super().__init__()
        self.name = ['erc', data_dir.split('/')[-2]]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_init() # 初始化容器信息
        self.prepare_data(stages=['train', 'valid', 'test'])
        self.num_classes = len(self.tokenizer_['labels']['ltoi'])

        self.max_seq_len = max_seq_lens[self.name[-1]] # 最大句子长度
        self.task = 'cls' # 分类任务

    def dataset_init(self, only='all'):
        # 初始化数据集要保存的内容
        self.info = {
            'num_conv': {'train': 0, 'valid': 0, 'test': 0},              # number of convs
            'num_conv_speaker': {'train': [], 'valid': [], 'test': []},   # number of speakers in each conv
            'num_conv_utt': {'train': [], 'valid': [], 'test': []},       # number of utts in each conv
            'num_conv_utt_token': {'train': [], 'valid': [], 'test': []}, # number of tokens in each utt

            'num_samp': {'train': 0, 'valid': 0, 'test': 0},            # number of reconstructed samples
            'num_samp_token': {'train': [], 'valid': [], 'test': []},   # number of tokens in each sample
            'emotion_category': {},                                     # n_class
        }
        
        # 映射字典
        path_tokenizer_ = self.data_dir + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}},   # label 字典
                'speakers': { 'stoi': {}, 'itos': {}, 'count': {}}, # speaker 字典
            }
            self.path_tokenizer_ = path_tokenizer_

        # 数据集
        self.datas = {'train': []}

    def speaker_label(self, speakers, labels):
        if self.path_tokenizer_ is None: return -1 # 已经加载好了

        # 记录speaker信息
        for speaker in speakers:
            if speaker not in self.tokenizer_['speakers']['stoi']: # 尚未记录
                self.tokenizer_['speakers']['stoi'][speaker] = len(self.tokenizer_['speakers']['stoi'])
                self.tokenizer_['speakers']['itos'][len(self.tokenizer_['speakers']['itos'])] = speaker
                self.tokenizer_['speakers']['count'][speaker] = 0
            self.tokenizer_['speakers']['count'][speaker] += 1

        # 记录label信息
        for label in labels:
            if label is None: continue
            if label not in self.tokenizer_['labels']['ltoi']:
                self.tokenizer_['labels']['ltoi'][label] = len(self.tokenizer_['labels']['ltoi'])
                self.tokenizer_['labels']['itol'][len(self.tokenizer_['labels']['itol'])] = label
                self.tokenizer_['labels']['count'][label] = 0
            self.tokenizer_['labels']['count'][label] += 1

    def prepare_data(self, stages=['train', 'valid', 'test']):
        for stage in stages:
            raw_path = f'{self.data_dir}/{stage}.raw.json'
            with open(raw_path, 'r', encoding='utf-8') as fp: raw_convs, convs = json.load(fp), []
            self.info['num_conv'][stage] = len(raw_convs)

            for ci, r_conv in enumerate(raw_convs):
                txts, spks, labs = [], [], []
                for utt in r_conv:
                    txt, spk, lab = utt['text'].strip(), utt['speaker'].strip(), utt.get('label')
                    txts.append(txt)
                    spks.append(spk)
                    labs.append(lab)

                    if spk not in self.info['num_conv_speaker'][stage]: self.info['num_conv_speaker'][stage].append(spk)

                assert len(txts) == len(spks) == len(labs)
                self.speaker_label(spks, labs) # tokenizer_ (speakers/labels)
                convs.append({
                    'idx': len(convs),
                    'texts': txts,
                    'speakers': spks,
                    'emotions': labs,
                })
                self.info['num_conv_utt'][stage].append(len(txts))
                self.info['num_conv_utt_token'][stage].append([len(txt.split()) for txt in txts])

                self.info['num_samp'][stage] += len(txts)

            self.datas[stage] = convs

    def train_dataloader(self):
        return DataLoader(
            self.datas['train'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datas['valid'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datas['test'], 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, dialogs):
        max_token_num = max([max(dialog['attention_mask'].sum(dim=-1)) for dialog in dialogs])
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'index' in col: 
                temp = torch.tensor([dialog[col] for dialog in dialogs])
            if 'ids' in col or 'mask' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)[:,:,0:max_token_num]
            if 'speakers' in col or 'labels' in col:
                temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
            inputs[col] = temp

        return inputs


if __name__ == "__main__":
    data_dir = f'./iec/'
    dataset = ERCDataset_Multi_PL(data_dir)
    
    plm_path = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(plm_path)
    dataset.get_vector(tokenizer, truncate=None)

    input()