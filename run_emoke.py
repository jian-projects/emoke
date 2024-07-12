
## cuda environment
import warnings, os, wandb, random, sys
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM']='false'

from config import config
from writer import JsonFile
from processor import Processor
from utils_processor import set_rng_seed

def run(args):
    if args.train['wandb']:
        wandb.init(
            project=f"project: {'-'.join(args.train['tasks'])}",
            name=f"{'-'.join(args.train['tasks'])}-seed-{args.train['seed']}",
        )
    set_rng_seed(args.train['seed']) # 固定随机种子

    # import model and dataset
    from Model_EmoKE_ import import_model
    model, dataset = import_model(args)

    # train or eval the model
    processor = Processor(args, model, dataset)
    if args.train['inference']:
        processor.loadState()
        result = processor._evaluate(stage='test')
    else: result = processor._train()
    if args.train['wandb']: wandb.finish()

    ## 2. output results
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['early_stop'],
            'lr':      args.train['learning_rate'],
            'lr_pre':  args.train['learning_rate_pre'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
            'ekl':     args.model['ekl'],
        },
        'metric': {
            'stop':    result['valid']['epoch'],
            'tr_mf1':  result['train'][dataset.met],
            'tv_mf1':  result['valid'][dataset.met],
            'te_mf1':  result['test'][dataset.met],
        },
    }
    return record


if __name__ == '__main__':
    args = config(task='erc', dataset='meld', framework=None, model='emoke')

    ## Parameters Settings
    args.model['scale'] = 'large'
    args.train['device_ids'] = [0]
    
    args.train['epochs'] = 6
    args.train['early_stop'] = 6
    args.train['batch_size'] = 32
    args.train['log_step_rate'] = 1.2
    args.train['log_step_rate_max'] = 3.0
    args.train['learning_rate'] = 3e-4
    args.train['learning_rate_pre'] = 3e-4
    args.train['save_model'] = 0
    args.train['inference'] = 0    
    args.train['do_test'] = True
    args.train['wandb'] = 0 # True   

    args.model['drop_rate'] = 0.1
    args.model['use_lora'] = 1
    args.model['use_rnn'] = 1 # 是否使用 RNN
    args.model['constraints'] = ['ekd', 'eka', 'ekp']
    args.model['setting'] = 'online'
    args.model['layer_num'] = 4
    args.model['ekl'] = 0.3
    
    seeds = []
    if seeds or args.train['inference']: # 按指定 seed 执行
        if not seeds: seeds = [args.train['seed']]
        recoed_path = f"{args.file['record']}{args.model['name']}_best.jsonl"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for seed in seeds:
            args.train['seed'] = seed
            record = run(args)
            record_show.write(record, space=False) 
    else: # 随机 seed 执行       
        recoed_path = f"{args.file['record']}{args.model['name']}_search.jsonl"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for c in range(100):
            args.train['seed'] = random.randint(1000,9999)+c
            record = run(args)
            record_show.write(record, space=False)