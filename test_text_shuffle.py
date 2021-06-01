import random
import os
import argparse
import numpy as np
from src.text_shuffle_preprocess import get_features
import json
import torch
import utils
from src.pytorch_modeling import BertForUniLMInference, BertConfig
from tqdm import tqdm
import src.official_tokenization as tokenization
import copy

SPIECE_UNDERLINE = '▁'

tokenizer = tokenization.BertTokenizer(vocab_file='pretrained_models/mtsn_base/vocab_13806.txt',
                                       do_lower_case=True)


def data_syn(features):
    for i, feature in enumerate(features):
        # 验证集记录其预测id和batch内对应的位置
        feature['batch_idx'] = i % args.eval_batch_size
        feature['pred_ids'] = []


def _is_chinese_char(cp):
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


# 判断是否是符号
def is_fuhao(c):
    if c in ('。', '，', ',', '！', '!', '？', '?', '；', ';', '、', '：', ':', '（', '(', '）', ')', '－', '~', '～',
             '「', '」', '《', '》', '"', '“', '”', '$', '『', '』', '—', '-', '‘', '’', '\'', '[', '【',
             ']', '】', '=', '|', '<', '>'):
        return True
    return False


# 聚合所有token为最终结果，这里主要处理中英文混合的情况
def combine_tokens(tokens):
    output_text = []
    for tok in tokens:
        if tok == '[UNK]':
            continue
        if len(tok) == 1 and _is_chinese_char(ord(tok)):
            output_text.append(tok)
        elif tok.startswith('##'):
            output_text.append(tok.replace('##', ''))
        elif len(output_text) > 0:
            if len(output_text[-1]) == 1 and (_is_chinese_char(ord(output_text[-1])) or is_fuhao(output_text[-1])):
                output_text.append(tok)
            else:
                output_text.append(' ' + tok)
        else:
            output_text.append(tok)

    output_text = "".join(output_text).strip()
    return output_text


# 基于batch的seq2seq推断，单个样本的seq2seq推断会简单很多，但是速度很慢(fp16 torch版本双卡大概XX分钟完成3000样本推断)
def batch_evaluate(model, device, dev_features, dev_steps, result_dir, END_id=99):
    '''
    :param model: model
    :param device: 数据转移到gpu
    :param dev_features: 验证集
    :param dev_steps: 验证步数
    :param result_dir: 输出路径
    :param END_id: 结束终结符，这里使用默认的[END]token
    '''
    samples = []
    with torch.no_grad():
        for i_step in tqdm(range(dev_steps), desc='Evaluating'):
            origin_batch_features = copy.deepcopy(dev_features[i_step * args.eval_batch_size:
                                                               (i_step + 1) * args.eval_batch_size])
            batch_features = list(origin_batch_features)

            # 保证batch内所有样本验证完毕才算结束
            while len(batch_features) > 0:
                input_ids_np = torch.tensor(np.array([f['input_ids'] for f in batch_features]),
                                            dtype=torch.long).to(device=device)
                input_mask_np = torch.tensor(np.array([f['input_mask'] for f in batch_features]),
                                             dtype=torch.long).to(device=device)
                context_mask_np = torch.tensor(np.array([f['context_mask'] for f in batch_features]),
                                               dtype=torch.long).to(device=device)
                segment_ids_np = torch.tensor(np.array([f['segment_ids'] for f in batch_features]),
                                              dtype=torch.long).to(device=device)
                output_idx_np = torch.tensor(np.array([f['output_idxs'] for f in batch_features]),
                                             dtype=torch.long).to(device=device)
                feed_data = {'input_ids': input_ids_np,
                             'attention_mask': input_mask_np,
                             'context_mask': context_mask_np,
                             'token_type_ids': segment_ids_np,
                             'output_idx': output_idx_np}  # [bs, 1] 表示当前要得到的token表示的位置(一般开始是context最后一个token或分隔符sep)

                # 由于样本完成就会被剔除batch，所以bs'<=bs (batch size)
                pred_logits = model(**feed_data)  # [bs', vocab_size]
                pred_logits = pred_logits.cpu().numpy()
                pred_ids = np.argmax(pred_logits, axis=1)  # [bs',]
                new_batch_features = []
                for j in range(len(batch_features)):
                    batch_idx = batch_features[j]['batch_idx']
                    target = batch_features[j]['output_idxs'][0] + 1  # 生成下一个字,target偏移
                    origin_batch_features[batch_idx]['pred_ids'].append(pred_ids[j])
                    if target < args.max_seq_length and pred_ids[j] != END_id:  # 如果未到长度上限，且不是[END], 继续生成
                        batch_features[j]['input_ids'][target] = pred_ids[j]
                        batch_features[j]['input_mask'][target] = 1
                        batch_features[j]['segment_ids'][target] = 1
                        batch_features[j]['output_idxs'] = [target]
                        new_batch_features.append(batch_features[j])
                batch_features = new_batch_features

            # 生成完了id转token
            for i in range(len(origin_batch_features)):
                pred_tokens = tokenizer.convert_ids_to_tokens(origin_batch_features[i]['pred_ids'])
                pred_text = []
                for token in pred_tokens:
                    if token == '[END]':
                        break
                    pred_text.append(token)

                pred_text = combine_tokens(pred_text)  # 拼接tokens
                pred_text = " ".join(pred_text.split())  # 你需要这步，信我

                sample = {'generate': pred_text, 'idx': int(origin_batch_features[i]['idx'])}
                samples.append(sample)

    with open(result_dir, 'w') as w:
        json.dump(samples, w, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')

    # training parameter
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--float16', type=bool, default=True)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vocab_size', type=int, default=13806)  # 21128, 8021, 13806

    # data dir
    parser.add_argument('--dev_dir', type=str, default='data/text_shuffle/test_with_answer.json')
    parser.add_argument('--bert_config_file', type=str, default='pretrained_models/mtsn_base/mtsn_base_config.json')
    parser.add_argument('--output_dir', type=str, default='check_points/mtsn_base_V0/output_total.json')
    parser.add_argument('--checkpoint_dir', type=str, default='check_points/mtsn_base_V0')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = parser.parse_args()
    utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # load the bert setting
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    bert_config.attention_probs_dropout_prob = args.dropout
    bert_config.hidden_dropout_prob = args.dropout

    # load data
    print('loading data...')
    dev_features = get_features(json.load(open(args.dev_dir)),
                                tokenizer, max_seq_length=512,
                                max_output_length=256, is_train=False)
    data_syn(dev_features)
    dev_steps = len(dev_features) // args.eval_batch_size
    if len(dev_features) % args.eval_batch_size != 0:
        dev_steps += 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # init model
    print('init model...')
    model = BertForUniLMInference(bert_config)
    utils.torch_init_model(model, args.checkpoint_dir + '/last_model.pth', key='model')
    model.to(device)

    if args.float16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level='O1')

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # evaluate
    model.eval()
    batch_evaluate(model, device, dev_features, dev_steps, args.output_dir)
