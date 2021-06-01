import random
import os
import argparse
import numpy as np
import json
import torch
import utils
from src.pytorch_modeling import BertForUniLM, BertConfig
from src.pytorch_optimization import AdamW, get_linear_schedule_with_warmup
from src.text_shuffle_preprocess import get_features
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import src.official_tokenization as tokenization
import time
import copy
import pickle
from rouge import Rouge

rouge = Rouge()

SPIECE_UNDERLINE = '▁'

tokenizer = tokenization.BertTokenizer(vocab_file='pretrained_models/mtsn_base/vocab_13806.txt',
                                       do_lower_case=True)


def data_syn(features):
    for i, feature in enumerate(features):
        # 验证集同步
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


def is_fuhao(c):
    if c in ('。', '，', ',', '！', '!', '？', '?', '；', ';', '、', '：', ':', '（', '(', '）', ')', '－', '~', '～',
             '「', '」', '《', '》', '"', '“', '”', '$', '『', '』', '—', '-', '‘', '’', '\'', '[', '【',
             ']', '】', '=', '|', '<', '>'):
        return True
    return False


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or is_fuhao(char):
            if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                output.append(SPIECE_UNDERLINE)
            output.append(char)
            output.append(SPIECE_UNDERLINE)
        else:
            output.append(char)
    return "".join(output)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == SPIECE_UNDERLINE:
        return True
    return False


def split_tokens(text):
    context_chs = _tokenize_chinese_chars(text)
    doc_tokens = []
    prev_is_whitespace = True
    for c in context_chs:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens


def get_score(y_true, y_pred):
    split_tokens_true = " ".join(split_tokens(y_true)).lower()
    split_tokens_pred = " ".join(split_tokens(y_pred)).lower()

    try:
        rouge_score = rouge.get_scores(split_tokens_pred, split_tokens_true)[0]['rouge-l']['f']
    except:
        rouge_score = 0

    return rouge_score * 100


def get_ter_score(y_true, y_pred):
    dp = [[0 for _ in range(len(y_pred) + 1)] for _ in range(len(y_true) + 1)]
    for i in range(1, len(dp)):
        dp[i][0] = i
    for i in range(1, len(dp[0])):
        dp[0][i] = i
    for i in range(1, len(dp)):
        for j in range(1, len(dp[i])):
            if y_true[i - 1] == y_pred[j - 1]:
                d = 0
            else:
                d = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + d)
    ter = dp[-1][-1] / (len(y_true) + 1e-5)
    return -dp[-1][-1]


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


def batch_evaluate(model, device, dev_features, dev_steps, result_dir, best_ter=0, END_id=99):
    ters = []
    with torch.no_grad():
        with open(result_dir, 'w') as w:
            with tqdm(total=dev_steps, desc='Evaluating') as pbar:
                for i_step in range(dev_steps):
                    origin_batch_features = copy.deepcopy(dev_features[i_step * args.eval_batch_size:
                                                                       (i_step + 1) * args.eval_batch_size])
                    batch_features = list(origin_batch_features)

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
                                     'output_idx': output_idx_np}

                        pred_logits = model(**feed_data)
                        pred_logits = pred_logits.cpu().numpy()
                        pred_ids = np.argmax(pred_logits, axis=1)  # [bs',]
                        new_batch_features = []
                        for j in range(len(batch_features)):
                            batch_idx = batch_features[j]['batch_idx']
                            target = batch_features[j]['output_idxs'][0] + 1  # 生成下一个字,target偏移
                            origin_batch_features[batch_idx]['pred_ids'].append(pred_ids[j])
                            if target < args.max_seq_length and pred_ids[j] != END_id:  # 如果未到长度上限，且不是[END]
                                batch_features[j]['input_ids'][target] = pred_ids[j]
                                batch_features[j]['input_mask'][target] = 1
                                batch_features[j]['segment_ids'][target] = 1
                                batch_features[j]['output_idxs'] = [target]
                                new_batch_features.append(batch_features[j])
                        batch_features = new_batch_features

                    for i in range(len(origin_batch_features)):
                        pred_tokens = tokenizer.convert_ids_to_tokens(origin_batch_features[i]['pred_ids'])
                        pred_text = []
                        for token in pred_tokens:
                            if token == '[END]':
                                break
                            pred_text.append(token)

                        pred_text = combine_tokens(pred_text)
                        pred_text = " ".join(pred_text.split())
                        true_text = origin_batch_features[i]['output_text']

                        sample = {'true_text': true_text, 'pred_text': pred_text}
                        w.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        if true_text is None or pred_text is None:
                            print('true_text', true_text)
                            print('pred_text', pred_text)
                        ter_score = get_ter_score(true_text, pred_text)
                        ters.append(ter_score)

                    pbar.set_postfix({'TERs': '{:.3f}'.format(np.mean(ters))})
                    pbar.update(1)

    if np.mean(ters) > best_ter:
        best_ter = np.mean(ters)
        model_to_save = model.module if hasattr(model, 'module') else model
        save_dict = {'model':model_to_save.state_dict(),
                     'opt':optimizer.state_dict()}
        torch.save(save_dict, args.checkpoint_dir + '/best_model.pth')
    return best_ter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.035)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--float16', type=bool, default=True)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--vocab_size', type=int, default=13806)  # 21128, 8021, 13806

    # data dir
    parser.add_argument('--train_dir', type=str, default='data/text_shuffle/train.json')
    parser.add_argument('--dev_dir', type=str, default='data/text_shuffle/test_with_answer.json')
    parser.add_argument('--train_feat_dir', type=str, default='data/text_shuffle/train_feat.pkl')
    parser.add_argument('--bert_config_file', type=str, default='pretrained_models/mtsn_base/mtsn_base_config.json')
    parser.add_argument('--init_restore_dir', type=str, default='pretrained_models/mtsn_base/model-250k.pth')
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
    # load data
    if not os.path.exists(args.train_feat_dir):
        train_features = get_features(json.load(open(args.train_dir)),
                                      tokenizer, max_seq_length=512,
                                      max_output_length=256, is_train=True)
        with open(args.train_feat_dir, 'wb') as w:
            pickle.dump(train_features, w)
    else:
        train_features = pickle.load(open(args.train_feat_dir, 'rb'))
    dev_features = get_features(json.load(open(args.dev_dir)),
                                tokenizer, max_seq_length=512,
                                max_output_length=256, is_train=False)
    random.seed(123)
    random.shuffle(dev_features)
    dev_features = dev_features[:args.eval_batch_size * 4]

    data_syn(dev_features)
    train_steps = len(train_features) // args.batch_size
    dev_steps = len(dev_features) // args.eval_batch_size
    if len(train_features) % args.batch_size != 0:
        train_steps += 1
    if len(dev_features) % args.eval_batch_size != 0:
        dev_steps += 1
    total_steps = train_steps * args.train_epochs

    print('steps per epoch:', train_steps)
    print('total steps:', total_steps)
    print('warmup steps:', int(args.warmup_rate * total_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # init model
    print('init model...')

    model = BertForUniLM(bert_config)
    utils.torch_init_model(model, args.init_restore_dir)
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_rate,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # get the optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(args.warmup_rate * total_steps),
                                                num_training_steps=total_steps)

    if args.float16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)
    all_context_mask = torch.tensor([f['context_mask'] for f in train_features], dtype=torch.long)
    all_output_idxs = torch.tensor([f['output_idxs'] for f in train_features], dtype=torch.long)
    all_output_ids = torch.tensor([f['output_ids'] for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_context_mask, all_output_idxs, all_output_ids)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    print('***** Training *****')
    model.train()
    global_steps = 0
    best_ter = -1000
    for i in range(int(args.train_epochs)):
        print('Starting epoch %d' % (i + 1))
        start_time = time.time()
        loss_values = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, context_mask, output_idx, output_ids = batch
            loss = model(input_ids=input_ids,
                         attention_mask=input_mask,
                         token_type_ids=segment_ids,
                         context_mask=context_mask,
                         output_idx=output_idx,
                         output_ids=output_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            loss_values.append(loss.item())
            if args.float16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.float16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_steps += 1

            if global_steps % args.log_interval == 0:
                show_str = 'Epoch:{}, Steps:{}/{}, Loss:{:.4f}'.format(i + 1, global_steps, total_steps,
                                                                       np.mean(loss_values))
                if global_steps > 1:
                    remain_seconds = (time.time() - start_time) * ((train_steps - step) / (step + 1e-5))
                    m, s = divmod(remain_seconds, 60)
                    h, m = divmod(m, 60)
                    remain_time = " remain:%02d:%02d:%02d" % (h, m, s)
                    show_str += remain_time
                print(show_str)

            if global_steps % args.save_interval == 0:
                # evaluate
                model.eval()
                result_dir = args.checkpoint_dir + '/pred_step' + str(global_steps) + '.json'
                best_ter = batch_evaluate(model, device, dev_features, dev_steps, result_dir, best_ter)
                model.train()

        # evaluate
        model.eval()
        result_dir = args.checkpoint_dir + '/pred_step' + str(global_steps) + '.json'
        best_ter = batch_evaluate(model, device, dev_features, dev_steps, result_dir, best_ter)
        model.train()
        model_to_save = model.module if hasattr(model, 'module') else model
        save_dict = {'model':model_to_save.state_dict(),
                     'opt':optimizer.state_dict()}
        torch.save(save_dict, args.checkpoint_dir + '/last_model.pth')

    print('Best TER:', best_ter)
