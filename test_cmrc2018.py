import os
import argparse
import json
import torch
import utils
from src.pytorch_modeling import BertConfig, BertForQuestionAnswering
from src.cmrc2018_output import write_predictions
from src.cmrc2018_evaluate import get_eval
import collections
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import src.official_tokenization as tokenization
from src.cmrc2018_preprocess import json2features


def evaluate(model, args, eval_examples, eval_features, device, global_steps,
             best_f1, best_em, best_f1_em, is_test=False):
    print("***** Eval *****")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    output_prediction_file = os.path.join(args.checkpoint_dir,
                                          "predictions_steps" + str(global_steps) + ".json")
    output_nbest_file = output_prediction_file.replace('predictions', 'nbest')

    all_input_ids = torch.tensor([f['input_ids'] for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.n_batch, shuffle=False)

    model.eval()
    all_results = []
    print("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature['unique_id'])
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    write_predictions(eval_examples, eval_features, all_results,
                      n_best_size=args.n_best, max_answer_length=args.max_ans_length,
                      do_lower_case=True, output_prediction_file=output_prediction_file,
                      output_nbest_file=output_nbest_file)

    tmp_result = get_eval(args.dev_file if not is_test else args.test_file, output_prediction_file)
    tmp_result['STEP'] = global_steps
    with open(args.log_file, 'a') as aw:
        aw.write(json.dumps(tmp_result) + '\n')
    print(tmp_result)

    if not is_test:
        if float(tmp_result['F1']) > best_f1:
            best_f1 = float(tmp_result['F1'])

        if float(tmp_result['EM']) > best_em:
            best_em = float(tmp_result['EM'])

        if float(tmp_result['F1']) + float(tmp_result['EM']) > best_f1_em:
            best_f1_em = float(tmp_result['F1']) + float(tmp_result['EM'])
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.checkpoint_dir + '/best_model.pth')

        model.train()

        return best_f1, best_em, best_f1_em


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1')

    # training parameter
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--float16', type=bool, default=True)  # only sm >= 7.0 (tensorcores)
    parser.add_argument('--max_ans_length', type=int, default=50)
    parser.add_argument('--n_best', type=int, default=20)
    parser.add_argument('--eval_epochs', type=float, default=0.5)
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--vocab_size', type=int, default=21128)  # 21128, 8021

    # data dir
    parser.add_argument('--train_dir', type=str,
                        default='data/cmrc2018/train_features.json')
    parser.add_argument('--train_file', type=str,
                        default='data/cmrc2018/cmrc2018_train.json')
    parser.add_argument('--dev_dir', type=str,
                        default='data/cmrc2018/trial_features.json')
    parser.add_argument('--dev_file', type=str,
                        default='data/cmrc2018/cmrc2018_trial.json')
    parser.add_argument('--test_dir', type=str,
                        default='data/cmrc2018/dev_features.json')
    parser.add_argument('--test_file', type=str,
                        default='data/cmrc2018/cmrc2018_dev.json')
    parser.add_argument('--bert_config_file', type=str,
                        default='pretrained_models/bert-wwm-ext/bert_config.json')
    parser.add_argument('--vocab_file', type=str,
                        default='pretrained_models/bert-wwm-ext/vocab.txt')
    parser.add_argument('--init_restore_dir', type=str,
                        default='check_points/roberta_cmrc2018_V1/best_model.pth')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='check_points/roberta_cmrc2018_V1/')
    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')

    # use some global vars for convenience
    args = parser.parse_args()
    # utils.check_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.float16))

    # load the bert setting
    bert_config = BertConfig.from_json_file(args.bert_config_file)

    # load data
    print('loading data...')
    tokenizer = tokenization.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    assert args.vocab_size == len(tokenizer.vocab)

    if not os.path.exists(args.test_dir):
        json2features(args.test_file, [args.test_dir.replace('_features', '_examples'), args.test_dir],
                      tokenizer, is_training=False,
                      max_seq_length=bert_config.max_position_embeddings)

    test_examples = json.load(open(args.test_dir.replace('_features', '_examples'), 'r'))
    test_features = json.load(open(args.test_dir, 'r'))
    # if os.path.exists(args.log_file):
    #     os.remove(args.log_file)

    # init model
    print('init model...')
    model = BertForQuestionAnswering(bert_config)
    utils.torch_init_model(model, args.init_restore_dir, key='model')
    model.to(device)

    if args.float16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level='O1')

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # test
    print('Testing...')
    evaluate(model, args, test_examples, test_features, device,
             0, 0, 0, 0, is_test=True)
