from tqdm import tqdm
import numpy as np
import collections


def get_features(data, tokenizer, max_seq_length=512, max_output_length=256, is_train=False):
    features = []
    for d in tqdm(data):
        input_tokens = []
        for tok in d['shuffled_tokens']:
            input_tokens.extend(tokenizer.tokenize(tok))

        length_limit = max_seq_length - 2  # [CLS] OR [END], [SEP]
        output_text = d['origin']
        if is_train:
            output_tokens = tokenizer.tokenize(output_text)[:max_output_length]
            length_limit -= len(output_tokens)
        else:
            output_tokens = []
        input_tokens = input_tokens[:length_limit]

        if is_train:
            total_input_tokens = ['[CLS]'] + input_tokens + ['[SEP]'] + output_tokens
            total_output_tokens = output_tokens + ['[END]']
            output_ids = tokenizer.convert_tokens_to_ids(total_output_tokens)
        else:
            total_input_tokens = ['[CLS]'] + input_tokens + ['[SEP]']
            output_ids = None

        input_ids = tokenizer.convert_tokens_to_ids(total_input_tokens)
        input_mask = [1] * len(total_input_tokens)
        context_mask = [1] * (len(input_tokens) + 2) + [0] * len(output_tokens)
        segment_ids = [0] * (len(input_tokens) + 2) + [1] * len(output_tokens)

        # 记录vocab_mask
        vocab_mask = [99]  # [END]=1
        for id_ in tokenizer.convert_tokens_to_ids(input_tokens):
            vocab_mask.append(id_)

        # output_idx从[SEP]到最后一个output_token
        output_idxs = list(np.arange(len(input_tokens) + 1, len(total_input_tokens)))
        output_idxs = [int(i) for i in output_idxs]

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            context_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(context_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if is_train:
            while len(output_ids) < max_output_length + 1:
                output_ids.append(-1)
                output_idxs.append(0)
            assert len(output_ids) == max_output_length + 1
            assert len(output_idxs) == max_output_length + 1
            token_dict = None
        else:
            token_dict = collections.defaultdict(int)
            for tok in tokenizer.convert_tokens_to_ids(input_tokens):
                token_dict[tok] += 1

        features.append({'idx': d['idx'],
                         'input_ids': input_ids,
                         'input_mask': input_mask,
                         'context_mask': context_mask,
                         'segment_ids': segment_ids,
                         'output_ids': output_ids,
                         'output_idxs': output_idxs,
                         'vocab_mask': vocab_mask,
                         'output_text': output_text,
                         'token_dict': token_dict})

    return features


SPIECE_UNDERLINE = '▁'


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


def token_split(input_text):
    context_chs = _tokenize_chinese_chars(input_text)
    prev_is_whitespace = True
    doc_tokens = []
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
