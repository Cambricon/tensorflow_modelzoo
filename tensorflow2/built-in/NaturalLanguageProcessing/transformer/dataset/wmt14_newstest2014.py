#-*- coding: utf-8 -*-
import sys
import numpy as np
import os
from . import tokenizer as tokenizer

logger = None

json_dict = {
    "batch_size": 1,
    "count":3003
}

class UnicodeRegex(object):
    def __init__(self):
      import re
      punctuation = self.property_chars("P")
      self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
      self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
      self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
      import unicodedata
      import six
      return "".join(six.unichr(x) for x in range(sys.maxunicode)
                   if unicodedata.category(six.unichr(x)).startswith(prefix))

class Newstest2014Encode():
    def __init__(self, item, pre_process=None, preprocess_params=None, calib_num=None, data_root_dir = None):
        self.dataset_root_dir = data_root_dir
        self.dataset_path = self.dataset_root_dir
        encode_file = preprocess_params["encode"]
        decode_file = preprocess_params["decode"]
        vocab = preprocess_params["vocab"]
        self.reference_file = os.path.join(self.dataset_path, decode_file)
        self.input_file = os.path.join(self.dataset_path, encode_file)
        self.output_file = os.path.join(self.dataset_path, "translate.en")
        self.vocab_file = os.path.join(self.dataset_path, vocab)
        self.sub_tokenizer = tokenizer.Subtokenizer(self.vocab_file)
        self.uregex = UnicodeRegex()
        self.case_num = item.get("count", 3003)
        self.calib_num = calib_num
        self.output_list = []
        self.translations = []
        self.batch_size = item.get("batch_size")
        self.item = item

    def unload_query_samples(self):
        self.output_list = []

    def get_input_shapes(testcase, preprocess_params):
        batch_size = int(testcase["batch_size"])
        feature_size = int(preprocess_params["max_seq_length"])
        shape = [batch_size, feature_size]
        return [shape]

    def _get_sorted_inputs(self, filename, case_num=None):
        with open(filename, "r") as f:
          records = f.read().split("\n")
          inputs = [record.strip() for i,record in enumerate(records)]
          if not inputs[-1]:
            inputs.pop()
        if case_num is not None:
            inputs = inputs[0:int(case_num)]
        if self.calib_num is not None:
          if self.calib_num < int(case_num):
            inputs = inputs[:self.calib_num]
        input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
        sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)
        sorted_inputs = [None] * len(sorted_input_lens)
        sorted_keys = [0] * len(sorted_input_lens)
        for i, (index, _) in enumerate(sorted_input_lens):
          sorted_inputs[i] = inputs[index]
          sorted_keys[i] = index
        return sorted_inputs, sorted_keys

    def load_query_samples(self):
        sorted_inputs, sorted_keys = self._get_sorted_inputs(self.input_file, self.case_num)
        num_decode_batches = (len(sorted_inputs) - 1) // self.batch_size + 1
        input_fn = []
        for i, line in enumerate(sorted_inputs):
          if i % self.batch_size == 0:
            input_fn_batch = []
            batch_num = (i // self.batch_size) + 1
            input_fn.append([[self.sub_tokenizer.encode(line, add_eos=True)], sorted_keys[i]])
        return input_fn


class Newstest2014Decode():
    def __init__(self, data_root_dir = None):
        self.bleu_score = 0
        self.dataset_root_dir = data_root_dir
        self.dataset_path = self.dataset_root_dir
        encode_file = "newstest2014.en"
        decode_file = "newstest2014.de"
        vocab = "vocab.ende.32768"
        self.reference_file = os.path.join(self.dataset_path, decode_file)
        self.input_file = os.path.join(self.dataset_path, encode_file)
        self.output_file = os.path.join(os.getcwd(), "translate.en")
        self.vocab_file = os.path.join(self.dataset_path, vocab)
        self.sub_tokenizer = tokenizer.Subtokenizer(self.vocab_file)
        self.sorted_keys = []
        self.res = []
        self.uregex = UnicodeRegex()

    def reset(self, result):
        result["res"] = []
        result["sorted_keys"] = None

    def __call__(self, output, data1, metex, result):
        self.res.append([output])
        self.sorted_keys.extend([data1])
        result["res"] = self.res
        result["sorted_keys"] = self.sorted_keys

    def summary(self, result, testcase=None, record=None):
        case_sensitive = False
        self._write_decode_to_file(result["res"], self.output_file)
        ref_lines = open(self.reference_file, "r").read().strip().splitlines()
        hyp_lines = open(self.output_file, "r").read().strip().splitlines()

        ref_lines = [ref_lines[idx] for idx in result["sorted_keys"]]
        assert len(ref_lines) == len(hyp_lines)
        if not case_sensitive:
          ref_lines = [x.lower() for x in ref_lines]
          hyp_lines = [x.lower() for x in hyp_lines]
        ref_tokens = [self._bleu_tokenize(x, self.uregex) for x in ref_lines]
        hyp_tokens = [self._bleu_tokenize(x, self.uregex) for x in hyp_lines]
        final_result = {}
        blue_score = self._compute_bleu(ref_tokens, hyp_tokens) * 100
        return blue_score


    def _bleu_tokenize(self, string, uregex=None):
        string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
        string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
        string = uregex.symbol_re.sub(r" \1 ", string)
        return string.split()

    def _get_ngrams_with_counter(self, segment, max_order):
        import collections
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
          for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
        return ngram_counts

    def _compute_bleu(self, reference_corpus, translation_corpus, max_order=4, use_bp=True):
        import math
        reference_length = 0
        translation_length = 0
        bp = 1.0
        geo_mean = 0

        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        precisions = []

        for (references, translations) in zip(reference_corpus, translation_corpus):
          reference_length += len(references)
          translation_length += len(translations)
          ref_ngram_counts = self._get_ngrams_with_counter(references, max_order)
          translation_ngram_counts = self._get_ngrams_with_counter(translations, max_order)

          overlap = dict((ngram, min(count, translation_ngram_counts[ngram])) for ngram, count in ref_ngram_counts.items())

          for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
          for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]

        precisions = [0] * max_order
        smooth = 1.0

        for i in range(0, max_order):
          if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
              precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                  i]
            else:
              smooth *= 2
              precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
          else:
            precisions[i] = 0.0

        if max(precisions) > 0:
          p_log_sum = sum(math.log(p) for p in precisions if p)
          geo_mean = math.exp(p_log_sum / max_order)

        if use_bp:
          ratio = translation_length / reference_length
          bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
        bleu = geo_mean * bp
        return np.float32(bleu)

    def _trim_and_decode(self, ids, subtokenizer):
        try:
          index = list(ids).index(tokenizer.EOS_ID)
          return subtokenizer.decode(ids[:index])
        except ValueError:  # No EOS found in sequence
          return subtokenizer.decode(ids)

    def _write_decode_to_file(self, output_decode_list, output_file):
        translations = []
        for i, prediction in enumerate(output_decode_list):
          for mm in prediction[0].tolist():
              translation = self._trim_and_decode(mm, self.sub_tokenizer)
              translations.append(translation)
        if output_file is not None:
          if os.path.isdir(output_file):
            raise ValueError("File output is a directory, will not save outputs to "
                             "file.")
          with open(self.output_file, "w") as f:
            for s in translations:
              f.write("%s\n" % s)
