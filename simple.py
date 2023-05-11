import argparse
import json
import numpy as np
import re
import torch
import tqdm
import transformers


class DetectGPT():

    def __init__(self, base_model, base_tokenizer, perturb_model, perturb_tokenizer, n_perturbations, n_perturbation_rounds, pct_words_masked, mask_top_p, span_length, batch_size, chunk_size, buffer_size, ceil_pct, device, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.perturb_model = perturb_model
        self.perturb_tokenizer = perturb_tokenizer
        self.n_perturbations = n_perturbations
        self.n_perturbation_rounds = n_perturbation_rounds
        self.pct_words_masked = pct_words_masked
        self.mask_top_p = mask_top_p
        self.span_length = span_length
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.ceil_pct = ceil_pct
        self.device = device

    # define regex to match all <extra_id_*> tokens, where * is an integer
    _pattern = re.compile(r"<extra_id_\d+>")

    def _tokenize_and_mask(self, text):
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = self.pct_words_masked * len(tokens) / (self.span_length + self.buffer_size * 2)
        if self.ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - self.span_length)
            end = start + self.span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text

    def _count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    # replace each masked span with a sample from T5 mask_model
    def _replace_masks(self, texts):
        n_expected = self._count_masks(texts)
        stop_id = self.perturb_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.perturb_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.perturb_model.generate(**tokens, max_length=150, do_sample=True, top_p=self.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
        return self.perturb_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def _extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self._pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def _apply_extracted_fills(self, masked_texts, extracted_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self._count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts

    def _perturb_texts_(self, texts):
        masked_texts = [self._tokenize_and_mask(x) for x in texts]
        raw_fills = self._replace_masks(masked_texts)
        extracted_fills = self._extract_fills(raw_fills)
        perturbed_texts = self._apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self._tokenize_and_mask(x) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self._replace_masks(masked_texts)
            extracted_fills = self._extract_fills(raw_fills)
            new_perturbed_texts = self._apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        return perturbed_texts

    def _perturb_texts(self, texts):
        outputs = []
        for i in tqdm.tqdm(range(0, len(texts), self.chunk_size), desc="Applying perturbations"):
            outputs.extend(self._perturb_texts_(texts[i:i + self.chunk_size]))
        return outputs

    # Get the log likelihood of each text under the base_model
    def _get_ll(self, text):
        with torch.no_grad():
            tokenized = self.base_tokenizer(text, return_tensors="pt").to(self.device)
            labels = tokenized.input_ids
            return -self.base_model(**tokenized, labels=labels).loss.item()

    def _get_lls(self, texts):
        return [self._get_ll(text) for text in texts]

    def _get_perturbation_results(self, original_text):
        results = []

        p_original_text = self._perturb_texts([x for x in original_text for _ in range(self.n_perturbations)])
        for _ in range(self.n_perturbation_rounds - 1):
            try:
                p_original_text = self.perturb_texts(p_original_text)
            except AssertionError:
                break

        assert len(p_original_text) == len(original_text) * self.n_perturbations, f"Expected {len(original_text) * self.n_perturbations} perturbed samples, got {len(p_original_text)}"

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "perturbed_original": p_original_text[idx * self.n_perturbations: (idx + 1) * self.n_perturbations]
            })

        return results

    def _score_perturbation_results(self, results):
        for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
            p_original_ll = self._get_lls(res["perturbed_original"])
            res["original_ll"] = self._get_ll(res["original"])
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1
        for res in results:
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            res['z-score'] = (res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std']
        return results

    def _run_perturbation_experiment(self, results):
        # compute diffs with perturbed
        predictions = []
        return predictions

    def detect(self, data):
        self.perturb_model.to(self.device)
        # run perturbation experiments
        results = self._get_perturbation_results(data)
        self.perturb_model.cpu()

        self.base_model.to(self.device)
        results = self._score_perturbation_results(results)
        self.base_model.cpu()

        self._run_perturbation_experiment(results)
        return results


def load_data(file_name, key):
    print(f'Loading dataset from {file_name} ...')
    data = [json.loads(line)[key] for line in open(file_name, "rt")]
    return data


def load_generative_model(name):
    print(f'Loading generative base model {name} ...')
    base_model = transformers.AutoModelForCausalLM.from_pretrained(name)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer

    
def load_mask_filling_model(name):
    print(f'Loading mask-filling perturbation model {name} ...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(name)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(name, model_max_length=n_positions)
    return mask_model, mask_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_key', type=str, default="text")
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_perturbations', type=int, default=10)
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--perturb_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--ceil_pct', action='store_true')
    args = parser.parse_args()

    # generic generative model
    args.base_model, args.base_tokenizer = load_generative_model(args.base_model_name)

    # generic fill-masking model
    args.perturb_model, args.perturb_tokenizer = load_mask_filling_model(args.perturb_model_name)

    # load dataset
    data = load_data(args.dataset, args.dataset_key)

    args = vars(args)
    for handled in ['dataset', 'dataset_key', 'base_model_name', 'perturb_model_name']:
        del args[handled]
    detect = DetectGPT(**args)

    predictions = detect.detect(data)
    print(predictions)

if __name__ == '__main__':
    main()
