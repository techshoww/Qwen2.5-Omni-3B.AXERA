import os
import numpy as np
import onnxruntime as ort
from axengine import InferenceSession
from ml_dtypes import bfloat16
# from scipy.special import softmax
from transformers import AutoTokenizer
import gc
import dill 
import torch
from torch import nn
from functools import wraps


def isin_mps_friendly(elements: torch.Tensor, test_elements: torch.Tensor | int) -> torch.Tensor:
    """
    Same as `torch.isin` without flags, but MPS-friendly. We can remove this function when we stop supporting
    torch <= 2.3. See https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075

    Args:
        elements (`torch.Tensor`): Input elements
        test_elements (`torch.Tensor` or `int`): The elements to check against.

    Returns:
        `torch.Tensor`: A boolean tensor of the same shape as `elements` that is True for `elements` in `test_elements`
        and False otherwise
    """

    if elements.device.type == "mps" and not is_torch_greater_or_equal_than_2_4:
        test_elements = torch.tensor(test_elements)
        if test_elements.ndim == 0:
            test_elements = test_elements.unsqueeze(0)
        return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()
    else:
        # Note: don't use named arguments in `torch.isin`, see https://github.com/pytorch/pytorch/issues/126045
        return torch.isin(elements, test_elements)

def do_repetition_penalty(input_ids, scores, penalty):
    score = torch.gather(scores, 1, input_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores_processed = scores.scatter(1, input_ids, score)
    return scores_processed

def do_supress_tokens(scores, suppress_tokens):
    vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
    suppress_token_mask = isin_mps_friendly(vocab_tensor, suppress_tokens)
    scores = torch.where(suppress_token_mask, -float("inf"), scores)
    return scores

def do_tempeture(scores, temperature):
    scores_processed = scores / temperature
    return scores_processed

def do_topk(scores, top_k, filter_value=-float("Inf")):
    top_k = min(top_k, scores.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed

def do_topp(scores, top_p, min_tokens_to_keep=1, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores_processed = scores.masked_fill(indices_to_remove, filter_value)
    return scores_processed

def do_logitnorm(scores):
    scores_processed = scores.log_softmax(dim=-1)
    return scores_processed

def post_process(input_ids, data, topk=1, topp=0.001, temperature=0.1, repetition_penalty=None, suppress_tokens=None, do_sample=True):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()

    if repetition_penalty is not None and repetition_penalty != 1.0:
        ids = torch.tensor(input_ids).reshape(1,-1)
        r = torch.from_numpy(r).reshape(1,-1)
        r = do_repetition_penalty(ids, r, repetition_penalty)
        r = r.numpy().flatten()
    
    if suppress_tokens is not None and len(suppress_tokens)>0:
        r = torch.from_numpy(r).reshape(1,-1)
        r = do_supress_tokens(r, torch.tensor(suppress_tokens))
        r = r.numpy().flatten()

    r = torch.from_numpy(r).reshape(1,-1)
    r = do_tempeture(r, temperature)
    r = do_topk(r, topk)
    r = do_topp(r, topp)
    # r = do_logitnorm(r)
    if do_sample:
        probs = nn.functional.softmax(r, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)[0].item()
    else:
        next_token = r.argmax(dim=-1)[0].item()
    return next_token, None, None
    # # topk
    # candidate_index = np.argpartition(r, -topk)[-topk:]
    # candidate_value = r[candidate_index]
    # # temperature
    # candidate_value /= temperature
    # # softmax
    # candidate_soft = softmax(candidate_value)
    # # topp
    # candidate_soft = top_p(candidate_soft, topp)
    # candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    # pos = np.random.multinomial(1, candidate_soft).argmax()
    # next_token = candidate_index[pos]
    # return next_token, candidate_index, candidate_soft


class InferEngine:
    def __init__(self, path):
        self.path = path

        model_type = os.path.splitext(path)[1]

        engine = None
        if model_type == ".onnx":
            engine = ort.InferenceSession
        elif model_type == ".axmodel":
            engine = InferenceSession
        else:
            raise NotImplementedError(f"Not supported model type: {model_type}")

        assert engine is not None

        self.session = engine(path)

    def run(self, outputnames, inputs, shape_group=None):
        if shape_group is None:
            outputs = self.session.run(outputnames, inputs)
        else:
            outputs = self.session.run(outputnames, inputs, shape_group=shape_group)
        return outputs

    def _unload(self):
        if isinstance(self.session, InferenceSession):
            self.session._sess._unload()
        else:
            del self.session

class AxModelInferStatic:
    def __init__(self, axmodel_path):
        self.session = InferEngine(axmodel_path)

    def __call__(self, inputs, shape_group=None):

        if shape_group is None:
            outputs = self.session.run(None, inputs)
        else:
            outputs = self.session.run(None, inputs, shape_group=shape_group)
        return outputs

    def _unload(self):
        self.session._unload()

class AxModelInferDynamic:
    def __init__(self, axmodel_path):
        self.axmodel_path = axmodel_path

    def __call__(self, inputs, shape_group=None):

        session = InferEngine(self.axmodel_path)
        if shape_group is None:
            outputs = session.run(None, inputs)
        else:
            outputs = session.run(None, inputs, shape_group=shape_group)

        return outputs


class AxModelInfer:
    def __init__(self, axmodel_path, run_dynamic=False):
        if run_dynamic:
            self.model = AxModelInferDynamic(axmodel_path)
        else:
            self.model = AxModelInferStatic(axmodel_path)

    def __call__(self, inputs, shape_group=None):
        try:
            outputs = self.model(inputs, shape_group)
        except Exception as e:
            if hasattr(self.model, "axmodel_path"):
                print(f"axmodel_path:{self.model.axmodel_path}")
            print(e)
            raise e

        return outputs

    def _unload(self):
        self.model._unload()

def lazyforward(func):
    @wraps(func)  
    def wrapper(self, *args, **kwargs):
        if self.lazy_load:
            self._load()

        result = func(self, *args, **kwargs)  
        
        if self.lazy_load:
            self._unload()

        return result
    return wrapper

class AxLMInfer:
    def __init__(
        self, cfg, model_dir, model_name, prefill_len, lastN, run_dynamic=False, lazy_load=True
    ):

        self.cfg = cfg
        self.model_dir = model_dir
        self.model_name = model_name
        self.prefill_len = prefill_len
        self.lastN = lastN
        self.run_dynamic = run_dynamic
        self.lazy_load = lazy_load and (not run_dynamic)

        self.num_hidden_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size

        if not self.lazy_load:
            self._load()
        
    def _load(self):
        self.prefill_decoder_sessins = []
        for i in range(self.num_hidden_layers):
            
            session = AxModelInfer(
                f"{self.model_dir}/{self.model_name}_p{self.prefill_len}_l{i}_together.axmodel",
                self.run_dynamic,
            )
            self.prefill_decoder_sessins.append(session)
        
        self.post_process_session = AxModelInfer(
            f"{self.model_dir}/{self.model_name}_post.axmodel", self.run_dynamic
        )

    def _unload(self):
        for session in self.prefill_decoder_sessins:
            session._unload()
        self.post_process_session._unload()

        self.prefill_decoder_sessins = None
        self.post_process_session = None
    
    def forward(self,*args, **kwargs):
        raise NotImplementedError("forward funciton was not implemented")

    @lazyforward
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

        

