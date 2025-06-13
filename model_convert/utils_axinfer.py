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


class AxModelInferStatic:
    def __init__(self, axmodel_path):
        self.session = InferEngine(axmodel_path)

    def __call__(self, inputs, shape_group=None):

        if shape_group is None:
            outputs = self.session.run(None, inputs)
        else:
            outputs = self.session.run(None, inputs, shape_group=shape_group)
        return outputs


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


class AxLMInfer:
    def __init__(
        self, cfg, model_dir, model_name, prefill_len, lastN, run_dynamic=False
    ):

        # model_name="qwen2_5_omni_text"
        self.cfg = cfg
        self.prefill_len = prefill_len

        self.num_hidden_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size

        self.prefill_decoder_sessins = []
        for i in range(self.num_hidden_layers):
            # session = InferenceSession(
            #     f"{model_dir}/{model_name}_p{prefill_len}_l{i}_together.AxModel"
            # )
            session = AxModelInfer(
                f"{model_dir}/{model_name}_p{prefill_len}_l{i}_together.axmodel",
                run_dynamic,
            )
            self.prefill_decoder_sessins.append(session)
        # self.post_process_session = InferenceSession(
        #     f"{model_dir}/{model_name}_post.axmodel"
        # )
        self.post_process_session = AxModelInfer(
            f"{model_dir}/{model_name}_post.axmodel", run_dynamic
        )
        self.embeds = np.load(f"{model_dir}/model.embed_tokens.weight.npy")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.lastN = lastN

        self.thinker_to_talker_proj = AxModelInfer(
            f"thinker_to_talker_proj.onnx", 
            run_dynamic
        )

    def embed_tokens(self, input_ids):
        # return np.take(self.embeds, input_ids, axis=0)
        output = np.take(self.embeds, input_ids, axis=0)
        output = self.thinker_to_talker_proj({"input": output})[0]
        return output

    def __call__(
        self,
        input_ids=None,
        input_embeds=None,
        position_ids=None,
    ):

        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or input_embeds"
            )

        if input_ids is None:
            print("input_embeds",input_embeds.shape)
            input_ids = [0]*input_embeds.shape[1]       # 添加占位符
            # with open("talker_input_ids", "rb") as f:
            #     talker_input_ids = dill.load(f)
            # input_ids = talker_input_ids.int().reshape(-1).tolist()
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds

        kv_dim = (
            self.cfg.hidden_size
            // self.cfg.num_attention_heads
            * self.cfg.num_key_value_heads
        )
        k_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=np.float32)
            for _ in range(self.cfg.num_hidden_layers)
        ]
        v_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=np.float32)
            for _ in range(self.cfg.num_hidden_layers)
        ]

        print("hidden_states.shape", hidden_states.shape)
        token_len = hidden_states.shape[1]
        prompt_ignore_length = token_len
        indices = np.zeros((3, self.prefill_len), dtype=np.uint32)
        print("token_len", token_len)
        print("position_ids.shape", position_ids.shape)
        indices[:, 0:token_len] = position_ids.squeeze(1).numpy().astype(np.uint32)
        print("indices", indices.shape)
        mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
        data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(np.float32)
        # thinker_token_embeds = []
        # thinker_hidden_states = []
        print("data", data.shape)
        data[:, 0:token_len] = hidden_states
        # thinker_token_embeds.append(data[:, 0:token_len])
        for i in range(token_len):
            mask[:, i, : i + 1] = 0

        mask = mask.astype(bfloat16)
        for i in range(self.num_hidden_layers):
            input_feed = {
                "K_cache": np.zeros((1, 1, self.hidden_size), dtype=np.float32),
                "V_cache": np.zeros((1, 1, self.hidden_size), dtype=np.float32),
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=1)

            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data[:, 0:token_len] = outputs[2][:, :token_len, :]

        post_out = self.post_process_session(
            {"input": data[:, token_len - 1 : token_len, :]}
        )[0]

        next_token, posssible_tokens, possible_soft = post_process(input_ids[prompt_ignore_length:], post_out, topk=40, topp=0.8, temperature=0.9, repetition_penalty=1.1, suppress_tokens=[8293])
        input_ids.append(next_token)
        print("prefill done!")

        # set to decoder

        start_ids = np.max(indices) + 1
        print("start ids",start_ids)
        
        mask = np.zeros((1, 1, self.lastN + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, : self.lastN] -= 65536
        mask[:, :, :token_len] = 0
        for start_indice in range(np.max(indices) + 1, self.lastN + 1):

            if self.prefill_len > 0 and start_indice < token_len:
                continue
            next_token = input_ids[start_indice]
            indices = np.array([start_ids], np.uint32).reshape((1, 1))
            start_ids += 1
            data = (
                self.embed_tokens( np.array([next_token]).reshape(1,1) )
                .reshape((1, 1, self.hidden_size))
                .astype(np.float32)
            )
            
            print("--------------------------------------indices", indices, "next_token", next_token)
            for i in range(self.cfg.num_hidden_layers):
                # print("decode layer:",i)
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }

                outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=0)

                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., start_indice] = 0
            if start_indice < token_len - 1:
                pass
            else:

                post_out = self.post_process_session({"input": data})[0]
                
                next_token, posssible_tokens, possible_soft = post_process(input_ids[prompt_ignore_length:], post_out, topk=40, topp=0.8, temperature=0.9, repetition_penalty=1.1, suppress_tokens=[8293])
                input_ids.append(next_token)
                print("---------------------------------next_token ", next_token)
                
            if next_token in [8292, 8294]:
                print("hit eos!")
                break

        return input_ids
