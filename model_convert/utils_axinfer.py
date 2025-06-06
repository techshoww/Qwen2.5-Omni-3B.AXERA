import os

import numpy as np
import onnxruntime as ort
from axengine import InferenceSession
from ml_dtypes import bfloat16
from scipy.special import softmax
from transformers import AutoTokenizer


def post_process(data, topk=1, topp=0.001, temperature=0.1):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    # def softmax(l: np.ndarray) -> np.ndarray:
    #     l_max = l - l.max()
    #     l_exp = np.exp(l_max)
    #     res = l_exp / np.sum(l_exp)
    #     return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft


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
        self, cfg, mode_dir, model_name, prefill_len, lastN, run_dynamic=False
    ):

        # model_name="qwen2_5_omni_text"
        self.cfg = cfg
        self.prefill_len = prefill_len

        self.num_hidden_layers = self.cfg.num_hidden_layers
        self.hidden_size = self.cfg.hidden_size

        self.prefill_decoder_sessins = []
        for i in range(self.num_hidden_layers):
            # session = InferenceSession(
            #     f"{mode_dir}/{model_name}_p{prefill_len}_l{i}_together.AxModel"
            # )
            session = AxModelInfer(
                f"{mode_dir}/{model_name}_p{prefill_len}_l{i}_together.axmodel",
                run_dynamic,
            )
            self.prefill_decoder_sessins.append(session)
        # self.post_process_session = InferenceSession(
        #     f"{mode_dir}/{model_name}_post.axmodel"
        # )
        self.post_process_session = AxModelInfer(
            f"{mode_dir}/{model_name}_post.axmodel", run_dynamic
        )
        self.embeds = np.load(f"{mode_dir}/model.embed_tokens.weight.npy")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_dir, trust_remote_code=True)
        self.lastN = lastN

    def embed_tokens(self, input_ids):
        return np.take(self.embeds, input_ids, axis=0)

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
            input_ids = []
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds

        kv_dim = (
            self.cfg.hidden_size
            // self.cfg.num_attention_heads
            * self.cfg.num_key_value_heads
        )
        k_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]
        v_caches = [
            np.zeros((1, self.lastN, kv_dim), dtype=bfloat16)
            for _ in range(self.cfg.num_hidden_layers)
        ]

        print("hidden_states.shape", hidden_states.shape)
        token_len = hidden_states.shape[1]

        indices = np.zeros((3, self.prefill_len), dtype=np.uint32)
        print("token_len", token_len)
        print("position_ids.shape", position_ids.shape)
        indices[:, 0:token_len] = position_ids.squeeze(1).numpy().astype(np.uint32)
        print("indices", indices.shape)
        mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
        data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(bfloat16)
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
                "K_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                "V_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
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

        # post_norm = []
        # for ti in range(token_len):

        #     pn, _ = self.post_process_session( {"input": data[:, ti:ti+1, :]})

        #     post_norm.append(pn)
        # post_norm = np.concatenate(post_norm, axis=1)

        # thinker_hidden_states.append(post_norm)
        print("post_out", post_out, post_out.shape)
        next_token, posssible_tokens, possible_soft = post_process(post_out, topk=1)
        posibles = [self.tokenizer.decode([t]) for t in posssible_tokens]
        posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
        input_ids.append(next_token)
        print("posibles", posibles)
        print(self.tokenizer.decode(input_ids[-1:]))
        print("prefill done!")

        # set to decoder

        start_ids = np.max(indices) + 1
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
                self.embeds[next_token, :]
                .reshape((1, 1, self.hidden_size))
                .astype(bfloat16)
            )
            # thinker_token_embeds.append(data)
            print("553, indices", indices.shape)
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
                # thinker_hidden_states.append(post_norm)
                print("post_out", post_out.shape)
                next_token, posssible_tokens, possible_soft = post_process(post_out)
                input_ids.append(next_token)
                print(self.tokenizer.decode(input_ids[-1:]))
            if next_token == self.tokenizer.eos_token_id:
                # print("hit eos!")
                break

        return input_ids
