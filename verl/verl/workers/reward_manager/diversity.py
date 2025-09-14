from collections import defaultdict
import numpy as np
import torch

from verl import DataProto


class DiversityRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        uid = data.non_tensor_batch["uid"] # this is a tensor of shape (batch_size, 1) that maps each response to its uid
        log_probs = data.batch["old_log_probs"].sum(dim=-1) if "old_log_probs" in data.batch else None
        # normalize the log_probs by length
        if log_probs is not None:
            log_probs = log_probs / attention_mask.sum(dim=-1)
            
        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        if data.non_tensor_batch.get("reward_model") is not None:
            ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        else:
            ground_truths = [None] * len(data)
        
        data_sources = data.non_tensor_batch[self.reward_fn_key] if self.reward_fn_key in data.non_tensor_batch else ['wildchat'] * len(data)
        extras = data.non_tensor_batch.get("extra_info", None)
        correctness = data.non_tensor_batch.get("correctness", None)

        scores = self.compute_score(
            data_source=data_sources,
            solution_str=responses_str,
            ground_truth=ground_truths,
            extra_info=extras,
            prompts=[self.tokenizer.decode(ids, skip_special_tokens=True) for ids in prompt_ids],
            uid=uid,
            log_probs=log_probs,
            correctness=correctness,
            **self.reward_kwargs,
        )

        return scores


    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        #if "rm_scores" in data.batch.keys():
        #    if return_dict:
        #        return {"reward_tensor": data.batch["rm_scores"]}
        #    else:
        #        return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key] if self.reward_fn_key in data.non_tensor_batch else ['wildchat'] * len(data)
        reward_extra_info = data.non_tensor_batch.get("extra_info", None)

        scores = self.verify(data)  
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                #for key, value in score.items():
                #    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
