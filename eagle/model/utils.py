import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def remove_image_token_except_first(
    input_ids,
    img_tok_index,
    hidden_states=None,
    attentions=None,
    image_features=None,
    topk=20
):
    
    """
    input_ids: [1, seq_len]
    hidden_states: [1, seq_len, dim] (optional)
    image_features: [1, N_img, dim] or [N_img, dim] (optional)
    
    Returns:
        filtered_input_ids: [1, new_seq_len]
        filtered_hidden_states: [1, new_seq_len, dim] or None
        filtered_image_features: [1, 1, dim] or None
    """
    input_ids = input_ids[0]  # [seq_len]
    device = input_ids.device

    # 이미지 토큰 위치 찾기
    image_token_indices = (input_ids == img_tok_index).nonzero(as_tuple=True)[0]  # [N_img]

    if len(image_token_indices) > 0:
        # 첫 번째 이미지 토큰만 유지
        keep_image_index = image_token_indices[0]
    else:
        keep_image_index = None  # 이미지 토큰이 아예 없는 경우

    # 전체 마스크 생성
    keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if keep_image_index is not None:
        # 일단 모든 이미지 토큰 제거
        keep_mask[input_ids == img_tok_index] = False
        # 첫 번째만 다시 True로 되돌림
        keep_mask[keep_image_index] = True

    # 필터링
    filtered_input_ids = input_ids[keep_mask].unsqueeze(0).to(device)  # [1, new_seq_len]

    filtered_hidden_states = None
    if hidden_states is not None:
        filtered_hidden_states = hidden_states[0][keep_mask, :].unsqueeze(0)  # [1, new_seq_len, dim]

    filtered_image_features = None
    if image_features is not None and keep_image_index is not None:
        first_local_index = (image_token_indices == keep_image_index).nonzero(as_tuple=True)[0].item()

        if image_features.dim() == 3:
            # [1, N_img, dim] → [1, 1, dim]
            filtered_image_features = image_features[:, first_local_index:first_local_index+1, :]
        else:
            # [N_img, dim] → [1, 1, dim]
            filtered_image_features = image_features[first_local_index:first_local_index+1, :].unsqueeze(0)

    return filtered_input_ids, filtered_hidden_states, filtered_image_features

def remove_image_token_except_last(input_ids, img_tok_index, hidden_states=None):
    # input_ids, loss_mask는 (1, seq_len) 형태라고 가정
    # hidden_states는 (1, seq_len, hidden_dim) 형태라고 가정
    
    # 먼저 (1, seq_len) -> (seq_len,) 으로 차원을 줄임
    flat_input_ids = input_ids.squeeze(0)   # (seq_len,)

    # 32000(img_tok_index)인 위치 전부 찾기
    positions = (flat_input_ids == img_tok_index).nonzero(as_tuple=True)[0]
    
    # 만약 32000이 여러 개라면, 마지막 위치만 남기고 다 제거할 마스크를 만든다
    if len(positions) > 1:
        last_pos = positions[-1]
        
        # 일단 전부 True로 초기화
        keep_mask = torch.ones_like(flat_input_ids, dtype=torch.bool)
        # 32000이었던 위치 전부 False로 설정
        keep_mask[positions] = False
        # 마지막 하나만 True로 되돌림
        keep_mask[last_pos] = True
    else:
        # 32000이 없거나 한 개만 있을 경우엔 전부 유지
        keep_mask = torch.ones_like(flat_input_ids, dtype=torch.bool)

    # 마스크대로 input_ids, loss_mask 추려서 (1, -1)로 형태 맞춤
    filtered_input_ids = flat_input_ids[keep_mask].unsqueeze(0)

    if hidden_states is not None:
        # hidden_states: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
        flat_hidden_states = hidden_states.squeeze(0)
        
        # keep_mask를 적용해 (남길 위치만 남기기)
        filtered_hidden_states = flat_hidden_states[keep_mask, :].unsqueeze(0)
        
        return filtered_input_ids, filtered_hidden_states
    
    return filtered_input_ids

def remove_image_token(input_ids, img_tok_index, hidden_states=None, attentions=None,
    image_features=None,
    topk=20):
    mask = input_ids != img_tok_index
    filtered_input_ids = input_ids[mask].view(1, -1).to(input_ids.device)
    if hidden_states is not None:
        filtered_hidden_states = hidden_states[:, mask[0], :]
        return filtered_input_ids, filtered_hidden_states
    
    return filtered_input_ids

def pool_image_token(input_ids, img_tok_index, hidden_states=None):
    mask = input_ids != img_tok_index
    filtered_input_ids = input_ids[mask].view(1, -1).to(input_ids.device)
    if hidden_states is not None:
        filtered_hidden_states = hidden_states[:, mask[0], :]
        return filtered_input_ids, filtered_hidden_states
    
    return filtered_input_ids

def keep_topk_image_token(
    input_ids,
    img_tok_index,
    hidden_states=None,
    attentions=None,
    image_features=None,
    topk=20
):
    """
    input_ids: [1, seq_len]
    hidden_states: [1, seq_len, dim]
    attentions: list of [1, heads, seq_len, seq_len]
    image_features: [1, N_img, dim] or [N_img, dim]
    """

    input_ids = input_ids[0]  # [seq_len]
    device = input_ids.device

    # 이미지 토큰 위치 찾기
    image_token_indices = (input_ids == img_tok_index).nonzero(as_tuple=True)[0]  # [N_img]

    if hidden_states is None:
        topk_indices_global = image_token_indices[:topk]
        topk_indices_local = torch.arange(len(topk_indices_global), device=device)
    else:
        assert attentions is not None, "hidden_states가 주어졌으면 attentions도 있어야 해!"
        last_layer_attn = attentions[0][0]              # [heads, seq_len, seq_len]
        
        # ★ 변경된 부분: 마지막 토큰 대신 CLS 토큰 사용
        cls_index = (input_ids == img_tok_index).nonzero(as_tuple=True)[0][0]
        avg_attention = last_layer_attn.mean(dim=0)      # [seq_len, seq_len]
        cls_token_attention = avg_attention[cls_index]   # [seq_len]
        image_token_scores = cls_token_attention[image_token_indices]
        
        topk = min(topk, image_token_scores.size(0))     # 안전 처리
        topk_indices_local = torch.topk(image_token_scores, topk).indices
        topk_indices_global = image_token_indices[topk_indices_local]

    # 마스크 생성
    text_mask = input_ids != img_tok_index
    topk_img_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    topk_img_mask[topk_indices_global] = True
    final_mask = text_mask | topk_img_mask

    # input_ids 필터링
    filtered_input_ids = input_ids[final_mask].unsqueeze(0).to(device)  # [1, new_seq_len]

    # hidden_states 필터링 (batch 유지)
    filtered_hidden_states = None
    if hidden_states is not None:
        filtered_hidden_states = hidden_states[0][final_mask, :].unsqueeze(0)  # [1, new_seq_len, dim]

    # image_features 필터링 (batch 유지)
    filtered_image_features = None
    if image_features is not None:
        if image_features.dim() == 3:
            # [1, N_img, dim] → 그대로 필터링
            filtered_image_features = image_features[:, topk_indices_local, :]  # [1, topk, dim]
        else:
            # [N_img, dim] → batch 추가
            filtered_image_features = image_features[topk_indices_local].unsqueeze(0)  # [1, topk, dim]

    return filtered_input_ids, filtered_hidden_states, filtered_image_features

def nothing_image_token(input_ids, img_tok_index, hidden_states=None):
    if hidden_states is not None:
        return input_ids, hidden_states
    
    return input_ids

def initialize_tree(input_ids, model, pixel_values, past_key_values, logits_processor, token_process, num_img_tokens):
    if token_process == 1:
        process_token = remove_image_token
    elif token_process == 2:
        process_token = pool_image_token
    elif token_process == 3:
        process_token = remove_image_token_except_last
    elif token_process == 4:
        process_token = remove_image_token_except_first
    elif token_process == 5:
        process_token = keep_topk_image_token
    else :
        process_token = nothing_image_token
    
    outputs, orig, hidden_states = model(
        input_ids, pixel_values, past_key_values=past_key_values, output_orig=True
    )
    
    image_features = None
    if pixel_values is not None:
        image_features = model.get_image_features(pixel_values)

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    ea_layer_device = model.ea_layer.fc.weight.device
    filtered_input_ids, filtered_hidden_states, filtered_image_features = process_token(
        input_ids=input_ids,
        img_tok_index=model.base_model.config.image_token_index,
        hidden_states=hidden_states,
        attentions=outputs.attentions,
        image_features=image_features)
    filtered_input_ids = torch.cat((filtered_input_ids, token.to(filtered_input_ids.device)), dim=1)
    filtered_input_ids = filtered_input_ids.to(ea_layer_device)
    filtered_hidden_states = filtered_hidden_states.to(ea_layer_device)
    # Clone the output hidden states
    
    
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(filtered_hidden_states, filtered_input_ids, model.base_model.language_model.lm_head,logits_processor,filtered_image_features)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token


def reset_tree_mode(
        model,
):
    model.base_model.language_model.model.tree_mask = None
    model.base_model.language_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        pixel_values,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    pixel_values=None
    position_ids = tree_position_ids + input_ids.shape[1]
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        pixel_values,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p,
        token_process,
        num_img_tokens,
):
    if token_process == 1:
        process_token = remove_image_token
    elif token_process == 2:
        process_token = pool_image_token
    elif token_process == 3:
        process_token = remove_image_token_except_last
    elif token_process == 4:
        process_token = remove_image_token_except_first
    elif token_process == 5:
        process_token = keep_topk_image_token
    else :
        process_token = nothing_image_token
        
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    
    #selected_tokens = model.tokenizer.batch_decode(candidates[None, best_candidate, 1: accept_length + 1].tolist())
    #print(selected_tokens)
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    ea_layer_device = model.ea_layer.fc.weight.device

    input_ids = input_ids.to(ea_layer_device)
    filtered_input_ids, _, _ = process_token(
        input_ids=input_ids,
        img_tok_index=model.base_model.config.image_token_index)
    filtered_input_ids = filtered_input_ids.to(ea_layer_device)
    filtered_input_ids = torch.cat((filtered_input_ids, token.to(filtered_input_ids.device)), dim=1)
    accept_hidden_state_new = accept_hidden_state_new.to(ea_layer_device)
    
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=filtered_input_ids, head=model.base_model.language_model.lm_head,logits_processor=logits_processor)

    new_token += int(accept_length) + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
