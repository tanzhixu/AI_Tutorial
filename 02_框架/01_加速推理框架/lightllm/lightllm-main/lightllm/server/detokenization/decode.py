from typing import Union, List

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..io_struct import ReqDetokenizationState


def decode_token(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    req: ReqDetokenizationState,
    new_token_id: int,
    eos_id: List[int],
) -> str:
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=req.skip_special_tokens)
    req.output_tokens.append(new_token)
    
    is_eos_id = new_token_id in eos_id
    if is_eos_id and not req.print_eos_token:
        return req.output_str
    
    if req.skip_special_tokens and new_token_id in tokenizer.all_special_ids and not is_eos_id:
        return req.output_str

    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(req.output_tokens)
        return output_text

    sep = " " if req.add_spaces_between_special_tokens else ""

    if new_token in tokenizer.added_tokens_encoder:
        if req.current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(req.current_sub_text)
            req.sub_texts.append(sub_text)
            req.current_sub_text = []
        req.sub_texts.append(new_token)
        return sep.join(req.sub_texts)
    else:
        req.current_sub_text.append(new_token)
        new_sub_text = tokenizer.convert_tokens_to_string(req.current_sub_text)
        return sep.join(req.sub_texts + [new_sub_text])
