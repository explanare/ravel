import torch
from tqdm import tqdm


def _generate_single_batch(pretrained_model,
                           tokenizer,
                           prompt_batch,
                           max_length=None,
                           prompt_max_length=32,
                           max_new_tokens=None,
                           sample_n=None,
                           **kwargs):
  if not sample_n:
    sample_n = 1
  if not max_new_tokens:
    assert max_length and prompt_max_length
    max_new_tokens = max_length - prompt_max_length
  input_batch = tokenizer(prompt_batch,
                          return_tensors="pt",
                          padding="max_length",
                          max_length=prompt_max_length,
                          truncation=True)
  input_ids = input_batch['input_ids'].to(pretrained_model.device)
  attention_mask = input_batch['attention_mask'].to(pretrained_model.device)
  with torch.no_grad():
    outputs = pretrained_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True if sample_n > 1 else False,
        num_return_sequences=sample_n,
        return_dict_in_generate=False,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs)
  preds = [(prompt_batch[i // sample_n], p) for i, p in enumerate(
      tokenizer.batch_decode(outputs, skip_special_tokens=True))]
  return preds


def generate_batched(pretrained_model,
                     tokenizer,
                     all_prompts,
                     max_length=None,
                     prompt_max_length=None,
                     max_new_tokens=None,
                     sample_n=None,
                     batch_size=32,
                     **kwargs):
  print('Total #prompts=%d' % len(all_prompts))
  pretrained_model = pretrained_model.eval()
  if prompt_max_length is None:
    # Estimate the max prompt length from the longest sequence in the batch.
    # This estimation assumes all sequences have similar token to character
    # ratio, which would not hold if some sequences are mostly English words,
    # while others are mostly digits, punctuations, etc.
    max_length_prompt = max(all_prompts, key=len)
    prompt_max_length = 8 * (len(tokenizer(max_length_prompt).input_ids) // 8 +
                             1)
    print('Set prompt_max_length=%d' % prompt_max_length)
  prompt_to_raw_outputs = []
  for batch_begin in tqdm(range(0, len(all_prompts), batch_size)):
    batch_prompts = all_prompts[batch_begin:batch_begin + batch_size]
    output_texts = _generate_single_batch(pretrained_model,
                                          tokenizer,
                                          batch_prompts,
                                          prompt_max_length=prompt_max_length,
                                          max_new_tokens=max_new_tokens,
                                          max_length=max_length,
                                          sample_n=sample_n,
                                          **kwargs)
    prompt_to_raw_outputs.extend(output_texts)
  return prompt_to_raw_outputs
