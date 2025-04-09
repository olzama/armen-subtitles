from comet import download_model, load_from_checkpoint
from huggingface_hub import login
import sys
from process_srt import combined_text
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

original = combined_text(sys.argv[1])
youtube_translation = combined_text(sys.argv[2])
google_translate_translation = combined_text(sys.argv[3])
GPT_4o_full_pipeline_translation_srt = combined_text(sys.argv[4])


model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def compute_perplexity_large(text, max_length=1024):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    nlls = []
    stride = max_length
    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i:i + max_length]
        if len(chunk) < 2:
            continue
        chunk = chunk.unsqueeze(0).to(model.device)
        with torch.no_grad():
            outputs = model(chunk, labels=chunk)
            neg_log_likelihood = outputs.loss * chunk.size(1)
        nlls.append(neg_log_likelihood)

    total_nll = torch.stack(nlls).sum()
    total_tokens = len(input_ids)
    return math.exp(total_nll / total_tokens)

def score(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    log_likelihood = outputs.loss.item()
    return math.exp(log_likelihood)  # Perplexity

print("Youtube perplexity: {} ".format(compute_perplexity_large(youtube_translation)))
print("Google translate perplexity: {} ".format(compute_perplexity_large(google_translate_translation)))
print("GPT-4o full pipeline perplexity: {} ".format(compute_perplexity_large(GPT_4o_full_pipeline_translation_srt)))

# with open("./hf-token.txt", "r") as myfile:
#     hf_token = myfile.read().replace('\n', '')
#
# login(token=hf_token)
#
# # Download a supported QE model (no reference needed)
# model_path = download_model("Unbabel/wmt22-cometkiwi-da")  # or use -no-doc variant
# model = load_from_checkpoint(model_path)
#
#
#
# # Define input data (source + hypothesis, no reference!)
# data = [
#     {"src": original, "mt": youtube_translation},
#     {"src": original, "mt": google_translate_translation},
#     {"src": original, "mt": GPT_4o_full_pipeline_translation_srt}
# ]
#
# # Predict quality scores
# predictions = model.predict(data, batch_size=8, gpus=0)
# print(predictions)
