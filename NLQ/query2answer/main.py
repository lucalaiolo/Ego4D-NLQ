"""
Main script to implement the second proposed extension for the Ego4D-NLQ project.
"""
import json
import options
from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor
import torch
import numpy as np
from utils.data_util import prepare_clip, cut_clip, build_prompt

def main(configs):
    data_path = configs.data_path
    clips_path = configs.clips_path
    output_path = configs.output_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import the model with quantization due to memory constraints
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        "LanguageBind/Video-LLaVA-7B-hf",
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["data"]

    output_data = []

    with torch.no_grad():
        for record in data:
            clip_uid = record["clip_uid"]
            query = record["query"]
            start_time = record["predicted_start_time"]
            end_time = record["predicted_end_time"]
            
            cut_clip_path = cut_clip(clips_path, clip_uid, start_time, end_time)
            vid = prepare_clip(cut_clip_path)
            prompt = build_prompt(query)

            inputs = processor(text=prompt, videos=torch.from_numpy(vid).to(device=device), return_tensors="pt")
            inputs = inputs.to(device)
            out = model.generate(**inputs, max_new_tokens=80)
            answer = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            answer = answer.split("ASSISTANT:")[1].strip()
            out_dict = record
            out_dict["model_answer"] = answer
            output_data.append(out_dict)
    
    out = {"data": output_data}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    

if __name__=="__main__":
    configs = options.read_command_line()
    main(configs)