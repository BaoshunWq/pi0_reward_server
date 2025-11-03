from datasets import load_dataset, Features, Value
from datasets import Image as HFImage


def make_data_for_trl(path: str,data_type: str):
    dpo_features = {
        "image": HFImage(decode=True),
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
        # "task": Value("string")
    }
    kto_features = {
        "image": HFImage(decode=True),
        "prompt": Value("string"),
        "response": Value("string"),
        "label": "int64",
        # "task": Value("string")
    }

        # 把单张图片放进 'images' 列（list of PIL），以便 VLM 的 processor 使用
    def to_images(ex):
        ex["images"] = [ex["image"]]  # 多图可放多张
        return ex

    if data_type == "dpo":
        features = Features(dpo_features)
        ds = load_dataset("json", data_files=path, features=features)
        ds = ds.map(to_images, remove_columns=["image"])
        return ds
    elif data_type == "kto":
        features = Features(kto_features)
        ds = load_dataset("json", data_files=path, features=features)
        ds = ds.map(to_images, remove_columns=["image"])
        return ds
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    


def generate_text_from_sample(model, processor, sample, max_new_tokens=256, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = sample["prompt"]
    image_inputs = []
    image = sample["images"][0]
    image_inputs.append([image])
    # Prepare the inputs for the model
    model_inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device
    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]  # Return the first decoded output text