import pandas as pd
import torch
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_normalization():
    # Sua doi INPUT_FILE va OUTPUT_FILE thanh path noi luu file CSV
    INPUT_FILE = "/media/nam/DATA/RAI_K63/HK9/NLP/comic-generation/data/full_story.csv"
    OUTPUT_FILE = "/media/nam/DATA/RAI_K63/HK9/NLP/comic-generation/data/full_story_formatted.csv"
    MODEL_PATH = "Qwen/Qwen3-8B"

    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: Không tìm thấy {INPUT_FILE}. Hãy chạy crawler.py trước hoặc kiểm tra lại đường dẫn Dataset.")
        return

    print(f"Đang tải Tokenizer và Model {MODEL_PATH} lên GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    PROMPT = """Nhiệm vụ của bạn là tóm tắt diễn biến truyện và viết lại truyện thành đúng 4 đoạn văn ngắn.
    Cấu trúc phải tuân thủ đầy đủ định dạng dưới đây.

    <|bos|>
    <|summary|> Tóm tắt đầy đủ diễn biến câu truyện theo trình tự trong 1 đến 2 câu tiếng Việt.
    <|paragraph_1|> Nội dung đoạn 1.
    <|paragraph_2|> Nội dung đoạn 2.
    <|paragraph_3|> Nội dung đoạn 3.
    <|paragraph_4|> Nội dung đoạn 4.
    <|eos|>

    Chỉ trả về các thẻ định dạng và nội dung bên trong. Không giải thích thêm."""

    @torch.inference_mode()
    def process_story(content):
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Câu truyện: {content}"}
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("Đang đọc dữ liệu thô...")
    df = pd.read_csv(INPUT_FILE)
    records = df.to_dict('records')
    new_records = []

    print("Bắt đầu chuẩn hóa dữ liệu...")
    for row_dict in tqdm(records, desc="Generating"):
        content = row_dict["content"]
        for i in range(1, 4):
            version_text = process_story(content)
            version_text = re.sub(r'<\|paragraph_[1-4]\|>', '<|paragraph|>', version_text)
            row_dict[f"version_{i}"] = version_text

        new_records.append(row_dict)

    final_df = pd.DataFrame(new_records)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nHoàn tất! Dữ liệu chuẩn hóa đã lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_normalization()
