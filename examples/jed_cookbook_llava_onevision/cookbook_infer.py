import argparse, os, json, time, math, sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# Centralized prompt presets. Most LLaVA-style models expect an <image> token.
PROMPTS = {
    "chart_qa": "You are a precise chart analyst. Given the chart, answer the question concisely.\n<image>\nQuestion: {q}\nAnswer:",
    "table_qa": "You are a table understanding assistant. Read the table carefully.\n<image>\nQuestion: {q}\nAnswer:",
    "document_ocr": "Extract the key text content from this document page. Return clean, readable text without hallucinations.\n<image>\nText:",
    "layout_reasoning": "Analyze the document layout (titles, sections, lists, tables, figures) and summarize the main points.\n<image>\nSummary:"
}

DEFAULT_QUESTIONS = {
    "chart_qa": "What is the main trend and the value at the last data point?",
    "table_qa": "What is the total of the last column and which row has the maximum value?",
    "document_ocr": "",
    "layout_reasoning": ""
}

def load_images(paths: List[str]) -> List[Image.Image]:
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        out.append(im)
    return out

def build_prompt(preset: str, user_question: str = None) -> str:
    if preset not in PROMPTS:
        raise ValueError(f"Unknown preset '{preset}'.")
    q = user_question
    if q is None or q.strip() == "":
        q = DEFAULT_QUESTIONS.get(preset, "")
    return PROMPTS[preset].format(q=q)

def pick_dtype(s: str):
    s = (s or "auto").lower()
    if s in ("auto", "float16", "fp16"): return torch.float16
    if s in ("bfloat16", "bf16"): return torch.bfloat16
    if s in ("float32", "fp32"): return torch.float32
    return torch.float16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", type=str, required=True, help="HF model ID or local path")    
    ap.add_argument("--images", nargs="+", required=True, help="Image paths or globs")    
    ap.add_argument("--preset", type=str, default="chart_qa", choices=list(PROMPTS.keys()))
    ap.add_argument("--question", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="runs/llava_cookbook")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="bfloat16|float16|float32|auto")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu or cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "answers").mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype(args.dtype)

    print(f"Loading model: {args.model_id} on {device} dtype={dtype}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    # expand globs
    image_paths: List[str] = []
    for pat in args.images:
        p = list(Path().glob(pat)) if any(ch in pat for ch in "*?[") else [Path(pat)]
        image_paths.extend([str(x) for x in p if x.is_file()])
    if not image_paths:
        print("No images found."); sys.exit(1)

    prompt = build_prompt(args.preset, args.question)
    results_path = out_dir / "results.jsonl"
    n_done = 0

    with results_path.open("w", encoding="utf-8") as f:
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            # Many LLaVA-style processors accept: processor(text=[prompt], images=[img], return_tensors="pt")
            inputs = processor(text=[prompt], images=[img], return_tensors="pt")
            if device.startswith("cuda"):
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            # Some processors supply a tokenizer; else AutoProcessor has one
            if hasattr(processor, "batch_decode") and callable(processor.batch_decode):
                text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
                text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            rec = {"image": img_path, "preset": args.preset, "prompt": prompt, "answer": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (out_dir/"answers"/ (Path(img_path).stem + ".md")).write_text(f"# Answer for {Path(img_path).name}\n\n{rec['answer']}\n", encoding="utf-8")
            n_done += 1
            print(f"[{n_done}/{len(image_paths)}] {img_path} -> done")

    print(f"Saved: {results_path}")

if __name__ == "__main__":
    main()
