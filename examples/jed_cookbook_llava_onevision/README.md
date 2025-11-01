# Jed’s LLaVA-OneVision Cookbook (Charts & Docs Inference)

This mini-folder adds **ready-to-run inference scripts** for LLaVA-OneVision style VLMs on charts, tables, and document pages.

It uses pure 🤗 Transformers (no extra training code). You pass a **model ID** that is compatible with the Transformers VLM API (e.g., *llava-* models exported to HF).

> Works on CPU for a smoke test; recommended: **GPU** (bf16/flash-attn if available).

---

## Quickstart

```bash
conda env create -f env.yml && conda activate llava-cookbook

# Single image, chart question
python cookbook_infer.py   --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf   --images ./samples/chart1.png   --preset chart_qa   --out-dir ./runs/chart_demo
```

- `--model-id`: any HF model compatible with `AutoProcessor` + vision-language generation.
- `--images`: path(s) or glob(s) to images; you can provide multiple.
- `--preset`: one of `chart_qa`, `table_qa`, `document_ocr`, `layout_reasoning`.
- Outputs: `runs/.../results.jsonl` and per-image markdown dumps in `runs/.../answers/`.

> If your model requires a different chat template or image token, adjust `PROMPTS` in `cookbook_infer.py` (it’s centralized).

---

## Examples

### Document OCR-ish extraction
```bash
python cookbook_infer.py   --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf   --images ./samples/doc_page.png   --preset document_ocr   --out-dir ./runs/doc_ocr
```

### Table QA
```bash
python cookbook_infer.py   --model-id llava-hf/llava-onevision-qwen2-7b-ov-hf   --images ./samples/table.png   --preset table_qa   --out-dir ./runs/table_qa
```

---

## Tips for better results

- Prefer **bf16** on recent GPUs: add `--dtype bfloat16` and run with CUDA.
- Use **longer max tokens** for dense docs: `--max-new-tokens 512` or more.
- Batch multiple pages by passing a glob: `--images './pdf_pages/*.png'`.
- If your model expects a special image token (e.g., `<image>`), the script inserts it for you based on the preset.

---

## Folder structure

```
examples/jed_cookbook_llava_onevision/
├── README.md
├── env.yml
├── cookbook_infer.py
└── prompts.yaml         # (optional; you can add your own presets here later)
```

> Place your test images under `samples/` (not included).

---

## Known compatibility

This script targets HF-converted LLaVA/OneVision-style checkpoints that support `AutoProcessor` + `generate` with image inputs.  
If your exact model name differs, just swap `--model-id`. If the processor returns a different expected input (e.g. `pixel_values` vs `images`), the script auto-detects common field names.
