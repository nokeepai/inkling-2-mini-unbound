#!/bin/bash
# Download the r16 LoRA files (GGUF + PEFT) from the build pod.
set -e
L="bash /Users/nikshepsvn/Code/inkling-abl/scripts/ink_lium.sh"
O=/Users/nikshepsvn/Code/inkling-abl/out/final/lora_r16
mkdir -p "$O/peft"
$L scp ink-gguf-1 /data/release/Inkling-2-Mini-Unbound-LoRA-r16.gguf "$O/Inkling-2-Mini-Unbound-LoRA-r16.gguf" --download > /dev/null
$L scp ink-gguf-1 /data/release/lora/peft-r16/adapter_config.json "$O/peft/adapter_config.json" --download > /dev/null
$L scp ink-gguf-1 /data/release/lora/peft-r16/adapter_model.safetensors "$O/peft/adapter_model.safetensors" --download > /dev/null
ls -la "$O" "$O/peft"
shasum -a 256 "$O/Inkling-2-Mini-Unbound-LoRA-r16.gguf" "$O/peft/adapter_model.safetensors"
