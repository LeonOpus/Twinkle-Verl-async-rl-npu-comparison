# Stage-2 final LoRA adapters (2026-08-04)

Qwen3-4B 上完成 125 optimizer steps、8000 trajectories 后的标准 PEFT LoRA。

| Framework | Directory | Weight bytes | SHA256 |
|---|---|---:|---|
| Twinkle | `twinkle_v125` | 132187856 | `5117558792aa38f91819e83fbe48a2cec5f488e9d1a10bd0bc8702c84edb9a69` |
| VERL | `verl_v125` | 66127744 | `f201c98beea1f0d609ef78db6e42e376129a298f70a43e213b2b11fc5ca397b7` |

权重由 Git LFS 保存；不包含基础模型、optimizer、完整 checkpoint 或合并模型。完整性见 `SHA256SUMS`。
