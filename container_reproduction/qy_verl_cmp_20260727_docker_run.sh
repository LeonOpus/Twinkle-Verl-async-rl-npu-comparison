#!/usr/bin/env bash
set -euo pipefail

# Reconstructed from docker inspect.
# Docker does not retain the originally typed docker run text.
# container_name=qy_verl_cmp_20260727
# container_id=c3e1170ae59a9357e0e512abd042baa67157a221e7f6d8272e0e0f28ea4a7975
# container_created=2026-07-27T05:08:01.646358757Z
# configured_image=quay.nju.edu.cn/ascend/verl:verl-9.0.0-910b-ubuntu22.04-py3.11-latest
# immutable_image_id=sha256:f58d5fbb02b21f5d2adb4713a5b9b9ac4edb929a0e62e3f5bd500fde8bf98825

docker run -i -t -d --name qy_verl_cmp_20260727 --hostname AI04 --privileged --runtime runc \
  --network host --ipc host --cgroupns private --shm-size 137438953472 --security-opt label=disable \
  --mount type=bind,src=/workspace,dst=/workspace --mount \
  type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly --mount \
  type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly --mount \
  type=bind,src=/mnt/lv_model,dst=/mnt/lv_model,bind-propagation=rslave --env WANDB_MODE=disabled \
  --env ASCEND_RT_VISIBLE_DEVICES=0,1 --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --entrypoint /bin/bash sha256:f58d5fbb02b21f5d2adb4713a5b9b9ac4edb929a0e62e3f5bd500fde8bf98825 -c \
  '    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&     source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh &&     source /usr/local/Ascend/nnal/atb/set_env.sh &&     exec "$@"' \
  -- bash
