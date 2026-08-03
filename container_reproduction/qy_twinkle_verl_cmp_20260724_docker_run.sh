#!/usr/bin/env bash
set -euo pipefail

# Reconstructed from docker inspect.
# Docker does not retain the originally typed docker run text.
# container_name=qy_twinkle_verl_cmp_20260724
# container_id=2db727e975c6091334c44a2276c765fc50962d3b3a3b5025a2e0f21368019c7c
# container_created=2026-07-24T09:33:44.060342377Z
# configured_image=swr.cn-southwest-2.myhuaweicloud.com/ascend-sact/twinkle-npu-a2:v5
# immutable_image_id=sha256:7ce36d0558311f40a1e99b753be34ececa343800dd93a1425da8ed9142c43d31

docker run -i -t -d --name qy_twinkle_verl_cmp_20260724 --hostname AI04 --workdir /workspace \
  --privileged --runtime runc --network host --ipc host --cgroupns private --shm-size 137438953472 \
  --security-opt label=disable --mount \
  type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly --mount \
  type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly --mount \
  type=bind,src=/mnt/lv_model,dst=/mnt/lv_model,bind-propagation=rslave --mount \
  type=bind,src=/workspace,dst=/workspace --env ASCEND_RT_VISIBLE_DEVICES=0,1,2 --env \
  PYTORCH_NPU_ALLOC_CONF=expandable_segments:True --entrypoint /bin/bash \
  sha256:7ce36d0558311f40a1e99b753be34ececa343800dd93a1425da8ed9142c43d31 -c \
  '    source /usr/local/Ascend/ascend-toolkit/set_env.sh &&     source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh &&     source /usr/local/Ascend/nnal/atb/set_env.sh &&     exec "$@"' \
  -- bash
