#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, for example: export PROJECT_ID=my-gcp-project}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-tesis-demo-gpu}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-200GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"

gcloud config set project "${PROJECT_ID}"

gcloud compute instances create "${VM_NAME}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --metadata="install-nvidia-driver=True"

echo
echo "VM created: ${VM_NAME} (${ZONE})"
echo "SSH with:"
echo "  gcloud compute ssh ${VM_NAME} --zone ${ZONE}"
