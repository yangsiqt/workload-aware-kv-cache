#!/usr/bin/env bash
set -euo pipefail

apt-get update -qq
apt-get install -y -qq ffmpeg git-lfs
git lfs install --skip-repo

ffmpeg -version | head -n 2
git lfs version
