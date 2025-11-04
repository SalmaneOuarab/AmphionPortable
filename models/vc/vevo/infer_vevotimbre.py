# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio

# -------------------- User Paths --------------------
# Path to your Vevo snapshot folder in Google Drive
VEVO_ROOT = "/content/drive/MyDrive/Vevo/snapshots/7edf4640c400c20542aa39c45b63f60e6c7baba0"

# Configs (from the original repo)
FMT_CFG_PATH = "./models/vc/vevo/config/Vq8192ToMels.json"
VOCODER_CFG_PATH = "./models/vc/vevo/config/Vocoder.json"

# Uploaded audio paths in Colab
CONTENT_WAV_PATH = "./uploaded_audio/source_feats.pt"
REFERENCE_WAV_PATH = "./uploaded_audio/arabic_male.wav"
OUTPUT_PATH = "./uploaded_audio/output_vevotimbre.wav"
# ----------------------------------------------------

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoints
TOKENIZER_CKPT_PATH = os.path.join(VEVO_ROOT, "tokenizer/vq8192")
FMT_CKPT_PATH = os.path.join(VEVO_ROOT, "acoustic_modeling/Vq8192ToMels")
VOCODER_CKPT_PATH = os.path.join(VEVO_ROOT, "acoustic_modeling/Vocoder")

# Initialize inference pipeline
inference_pipeline = VevoInferencePipeline(
    content_style_tokenizer_ckpt_path=TOKENIZER_CKPT_PATH,
    fmt_cfg_path=FMT_CFG_PATH,
    fmt_ckpt_path=FMT_CKPT_PATH,
    vocoder_cfg_path=VOCODER_CFG_PATH,
    vocoder_ckpt_path=VOCODER_CKPT_PATH,
    device=device,
)

# -------------------- Function --------------------
def vevo_timbre(content_wav_path, reference_wav_path, output_path):
    """
    Run Vevo timbre conversion on the input audio.
    """
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        flow_matching_steps=32,
    )
    save_audio(gen_audio, output_path)
    print(f"Saved output audio at: {output_path}")

# -------------------- Example --------------------
if __name__ == "__main__":
    vevo_timbre(CONTENT_WAV_PATH, REFERENCE_WAV_PATH, OUTPUT_PATH)
