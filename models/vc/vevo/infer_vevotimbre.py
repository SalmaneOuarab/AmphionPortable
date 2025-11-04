import os
import torch
from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio

# -------------------- Paths --------------------
# Models (Drive)
VEVO_ROOT = "/content/drive/MyDrive/Vevo/models--amphion--Vevo/snapshots/7edf4640c400c20542aa39c45b63f60e6c7baba0"

# Configs (repo)
FMT_CFG_PATH = "./models/vc/vevo/config/Vq8192ToMels.json"
VOCODER_CFG_PATH = "./models/vc/vevo/config/Vocoder.json"

# Audio files (repo)
CONTENT_WAV_PATH = "./models/vc/vevo/wav/source_feats.pt"
REFERENCE_WAV_PATH = "./models/vc/vevo/wav/arabic_male.wav"
OUTPUT_PATH = "./models/vc/vevo/wav/output_vevotimbre.wav"

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Model checkpoints --------------------
TOKENIZER_CKPT_PATH = os.path.join(VEVO_ROOT, "tokenizer/vq8192")
FMT_CKPT_PATH = os.path.join(VEVO_ROOT, "acoustic_modeling/Vq8192ToMels")
VOCODER_CKPT_PATH = os.path.join(VEVO_ROOT, "acoustic_modeling/Vocoder")

# -------------------- Initialize pipeline --------------------
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
