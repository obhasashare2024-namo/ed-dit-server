#!/usr/bin/env python3
"""
Convert ComfyUI MiniMax H3 Stage 1 .pt bundle to high-performance .edbin format for ed-dit-server.
"""
import sys
import struct
import torch

def export_stage1(pt_path, out_edbin_path):
    print(f"Loading {pt_path}...")
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    meta = data.get("meta", {})
    context = data["context"].float().contiguous()
    token_tags = data["token_tags"].float().contiguous()
    vid_lat = data["latent"].float().contiguous()
    aud_lat = data["audio_latent"].float().contiguous()
    kf_lat = data["kf_latent"].float().contiguous()
    kf_idx = int(data.get("kf_idx", 0))

    width = int(meta.get("width", 768))
    height = int(meta.get("height", 448))
    length = int(meta.get("length", 73))
    steps = int(meta.get("steps", 6))
    video_shift = float(meta.get("video_shift", 12.0))
    audio_shift = float(meta.get("audio_shift", 3.0))

    with open(out_edbin_path, "wb") as f:
        f.write(b"H3STG1\x01\x00")
        f.write(struct.pack("<iiii ff", width, height, length, steps, video_shift, audio_shift))

        def write_tensor(t):
            f.write(struct.pack("<i", t.ndim))
            for dim in t.shape:
                f.write(struct.pack("<q", dim))
            f.write(t.numpy().tobytes())

        write_tensor(context)
        write_tensor(token_tags)
        write_tensor(vid_lat)
        write_tensor(aud_lat)
        write_tensor(kf_lat)
        f.write(struct.pack("<i", kf_idx))

    print(f"Successfully wrote {out_edbin_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: export_stage1_for_ed.py <stage1_input.pt> <output.edbin>")
        sys.exit(1)
    export_stage1(sys.argv[1], sys.argv[2])
