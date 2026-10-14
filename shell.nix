{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [
    pkgs.nodejs
    (pkgs.python3.withPackages (ps: with ps; [
      torch
      torchvision
      pillow
      einops
      omegaconf
      safetensors
      huggingface-hub
      onnx
      onnxruntime
      onnxscript
      numpy
      accelerate
    ]))
  ];
}
