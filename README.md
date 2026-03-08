# Linear Attention for AWS Trainium

Custom NKI kernels for BASED (polynomial feature map) causal linear attention on AWS Trainium.

## Repo Structure
```
kernels/
  pytorch_attention.py      # PyTorch reference + NKI wrapper
  nki_attention.py           # Parallel fused NKI kernel
  nki_attention_chunked.py   # Chunked recurrent NKI kernel
benchmarks/
  based_attention_benchmark.py     # Correctness & perf: PyTorch vs NKI
  chunked_attention_benchmark.py   # Correctness & perf: chunked NKI
old/                         # Legacy/experimental files
```

## Setup
Reserve a Trainium instance on AWS with an **Ubuntu Deep Learning** AMI, then SSH in.

```bash
neuron-ls  # verify Neuron devices are visible
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export NEURON_FRAMEWORK_DEBUG=1
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
```

## Running
```bash
python benchmarks/based_attention_benchmark.py
python benchmarks/chunked_attention_benchmark.py
```
