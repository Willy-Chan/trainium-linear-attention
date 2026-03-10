# Linear Attention for AWS Trainium

Custom NKI kernels for BASED (polynomial feature map) causal linear attention on AWS Trainium.

## Repo Structure
```
kernels/
  pytorch_attention.py        # PyTorch reference + NKI wrapper
  nki_attention.py             # Parallel fused NKI kernel (O(N²), SPMD multi-core)
  nki_attention_chunked.py     # Chunked recurrent NKI kernel (O(N), HBM state)
  nki_attention_sliding.py     # Sliding window softmax NKI kernel (O(NW))
benchmarks/
  based_attention_benchmark.py       # Correctness & perf: parallel kernel
  chunked_attention_benchmark.py     # Correctness & perf: chunked kernel
  sliding_attention_benchmark.py     # Correctness & perf: sliding window kernel
  crossover_benchmark.py             # XLA vs traced dispatch across sequence lengths
  baremetal_benchmark.py             # Raw kernel latency via nki.benchmark
old/                           # Legacy/experimental files
```

## Setup
Reserve a Trainium instance on AWS with an **Ubuntu Deep Learning** AMI, then SSH in.

```bash
neuron-ls  # verify Neuron devices are visible
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PYTHONPATH=/home/ubuntu/trainium-linear-attention
```

## Running
```bash
python benchmarks/based_attention_benchmark.py
python benchmarks/chunked_attention_benchmark.py
python benchmarks/sliding_attention_benchmark.py
python benchmarks/crossover_benchmark.py
python benchmarks/baremetal_benchmark.py
```
