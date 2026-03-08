# Linear Attention but for AWS Trainium Chips

Not in existing NKI library.


## Repo Setup
- Reserve a Trainium instance on AWS. Make sure you use an **Ubuntu Deep Learning** AMI.

- SSH into that machine (e.g. `ssh  -i ~/.ssh/willychan.pem ubuntu@ec2-100-54-67-174.compute-1.amazonaws.com`).

- Check everything works with `neuron-ls`.

- Run the following commands to make sure your python/PyTorch environment is set up correctly:
```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export NEURON_FRAMEWORK_DEBUG=1
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
```



## Running relevant scripts
```bash
python baseline_based.py
```
