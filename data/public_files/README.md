# QRneuron
This is the torch implementation of [Identifying Query-Relevant Neurons in Large Language Models for Long-Form Texts]. <br>
The code of computing neuron attribution is adapted from [EleutherAI](https://github.com/EleutherAI/knowledge-neurons)， and we thank them for their effort and contribution.

Functions
* extract QR neurons ✅
* edit knowledge ❌
* code examples ❌
* test cases ❌

# install
```
pip install -i https://test.pypi.org/simple/ keyneuron==0.0.2
```
[pypi link](https://test.pypi.org/project/keyneuron/0.0.2/#description)
# example

```python
import random
import json
from pathlib import Path
from keyneuron import KeyNeuron

random.seed(42)

DATA_PATH = Path("data/domain_sample_multi_choice_qa.json")
DATA_SAMPLE = json.load(open(DATA_PATH))
print("[ sample number = {a}]".format(a=len(DATA_SAMPLE)))

KN = KeyNeuron(
    model_name = "meta-llama/Llama-2-7b-chat-hf",
    data_samples = DATA_SAMPLE,
    result_dir = 'data/',
    common_threshold=0.3,
    top_v=20, 
    attr_threshold=0.2,
    option_letters = ["A", "B", "C", "D"],
    batch_size = 8,
    steps = 16,
)
# extract key neurons and store them in the result_dir
KN._extract_key_neuron()

```
After, you can find the key neuron file in `data/`


## Citation
If you find our paper and code useful, please give us a citation :blush:
```bibtex
@article{chen2024identifying,
  title={Identifying Query-Relevant Neurons in Large Language Models for Long-Form Texts},
  author={Chen, Lihu and Dejl, Adam and Toni, Francesca},
  journal={arXiv preprint arXiv:2406.10868},
  year={2024}
}
```

