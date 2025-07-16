import logging
from typing import List, Optional, Tuple, Callable
import collections
import math
import einops
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM
from ._patch import *
from ._config import ATTRIBUTION_CONFIG


# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

class NeuronAtrribution:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        option_letters: List[str] = ["A", "B", "C", "D"]
    ):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        finally:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.eval()
        logger.info(self.model)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("[ total number of parameters = {a} ]".format(a=pytorch_total_params))

        self.model_name = model_name
       
        self.option_letters = option_letters
        self.OPTION_IDS = [self.tokenizer.convert_tokens_to_ids(o) for o in self.option_letters]
        
        
        self.baseline_activations = None
        
        
        if "gpt" in self.model_name :
            self.model_type = "gpt"
        else: self.model_type = "glu_model"
        
        self.transformer_layers_attr = ATTRIBUTION_CONFIG[self.model_type]["transformer_layers_attr"]
        self.input_ff_attr = ATTRIBUTION_CONFIG[self.model_type]["input_ff_attr"]
        self.output_ff_attr = ATTRIBUTION_CONFIG[self.model_type]["output_ff_attr"]


    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        mask_idx = -1
        if target is not None:
            target = self.tokenizer.convert_tokens_to_ids(target)
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )

        n_sampling_steps = 1  
        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_idx = target_label
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            option_probs = [(option_id, probs[:, option_id].item()) for option_id in self.OPTION_IDS]
            argmax_id, argmax_prob = sorted(option_probs, key= lambda e:e[1], reverse=True)[0]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            
            all_argmax_probs.append(argmax_prob)
            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())



    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
        )
        return out

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )
 
        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
      
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def get_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        pbar: bool = False,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        """

        scores = []
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps
            )
            scores.append(layer_scores)
        temp_device = scores[0].device
        scores = [score.to(temp_device) for score in scores]

        return torch.stack(scores)

    
    def get_integrated_gradients(
        self,
        neurons,
        prompts: List[str],
        ground_truths: List[str],
        batch_size: int = 10,
        steps: int = 20,
        quiet=False,
    ):
        """
        Calulates integrated gradients for a given prompt and ground truth.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        
        """
        
        grads = dict()
        for (prompt,ground_truth) in tqdm(
            zip(prompts, ground_truths), desc="Getting integrated gradients for each prompt...", disable=quiet
        ):
            attribution_scores = self.get_scores(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                pbar=False,
            ).detach().cpu()
            temp_dict = dict()
            for neuron in neurons:
                score = attribution_scores[neuron[0], neuron[1]].item()
                temp_dict[str(neuron[0]) + "__" + str(neuron[1])] = score
            grads[ground_truth] = temp_dict
        return grads
            

    def get_neuron_attribution(
        self,
        prompts: List[str],
        ground_truths: List[str],
        batch_size: int = 10,
        steps: int = 20,
        threshold: Optional[float] = 0.3,
        quiet=False,
    ) -> List[List[int]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.

        `prompts`: list of str
            the prompts to get neuron attributions
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            threshold for the neuron attribution

        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get neuron attributon scores"

        neurons, sum_neuron_attr_scores, neuron_freq = [], dict(), dict()
        for (prompt,ground_truth) in tqdm(
            zip(prompts, ground_truths), desc="Getting the neuron attribution score for each prompt...", disable=quiet
        ):
            
            attribution_scores = self.get_scores(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
            )
            
            adaptive_threshold = attribution_scores.max().item() * threshold
            indexes = torch.nonzero(attribution_scores > adaptive_threshold)
            neuron_attr_scores = dict([((index[0].item(), index[1].item()), round(attribution_scores[index[0], index[1]].item(), 6)) for index in indexes])
            
            for neuron, score in neuron_attr_scores.items():
                if neuron not in neurons:
                    neurons.append(neuron)
                
                if neuron not in sum_neuron_attr_scores:
                    sum_neuron_attr_scores[neuron] = 0
                sum_neuron_attr_scores[neuron] += score
                
                if neuron not in neuron_freq:
                    neuron_freq[neuron] = 0
                neuron_freq[neuron] += 1
                

        final_neuron_attr_scores = [sum_neuron_attr_scores[tuple(n)] for n in neurons]
        final_neuron_freq = [neuron_freq[tuple(n)] for n in neurons]
        assert len(neurons) == len(final_neuron_attr_scores) == len(final_neuron_freq)
        return neurons, final_neuron_attr_scores, final_neuron_freq


    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        n_sampling_steps = 1  
        integrated_grads = []

        for i in range(n_sampling_steps):
            
             
            encoded_input, mask_idx, target_label = self._prepare_inputs(
                prompt, ground_truth
            )
            (
                baseline_outputs,
                baseline_activations,
            ) = self.get_baseline_with_activations(
                encoded_input, layer_idx, mask_idx
            )
            

            # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
            # and calculate the integrated gradient of the masked position at each step
            # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
            scaled_weights = self.scaled_input(
                baseline_activations, steps=steps, 
            )
            scaled_weights.requires_grad_(True)

            integrated_grads_this_step = []  # to store the integrated gradients

            for batch_weights in scaled_weights.chunk(n_batches):
                # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                # first tile the inputs to the correct batch size
                inputs = {
                    "input_ids": einops.repeat(
                        encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                    ),
                    "attention_mask": einops.repeat(
                        encoded_input["attention_mask"],
                        "b d -> (r b) d",
                        r=batch_size,
                    ),
                }
                # then patch the model to replace the activations with the scaled activations
                patch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    mask_idx=mask_idx,
                    replacement_activations=batch_weights,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

                # then forward through the model to get the logits
                outputs = self.model(**inputs)

                # then calculate the gradients for each step w/r/t the inputs
                probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                if n_sampling_steps > 1:
                    target_idx = target_label[i]
                else:
                    target_idx = target_label
                grad = torch.autograd.grad(
                    torch.unbind(probs[:, target_idx]), batch_weights
                )[0]

                grad = grad.sum(dim=0)
                integrated_grads_this_step.append(grad)

                unpatch_ff_layer(
                    self.model,
                    layer_idx=layer_idx,
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )

            # then sum, and multiply by W-hat / m
            integrated_grads_this_step = torch.stack(
                integrated_grads_this_step, dim=0
            ).sum(dim=0)
            integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
            integrated_grads.append(integrated_grads_this_step)

  
        integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
            integrated_grads
        )
        return integrated_grads
    
