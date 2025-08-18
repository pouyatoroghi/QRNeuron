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

def tensors_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Check if two tensors are equal (with tolerances for floating point comparisons)
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance (for float comparisons)
        atol: Absolute tolerance (for float comparisons)
    
    Returns:
        bool: True if tensors are equal within tolerances
    """
    # Check if shapes match
    if tensor1.shape != tensor2.shape:
        return False
    
    # Check if dtypes match
    if tensor1.dtype != tensor2.dtype:
        return False
    
    # Exact comparison for integer/boolean types
    if tensor1.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
        return torch.equal(tensor1, tensor2)
    
    # Approximate comparison for floating point types
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

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
       
        # self.option_letters = option_letters
        # self.OPTION_IDS = [self.tokenizer.convert_tokens_to_ids(o) for o in self.option_letters]
        
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

    def _prepare_inputs(self, prompt, gold=None, encoded_input=None):
        """
        Prepare model inputs and process gold tokens if provided.
    
        Args:
            prompt: Input text string
            gold: Optional gold text to tokenize (default: None)
            encoded_input: Pre-encoded inputs (optional)
        
        Returns:
            tuple: (encoded_input, gold_ids, gold_len)
                encoded_input: Tokenized input dict
                gold_ids: Tensor of gold token IDs (None if no gold)
                gold_len: Length of gold tokens (0 if no gold)
        """
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
        gold_ids = None
        gold_len = 0
    
        if gold is not None:
            gold_encoded = self.tokenizer(gold, return_tensors="pt").to(self.model.device)
            gold_ids = gold_encoded['input_ids']#[0]  # Get first sequence
            gold_len = gold_ids.shape[0]  # Get length
    
        return encoded_input, gold_ids, gold_len

    # def _generate(self, prompt, ground_truth):
    #     encoded_input, mask_idx, target_label = self._prepare_inputs(
    #         prompt, ground_truth
    #     )

    #     n_sampling_steps = 1  
    #     all_gt_probs = []
    #     all_argmax_probs = []
    #     argmax_tokens = []
    #     argmax_completion_str = ""

    #     for i in range(n_sampling_steps):
    #         if i > 0:
    #             # retokenize new inputs
    #             encoded_input, mask_idx, target_label = self._prepare_inputs(
    #                 prompt, ground_truth
    #             )
    #         outputs = self.model(**encoded_input)
    #         probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
    #         target_idx = target_label
    #         gt_prob = probs[:, target_idx].item()
    #         all_gt_probs.append(gt_prob)

    #         # get info about argmax completion
    #         option_probs = [(option_id, probs[:, option_id].item()) for option_id in self.OPTION_IDS]
    #         argmax_id, argmax_prob = sorted(option_probs, key= lambda e:e[1], reverse=True)[0]
    #         argmax_tokens.append(argmax_id)
    #         argmax_str = self.tokenizer.decode([argmax_id])
            
    #         all_argmax_probs.append(argmax_prob)
    #         prompt += argmax_str
    #         argmax_completion_str += argmax_str

    #     gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
    #     argmax_prob = (
    #         math.prod(all_argmax_probs)
    #         if len(all_argmax_probs) > 1
    #         else all_argmax_probs[0]
    #     )
    #     return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

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
        # tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        # out = (
        #     tiled_activations
        #     * torch.linspace(start=0, end=1, steps=steps).to(tiled_activations.device)[:, None]
        # )

        # scale_factors = torch.linspace(0, 1, steps, device=activations.device).view(-1, 1, 1)  # [steps, 1, 1]\
        # print((activations.unsqueeze(0) * scale_factors).shape, out.shape)
        # aa = (activations.unsqueeze(0) * scale_factors).reshape(-1, activations.size(-1))
        # print(tensors_equal(aa, out))
        scale_factors = torch.linspace(0, 1, steps, device=activations.device).view(-1, 1)  # [steps, 1]
        # print((activations * scale_factors).shape, out.shape)
        # aa = (activations * scale_factors)
        return activations * scale_factors
        # print("Main", tensors_equal(aa, out))
        
        # return (activations.unsqueeze(0) * scale_factors).reshape(-1, activations.size(-1))  # [steps * b, d]
        # return out

    def get_baseline_with_activations(
        self, 
        encoded_input: dict, 
        layer_idx: int, 
        generation_length: int
    ):
        """
        Gets baseline outputs and raw activations during generation using past_key_values.
    
        Args:
            encoded_input: {'input_ids': torch.Tensor, 'attention_mask': torch.Tensor}
            layer_idx: Layer index to monitor
            generation_length: Number of new tokens to generate
        
        Returns:
            tuple: (final_outputs, activations_dict)
                final_outputs: Model outputs after generation
                activations_dict: {position: activations_tensor} for each generated position
        """
        activations_dict = {}
        original_length = encoded_input['input_ids'].shape[1]
        past_key_values = None
    
        # Hook setup - this will capture the raw activations you want
        def get_activation_hook(position):
            def hook_fn(acts):
                # Only store the activations for the newest token
                activations_dict[position] = acts[:, -1, :].detach()
            return hook_fn
    
        # Initial setup
        handle = register_hook(
            self.model,
            layer_idx=layer_idx,
            f=get_activation_hook(0),
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )
    
        with torch.no_grad():
            # First forward pass
            outputs = self.model(**encoded_input, use_cache=True)
            past_key_values = outputs.past_key_values
        
            # Sequential generation
            for i in range(1, generation_length):
                current_pos = i
                handle.remove()  # Remove previous hook
            
                # Register new hook for current position
                handle = register_hook(
                    self.model,
                    layer_idx=layer_idx,
                    f=get_activation_hook(current_pos),
                    transformer_layers_attr=self.transformer_layers_attr,
                    ff_attrs=self.input_ff_attr,
                )
            
                # Generate next token using cached KV
                next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=torch.cat([
                        encoded_input['attention_mask'],
                        torch.ones_like(next_token)
                    ], dim=1),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
    
        handle.remove()  # Clean up final hook
    
        # Memory management
        del encoded_input, past_key_values, outputs
        torch.cuda.empty_cache()
    
        return activations_dict
    
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
        # encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                # encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps
            )
            # scores.append(layer_scores)
        # temp_device = scores[0].device
        # scores = [score.to(temp_device) for score in scores]

        # return torch.stack(scores)

            # Move to CPU immediately to free GPU memory
            scores.append(layer_scores.detach().cpu())
            torch.cuda.empty_cache()

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
        for (prompt, ground_truth) in tqdm(
            zip(prompts, ground_truths), desc="Getting integrated gradients for each prompt...", disable=False#quiet
        ):
            attribution_scores = self.get_scores(
                prompt,
                ground_truth,
                batch_size=batch_size,
                steps=steps,
                pbar=False,
            )#.detach().cpu()
            # temp_dict = dict()
            # for neuron in neurons:
            #     score = attribution_scores[neuron[0], neuron[1]].item()
            #     temp_dict[str(neuron[0]) + "__" + str(neuron[1])] = score

            temp_dict = {f"{neuron[0]}__{neuron[1]}": attribution_scores[neuron[0], neuron[1]].item() for neuron in neurons}
            
            grads[ground_truth] = temp_dict

            # Free memory immediately after each prompt
            del attribution_scores
            torch.cuda.empty_cache()
            
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
        for (prompt, ground_truth) in tqdm(
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

            del attribution_scores
            torch.cuda.empty_cache()

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

        n_sampling_steps = 1

        # Prepare inputs
        encoded_input, gold_ids, gold_len = self._prepare_inputs(prompt, ground_truth, encoded_input)

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
            
        # Get baseline activations for ALL positions
        baseline_activations_dict = self.get_baseline_with_activations(encoded_input, layer_idx, gold_len)

        final_integrated_grads = None

        new_mask = torch.full((batch_size, 1), 1)
        
        for pos in range(gold_len):    
            # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
            # and calculate the integrated gradient of the masked position at each step
            # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
            baseline_activations = baseline_activations_dict[pos]
            
            scaled_weights = self.scaled_input(baseline_activations, steps=steps)
            scaled_weights.requires_grad_(True)

            # Initialize accumulator for integrated gradients
            integrated_grads_accumulator = None
            
            for i in range(n_sampling_steps):
                
                # Initialize accumulator for this step's gradients
                step_grads_accumulator = None

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=-1,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, -1, :], dim=-1)

                    print(probs[:, gold_ids[pos]])

                    print(torch.unbind(probs[:, gold_ids[pos]]))
  
                    grad = torch.autograd.grad(torch.unbind(probs[:, gold_ids[pos]]), batch_weights)[0]

                    grad = grad.sum(dim=0)

                    if step_grads_accumulator is None:
                        step_grads_accumulator = grad
                    else:
                        step_grads_accumulator += grad
                    # integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # Free GPU memory immediately after each batch
                    del outputs, probs, grad
                    torch.cuda.empty_cache()

                # then sum, and multiply by W-hat / m
                # integrated_grads_this_step = torch.stack(integrated_grads_this_step, dim=0).sum(dim=0)
                step_grads_accumulator *= baseline_activations.squeeze(0) / steps
                # integrated_grads.append(integrated_grads_this_step)
                if integrated_grads_accumulator is None:
                    integrated_grads_accumulator = step_grads_accumulator
                else:
                    integrated_grads_accumulator += step_grads_accumulator

                # Free memory after each sampling step
                del baseline_outputs, baseline_activations, scaled_weights, step_grads_accumulator
                torch.cuda.empty_cache()

            # Average the accumulated gradients
            integrated_grads_accumulator /= n_sampling_steps

            if final_integrated_grads:
                final_integrated_grads += integrated_grads_accumulator
            else:
                final_integrated_grads = integrated_grads_accumulator

            # Free GPU memory immediately after each batch
            del integrated_grads_accumulator
            torch.cuda.empty_cache()

            new_token_ids = torch.full((batch_size, 1), gold_ids[pos])
            
            # Append new token from the gold to input_ids and attention_mask
            inputs["input_ids"] = torch.cat([
                                            inputs["input_ids"], 
                                            new_token_ids
                                            ], dim=1)  # Shape: (batch, seq_len + 1)

            inputs["attention_mask"] = torch.cat([
                                                inputs["attention_mask"],
                                                new_mask  # 1 for new tokens
                                                ], dim=1)  # Shape: (batch, seq_len + 1)
            
        return final_integrated_grads

