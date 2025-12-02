from tsrl.environments import generate_candle_features
from tsrl.experiments.market.experiment import MarketExperiment
from pathlib import Path
import numpy as np
import copy
import torch
from prettytable import PrettyTable
from plotly import graph_objects as go


# TIES-Merging class implementation
class TIESMerging:
    def __init__(self, model_list, init_model, top_k=20, scale=1.0):
        """
        model_list: List of fine-tuned models' parameters (each is a dictionary of tensors)
        init_model: Initial pretrained model's parameters (dictionary of tensors)
        top_k: Percentage of top-k parameters to keep during trimming (default is 20%)
        scale: Scaling factor for the final merge (lambda in the algorithm)
        """
        self.model_list = model_list
        self.init_model = init_model
        self.top_k = top_k
        self.scale = scale

    def create_task_vectors(self):
        """
        Create task vectors by subtracting the initial model parameters from each fine-tuned model's parameters.
        """
        task_vectors = []
        for model in self.model_list:
            task_vector = {}
            for key in model.keys():
                task_vector[key] = model[key] - self.init_model[key]
            task_vectors.append(task_vector)
        return task_vectors

    def trim(self, task_vectors):
        """
        Trim the task vectors by keeping only the top-k% largest-magnitude parameters.
        """
        trimmed_vectors = []
        for task_vector in task_vectors:
            trimmed_vector = {}
            for key in task_vector.keys():
                param = task_vector[key]
                # Flatten, get top-k% based on magnitude, and then reshape
                flat_param = param.view(-1)
                threshold = torch.quantile(torch.abs(flat_param), 1 - self.top_k / 100)
                trimmed_param = torch.where(torch.abs(flat_param) >= threshold, flat_param,
                                            torch.zeros_like(flat_param))
                trimmed_vector[key] = trimmed_param.view(param.shape)
            trimmed_vectors.append(trimmed_vector)
        return trimmed_vectors

    def elect_signs(self, trimmed_vectors):
        """
        Elect the sign for each parameter by majority across task vectors.
        """
        elected_signs = {}
        # Get the keys from the first task vector (assuming all models have the same keys)
        for key in trimmed_vectors[0].keys():
            # Stack the parameter values from each task vector for the current parameter key
            signs = torch.stack([torch.sign(tv[key]) for tv in trimmed_vectors], dim=0)
            # Elect the sign by taking the majority across models (sign with highest total magnitude)
            elected_sign = torch.sign(torch.sum(signs, dim=0))
            elected_signs[key] = elected_sign
        return elected_signs

    def merge(self, trimmed_vectors, elected_signs):
        """
        Merge task vectors using the elected signs and taking the mean of values with the same sign.
        """
        merged_vector = {}
        for key in trimmed_vectors[0].keys():
            aligned_params = []
            for task_vector in trimmed_vectors:
                param = task_vector[key]
                aligned_param = torch.where(torch.sign(param) == elected_signs[key], param, torch.zeros_like(param))
                aligned_params.append(aligned_param)
            merged_vector[key] = torch.mean(torch.stack(aligned_params, dim=0), dim=0)
        return merged_vector

    def merge_models(self):
        """
        Perform the full TIES-Merging process: Create task vectors, trim, elect signs, and merge.
        """
        task_vectors = self.create_task_vectors()
        trimmed_vectors = self.trim(task_vectors)
        elected_signs = self.elect_signs(trimmed_vectors)
        merged_task_vector = self.merge(trimmed_vectors, elected_signs)

        # Apply scaling and add to the initial model
        merged_model = {}
        for key in self.init_model.keys():
            merged_model[key] = self.init_model[key] + self.scale * merged_task_vector[key]

        return merged_model
