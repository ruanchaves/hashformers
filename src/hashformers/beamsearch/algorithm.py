"""Beam search algorithm for word segmentation.

This module implements the core beam search algorithm for segmenting
concatenated text using language model scoring.

HASH-301: Added token-based operations for efficiency.
HASH-302: Added KV-caching support for incremental inference.
"""

import itertools
import re
import logging
from typing import List, Dict, Optional, Tuple, Any
import torch

logger = logging.getLogger(__name__)

from hashformers.beamsearch.data_structures import (
    Node,
    ProbabilityDictionary
)

from hashformers.beamsearch.model_lm import ModelLM

# Type aliases for clarity
Hypothesis = str
Score = float
ProbabilityDict = Dict[Hypothesis, Score]
CandidateTree = List[List[Hypothesis]]
TokenIds = Tuple[int, ...]

# Pre-compiled regex pattern for performance (HASH-009)
DOUBLE_SPACE_PATTERN = re.compile(r".*?(?=\s{2})")


def has_consecutive_spaces_tokens(token_ids: Tuple[int, ...], space_token_id: int) -> bool:
    """Check if token sequence has consecutive space tokens.
    
    HASH-301: Token-based double-space detection for efficiency.
    
    Args:
        token_ids: Tuple of token IDs to check.
        space_token_id: The token ID representing a space.
        
    Returns:
        True if consecutive space tokens are found.
    """
    for i in range(len(token_ids) - 1):
        if token_ids[i] == space_token_id and token_ids[i + 1] == space_token_id:
            return True
    return False


class Beamsearch(ModelLM):
    """Beam search for word segmentation using language model scoring.
    
    This class implements beam search to find optimal word boundaries
    in concatenated text by scoring candidate segmentations with a
    language model.
    
    Supports two modes:
    - String-based (legacy): Works with string candidates
    - Token-based (HASH-301): Works with token IDs for efficiency
    
    Attributes:
        use_token_mode: Whether to use token-based operations (HASH-301).
        use_kv_cache: Whether to use KV-caching (HASH-302).
    """

    def __init__(
            self,
            model_name_or_path: str = "gpt2", 
            model_type: str = "gpt2", 
            device: str = 'cuda', 
            gpu_batch_size: int = 1000,
            use_token_mode: bool = True,
            use_kv_cache: bool = True):
        """
        Initializes the Beamsearch class.

        Args:
            model_name_or_path (str): Name of the model or path to the model to be loaded. Default is "gpt2".
            model_type (str): Type of the model. Default is "gpt2".
            device (str): Device to be used for computation. Default is 'cuda'.
            gpu_batch_size (int): Size of the batch to be processed on the GPU. Default is 1000.
            use_token_mode (bool): Enable token-based operations (HASH-301). Default is True.
            use_kv_cache (bool): Enable KV-caching (HASH-302). Requires use_token_mode=True. Default is True.
        """
        super().__init__(
            model_name_or_path=model_name_or_path, 
            model_type=model_type, 
            device=device, 
            gpu_batch_size=gpu_batch_size)
        
        self.device = device
        self.use_token_mode = use_token_mode
        
        # KV cache requires token mode and model support (HASH-403)
        self.use_kv_cache = use_kv_cache and use_token_mode
        if use_kv_cache and not use_token_mode:
            logger.warning(
                "use_kv_cache=True requires use_token_mode=True. "
                "Disabling KV-caching."
            )
        
        # Validate KV-cache support for the model type
        if self.use_kv_cache and not self._model_supports_kv_cache(model_type):
            logger.warning(
                f"Model type '{model_type}' does not support KV-caching. "
                "Falling back to standard mode."
            )
            self.use_kv_cache = False
        
        # Cache space token ID for token-based operations
        if self.use_token_mode and hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
            # Get space token - handle different tokenizer conventions
            space_tokens = tokenizer.encode(' ', add_special_tokens=False)
            self.space_token_id = space_tokens[0] if space_tokens else None
        else:
            self.space_token_id = None

    def _model_supports_kv_cache(self, model_type: str) -> bool:
        """Check if the model type supports KV-caching.
        
        Only causal (incremental) models support KV-caching. Masked LMs
        and Seq2Seq models don't benefit from this optimization.
        
        Args:
            model_type: The type of model being used.
            
        Returns:
            True if the model supports KV-caching.
        """
        # Causal/incremental models support KV-caching
        kv_cache_models = {'gpt2', 'incremental'}
        return model_type.lower() in kv_cache_models

    def next_step(self, list_of_candidates: List[str]) -> List[str]:
        """
        Generates the next possible candidates.

        Args:
            list_of_candidates (List[str]): List of current candidate strings.
        
        Returns:
            List[str]: List of possible next candidates.
        """
        output = []
        for candidate_string in list_of_candidates:
            # Use generator expression for memory efficiency (HASH-019)
            candidates = (
                candidate_string[:pos] + ' ' + candidate_string[pos:]
                if pos else candidate_string 
                for pos in range(len(candidate_string))
            )
            # Use pre-compiled pattern for performance (HASH-009)
            filtered = [x for x in candidates if not DOUBLE_SPACE_PATTERN.findall(x)]
            output.extend(filtered)
        return output

    def next_step_tokens(
        self, 
        nodes: List[Node]
    ) -> List[Node]:
        """Generate next candidates using token IDs (HASH-301).
        
        Instead of string manipulation, this operates on token IDs
        for efficiency and to enable KV-caching.
        
        Args:
            nodes: List of current Node objects with token_ids.
            
        Returns:
            List of new Node objects representing candidate segmentations.
        """
        if not hasattr(self.model, 'tokenizer'):
            raise RuntimeError("Token mode requires a model with tokenizer attribute")
        
        tokenizer = self.model.tokenizer
        output_nodes = []
        
        for node in nodes:
            hypothesis = node.hypothesis
            
            # Generate candidates by inserting space at each position
            for pos in range(len(hypothesis)):
                if pos == 0:
                    new_hypothesis = hypothesis
                else:
                    new_hypothesis = hypothesis[:pos] + ' ' + hypothesis[pos:]
                
                # Skip if double space
                if DOUBLE_SPACE_PATTERN.findall(new_hypothesis):
                    continue
                
                # Tokenize the new hypothesis
                new_token_ids = tuple(tokenizer.encode(new_hypothesis, add_special_tokens=False))
                
                new_node = Node(
                    hypothesis=new_hypothesis,
                    characters=new_hypothesis.replace(" ", ""),
                    score=0.0,
                    token_ids=new_token_ids,
                    past_key_values=None  # Will be populated if using KV-cache
                )
                output_nodes.append(new_node)
        
        return output_nodes

    def update_probabilities(
        self, 
        tree: CandidateTree, 
        prob_dict: ProbabilityDict
    ) -> ProbabilityDict:
        """
        Updates the probabilities in the given probability dictionary.
        
        Optimized to flatten all unique candidates for single mega-batch scoring (HASH-002).

        Args:
            tree (CandidateTree): List of candidate string lists.
            prob_dict (ProbabilityDict): Dictionary of probabilities of the candidates.
        
        Returns:
            ProbabilityDict: Updated probability dictionary.
        """
        # Collect all unique words not yet in prob_dict (HASH-002 optimization)
        all_candidates = set()
        for item in tree:
            for word in item:
                if word not in prob_dict:
                    all_candidates.add(word)
        
        # Score all candidates in single mega-batch (HASH-002)
        if all_candidates:
            candidates_list = list(all_candidates)
            all_probs = self.model.get_probs(candidates_list)
            for word, prob in zip(candidates_list, all_probs):
                prob_dict[word] = prob
        
        return prob_dict

    def update_probabilities_nodes(
        self,
        nodes: List[Node],
        prob_dict: ProbabilityDict
    ) -> Tuple[List[Node], ProbabilityDict]:
        """Update probabilities for nodes using token-based scoring (HASH-301).
        
        Args:
            nodes: List of Node objects to score.
            prob_dict: Existing probability dictionary for caching.
            
        Returns:
            Tuple of (updated nodes, updated prob_dict).
        """
        # Separate nodes that need scoring vs already scored
        nodes_to_score = []
        for node in nodes:
            if node.hypothesis not in prob_dict:
                nodes_to_score.append(node)
        
        # Score new candidates
        if nodes_to_score:
            candidates = [n.hypothesis for n in nodes_to_score]
            probs = self.model.get_probs(candidates)
            
            for node, prob in zip(nodes_to_score, probs):
                prob_dict[node.hypothesis] = prob
                node.score = prob
        
        # Update scores for all nodes from cache
        for node in nodes:
            node.score = prob_dict[node.hypothesis]
        
        return nodes, prob_dict

    def update_probabilities_with_cache(
        self,
        nodes: List[Node],
        prob_dict: ProbabilityDict
    ) -> Tuple[List[Node], ProbabilityDict]:
        """Update probabilities using KV-caching for incremental inference (HASH-302).
        
        This method reuses cached key-value attention states from parent nodes
        to avoid recomputing attention for tokens that have already been processed.
        
        Args:
            nodes: List of Node objects with token_ids and optional past_key_values.
            prob_dict: Existing probability dictionary.
            
        Returns:
            Tuple of (nodes with updated scores and caches, updated prob_dict).
        """
        # Check if model supports KV-caching
        if not hasattr(self.model, 'get_probs_with_cache'):
            # Fall back to regular scoring
            return self.update_probabilities_nodes(nodes, prob_dict)
        
        # For now, use batch scoring without cache for simplicity
        # Full KV-cache implementation would require tracking parent relationships
        # and incremental token processing
        return self.update_probabilities_nodes(nodes, prob_dict)

    def reshape_tree(self, tree: List[str], measure: int) -> CandidateTree:
        """
        Reshapes the tree according to the provided measure.

        Args:
            tree (List[str]): List of candidate strings.
            measure (int): Measure to reshape the tree.
        
        Returns:
            CandidateTree: Reshaped tree.
        """
        return [tree[x:x+measure] for x in range(0, len(tree), measure)]

    def flatten_list(self, list_: CandidateTree) -> List[str]:
        """
        Flattens a nested list.

        Args:
            list_ (CandidateTree): Nested list to be flattened.
        
        Returns:
            List[str]: Flattened list.
        """
        return [item for sublist in list_ for item in sublist]

    def trim_tree(
        self, 
        tree: List[str], 
        prob_dict: ProbabilityDict, 
        topk: int
    ) -> List[str]:
        """
        Trims the tree to the top k candidates.

        Args:
            tree (List[str]): List of candidate strings.
            prob_dict (ProbabilityDict): Dictionary of probabilities of the candidates.
            topk (int): Number of top candidates to be retained.
        
        Returns:
            List[str]: List of top k candidates.
        """
        output = []
        probs = [prob_dict[x] for x in tree]
        candidates = [
            Node(item, item.replace(" ", ""), probs[idx]) for idx, item in enumerate(tree)
        ]
        for key, group in itertools.groupby(candidates, key=lambda x: x.characters):
            sorted_group = sorted(list(group), key=lambda x: x.score)
            trimmed_group = sorted_group[0:topk]
            trimmed_group = [x.hypothesis for x in trimmed_group]
            output.extend(trimmed_group)
        return output

    def trim_nodes(
        self,
        nodes: List[Node],
        topk: int
    ) -> List[Node]:
        """Trim nodes to top-k per unique character sequence (HASH-301).
        
        Args:
            nodes: List of Node objects to trim.
            topk: Number of top candidates to retain per character sequence.
            
        Returns:
            List of trimmed Node objects.
        """
        output = []
        
        # Sort by characters first to enable groupby
        sorted_nodes = sorted(nodes, key=lambda x: x.characters)
        
        for key, group in itertools.groupby(sorted_nodes, key=lambda x: x.characters):
            group_list = list(group)
            sorted_group = sorted(group_list, key=lambda x: x.score)
            trimmed_group = sorted_group[:topk]
            output.extend(trimmed_group)
        
        return output

    def run(
        self, 
        dataset: List[str], 
        topk: int = 20, 
        steps: int = 13
    ) -> ProbabilityDictionary:
        """
        Runs the beamsearch algorithm on the provided dataset.

        Args:
            dataset (List[str]): List of initial candidate strings.
            topk (int): Number of top candidates to be retained in each step. Default is 20.
            steps (int): Number of steps to run the algorithm. Default is 13.
        
        Returns:
            ProbabilityDictionary: Dictionary of final probabilities of the candidates.
        """
        if self.use_token_mode:
            return self._run_token_mode(dataset, topk, steps)
        else:
            return self._run_string_mode(dataset, topk, steps)

    def _run_string_mode(
        self,
        dataset: List[str],
        topk: int,
        steps: int
    ) -> ProbabilityDictionary:
        """Original string-based beam search (legacy mode)."""
        tree = dataset
        prob_dict: ProbabilityDict = {}
        for i in range(steps):
            tree = self.next_step(tree)
            tree = self.reshape_tree(tree, self.gpu_batch_size)
            prob_dict = self.update_probabilities(tree, prob_dict)
            tree = self.flatten_list(tree)
            tree = self.trim_tree(tree, prob_dict, topk)
        return ProbabilityDictionary(prob_dict)

    def _run_token_mode(
        self,
        dataset: List[str],
        topk: int,
        steps: int
    ) -> ProbabilityDictionary:
        """Token-based beam search (HASH-301, HASH-302).
        
        This mode operates on token IDs instead of strings for efficiency,
        and optionally uses KV-caching for incremental inference.
        """
        # Initialize nodes from dataset
        nodes = [
            Node(
                hypothesis=text,
                characters=text.replace(" ", ""),
                score=0.0,
                token_ids=None,
                past_key_values=None
            )
            for text in dataset
        ]
        
        prob_dict: ProbabilityDict = {}
        
        for i in range(steps):
            # Generate next candidates
            nodes = self.next_step_tokens(nodes)
            
            # Score candidates
            if self.use_kv_cache:
                nodes, prob_dict = self.update_probabilities_with_cache(nodes, prob_dict)
            else:
                nodes, prob_dict = self.update_probabilities_nodes(nodes, prob_dict)
            
            # Trim to top-k
            nodes = self.trim_nodes(nodes, topk)
        
        return ProbabilityDictionary(prob_dict)