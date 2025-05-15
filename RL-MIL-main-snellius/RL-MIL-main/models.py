from abc import ABC, abstractmethod
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Any, Optional # Added for type hinting

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score, r2_score, roc_auc_score

def build_layers(sizes):
    layers = []

    for in_size, out_size in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_size, out_size))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BaseNetwork(nn.Module):
    def __init__(self, layer_sizes=None):
        super(BaseNetwork, self).__init__()
        self.layer_sizes = layer_sizes
        if self.layer_sizes:
            self.net = build_layers(layer_sizes)
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.layer_sizes:
            # batch_size, bag_size, d = x.size()
            # x = x.view(batch_size * bag_size, d)
            x = self.net(x)
            # x = x.view(batch_size, bag_size, -1)
        return x


class MultiTaskBaseMIL(nn.Module, ABC):
    def __init__(
        self,
        input_dim_instance: int,
        task_configs: List[Dict[str, Any]],
        autoencoder_layer_sizes: Optional[List[int]] = None,
        default_head_hidden_dim: int = 128,
        default_head_dropout_p: float = 0.1
    ):
        super(MultiTaskBaseMIL, self).__init__()
        self.input_dim_instance = input_dim_instance
        self.task_configs = task_configs # Store for reference by PolicyNetwork
        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        if self.autoencoder_layer_sizes:
            aggregated_feature_dim = self.autoencoder_layer_sizes[-1]
        else:
            aggregated_feature_dim = input_dim_instance

        self.task_heads = nn.ModuleDict()
        for task_config in self.task_configs:
            task_name = task_config['name']
            output_dim_task = task_config['output_dim']
            head_hidden_dim = task_config.get('head_hidden_dim', default_head_hidden_dim)
            head_dropout_p = task_config.get('head_dropout_p', default_head_dropout_p)

            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(aggregated_feature_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=head_dropout_p),
                nn.Linear(head_hidden_dim, output_dim_task)
            )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for base_network and all task_heads
        for m in self.modules(): # self.modules() includes base_network and task_heads
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        instance_embeddings = self.base_network(x)
        aggregated_representation = self.aggregate(instance_embeddings)

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(aggregated_representation)
        return outputs

    def get_aggregated_data(self, x: torch.Tensor) -> torch.Tensor:
        instance_embeddings = self.base_network(x)
        aggregated_representation = self.aggregate(instance_embeddings)
        return aggregated_representation

    @abstractmethod
    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        pass

class BaseMLP(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_p: float = 0.5,
        autoencoder_layer_sizes=None,
    ):
        super(BaseMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_network(x)
        x = self.aggregate(x)  # Aggregate the data
        x = self.mlp(x)  # Apply the MLP
        return x

    def get_aggregated_data(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_network(x)
        x = self.aggregate(x)
        return x

    @abstractmethod
    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for data aggregation. This method should be implemented by any class that inherits from BaseMLP.

        :param x: input data
        :return: aggregated data
        """
        pass


class MultiTaskMeanMLP(MultiTaskBaseMIL):
    def __init__(
        self,
        input_dim_instance: int,
        task_configs: List[Dict[str, Any]],
        autoencoder_layer_sizes: Optional[List[int]] = None,
        default_head_hidden_dim: int = 128,
        default_head_dropout_p: float = 0.1
    ):
        super().__init__(
            input_dim_instance,
            task_configs,
            autoencoder_layer_sizes,
            default_head_hidden_dim,
            default_head_dropout_p
        )

    def aggregate(self, x):
        return torch.mean(x, dim=1)


class MultiTaskMaxMLP(MultiTaskBaseMIL):
    def __init__(
            self,
            input_dim_instance: int,
            task_configs: List[Dict[str, Any]],
            autoencoder_layer_sizes: Optional[List[int]] = None,
            default_head_hidden_dim: int = 128,
            default_head_dropout_p: float = 0.1
        ):
        super().__init__(
            input_dim_instance,
            task_configs,
            autoencoder_layer_sizes,
            default_head_hidden_dim,
            default_head_dropout_p
        )

    def aggregate(self, x):
        return torch.max(x, dim=1).values


class MultiTaskAttentionMLP(MultiTaskBaseMIL):
    def __init__(
            self,
            input_dim_instance: int,
            task_configs: List[Dict[str, Any]],
            autoencoder_layer_sizes: Optional[List[int]] = None,
            is_linear_attention: bool = True,
            attention_size: int = 64,
            attention_dropout_p: float = 0.1,
            default_head_hidden_dim: int = 128,
            default_head_dropout_p: float = 0.1
        ):
        self.is_linear_attention = is_linear_attention # Must be set before super init if init_attention uses it
        self.attention_size = attention_size
        self.attention_dropout_p = attention_dropout_p

        # Determine attention_input_dim before calling super
        attention_input_dim = autoencoder_layer_sizes[-1] if autoencoder_layer_sizes else input_dim_instance

        super().__init__(
            input_dim_instance,
            task_configs,
            autoencoder_layer_sizes,
            default_head_hidden_dim,
            default_head_dropout_p
        )

        self.init_attention(attention_input_dim)
        self.initialize_weights() # Re-initialize to catch attention_layer

    def init_attention(self, attention_input_dim: int):
        if self.is_linear_attention:
            self.attention_layer = nn.Linear(attention_input_dim, 1)
        else:
            self.attention_layer = nn.Sequential(
                nn.Linear(attention_input_dim, self.attention_size),
                nn.Dropout(p=self.attention_dropout_p),
                nn.Tanh(),
                nn.Linear(self.attention_size, 1),
            )

    def aggregate(self, x):
        attention_weights = self.attention_layer(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        return torch.sum(x * attention_weights, dim=1)


class MultiTaskApproxRepSet(nn.Module):
    def __init__(
            self,
            input_dim_instance: int,
            task_configs: List[Dict[str, Any]],
            n_hidden_sets: int,
            n_elements: int,
            autoencoder_layer_sizes: Optional[List[int]] = None,
            fc1_hidden_dim: int = 32,
            default_head_hidden_dim: int = 128,
            default_head_dropout_p: float = 0.1
        ):

        super().__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        self.task_configs = task_configs # Store for reference
        self.fc1_hidden_dim = fc1_hidden_dim
        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        repset_input_dim = self.autoencoder_layer_sizes[-1] if self.autoencoder_layer_sizes else input_dim_instance
        self.Wc = nn.Parameter(torch.FloatTensor(repset_input_dim, n_hidden_sets * n_elements))
        self.fc1 = nn.Linear(n_hidden_sets, self.fc1_hidden_dim)
        self.relu = nn.ReLU()

        self.task_heads = nn.ModuleDict()
        for task_config in self.task_configs:
            task_name = task_config['name']
            output_dim_task = task_config['output_dim']
            head_hidden_dim = task_config.get('head_hidden_dim', default_head_hidden_dim)
            head_dropout_p = task_config.get('head_dropout_p', default_head_dropout_p)
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(self.fc1_hidden_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=head_dropout_p),
                nn.Linear(head_hidden_dim, output_dim_task)
            )
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wc.data)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def aggregate_and_encode(self, x: torch.Tensor) -> torch.Tensor:
        t = self.base_network(x)
        t = self.relu(torch.matmul(t, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t, _ = torch.max(t, dim=2)
        t = torch.sum(t, dim=1)
        t = self.relu(self.fc1(t))
        return t

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        aggregated_encoded_representation = self.aggregate_and_encode(x)
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(aggregated_encoded_representation)
        return outputs


class SimpleMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout_p: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# class StratifiedRandomBaseline:
#     """
#     class_counts: dict of class counts having the labels as keys and the counts as values.
#     It is compatible with the value_counts() method of pandas.
#     """

#     def __init__(self, class_counts):
#         self.class_labels = list(class_counts.keys())
#         counts = list(class_counts.values())
#         total_count = sum(counts)
#         self.probs = [count / total_count for count in counts]

#     def __call__(self, size):
#         choices = np.random.choice(self.class_labels, size=size, p=self.probs)
#         return choices


# class MajorityBaseline:
#     """
#     class_counts: dict of class counts having the labels as keys and the counts as values.
#     It is compatible with the value_counts() method of pandas.
#     """

#     def __init__(self, class_counts):
#         self.class_labels = list(class_counts.keys())
#         counts = list(class_counts.values())
#         self.majority_class = self.class_labels[np.argmax(counts)]

#     def __call__(self, size):
#         choices = np.full(size, self.majority_class)
#         return choices

# def random_model(train_dataframe, test_dataframe, args, logger):
#     # Get the class counts
#     class_counts = train_dataframe["labels"].value_counts().to_dict()
#     # Initialize the random baseline
#     random_baseline = StratifiedRandomBaseline(class_counts)
#     # Get the predictions
#     predictions = random_baseline(size=len(test_dataframe["labels"].tolist()))
#     # Get the ground truth
#     ground_truth = test_dataframe["labels"].values
#     # Get the precision, recall, f1
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true=ground_truth, y_pred=predictions, average="macro"
#     )
#     # Get the accuracy
#     accuracy = (predictions == ground_truth).sum() / len(ground_truth)
#     if not args.no_wandb:
#         # Log the metrics
#         wandb.init(
#             tags=[
#                 f"BAG_SIZE_{args.bag_size}",
#                 f"BASELINE_{args.baseline}",
#                 f"LABEL_{args.label}",
#                 f"EMBEDDING_MODEL_{args.embedding_model}",
#                 f"DATASET_{args.dataset}",
#                 f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
#                 f"RANDOM_SEED_{args.random_seed}"
#                 f"EMBEDDING_MODEL_{args.embedding_model}",
#             ],
#             entity=args.wandb_entity,
#             project=args.wandb_project,
#             name=f"{args.dataset}_{args.baseline}_{args.label}",
#         )
#         wandb.log(
#             {
#                 "test/accuracy": accuracy,
#                 "test/precision": precision,
#                 "test/recall": recall,
#                 "test/f1": f1,
#             }
#         )
#     logger.info(f"test/accuracy: {accuracy}")
#     logger.info(f"test/precision: {precision}")
#     logger.info(f"test/recall: {recall}")
#     logger.info(f"test/f1: {f1}")

#     # Confusion matrix
#     cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
#     # Log the confusion matrix
#     logger.info(f"Confusion matrix:\n{cm}")

# def majority_model(train_dataframe, test_dataframe, args, logger):
#     # Get the class counts
#     class_counts = train_dataframe["labels"].value_counts().to_dict()
#     # Initialize the majority baseline
#     majority_baseline = MajorityBaseline(class_counts)
#     # Get the predictions
#     predictions = majority_baseline(size=len(test_dataframe["labels"]))
#     # Get the ground truth
#     ground_truth = test_dataframe["labels"].values
#     # Get the precision, recall, f1
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true=ground_truth, y_pred=predictions, average="macro"
#     )
#     # Get the accuracy
#     accuracy = (predictions == ground_truth).sum() / len(ground_truth)
#     if not args.no_wandb:
#         # Log the metrics
#         wandb.init(
#             tags=[
#                 f"BAG_SIZE_{args.bag_size}",
#                 f"BASELINE_{args.baseline}",
#                 f"LABEL_{args.label}",
#                 f"EMBEDDING_MODEL_{args.embedding_model}",
#                 f"DATASET_{args.dataset}",
#                 f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
#                 f"RANDOM_SEED_{args.random_seed}"
#                 f"EMBEDDING_MODEL_{args.embedding_model}",
#             ],
#             entity=args.wandb_entity,
#             project=args.wandb_project,
#             name=f"{args.dataset}_{args.baseline}_{args.label}",
#         )
#         wandb.log(
#             {
#                 "test/accuracy": accuracy,
#                 "test/precision": precision,
#                 "test/recall": recall,
#                 "test/f1": f1,
#             }
#         )
#     logger.info(f"test/accuracy: {accuracy}")
#     logger.info(f"test/precision: {precision}")
#     logger.info(f"test/recall: {recall}")
#     logger.info(f"test/f1: {f1}")

#     # Confusion matrix
#     cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
#     # Log the confusion matrix
#     logger.info(f"Confusion matrix:\n{cm}")

# def create_mil_model(args):
#     if args.baseline == "MaxMLP":
#         model = MaxMLP(
#             input_dim=args.input_dim,
#             hidden_dim=args.hidden_dim,
#             output_dim=args.number_of_classes,
#             dropout_p=args.dropout_p,
#             autoencoder_layer_sizes=args.autoencoder_layer_sizes,
#         )
#     elif args.baseline == "MeanMLP":
#         model = MeanMLP(
#             input_dim=args.input_dim,
#             hidden_dim=args.hidden_dim,
#             output_dim=args.number_of_classes,
#             dropout_p=args.dropout_p,
#             autoencoder_layer_sizes=args.autoencoder_layer_sizes,
#         )
#     elif args.baseline == "AttentionMLP":
#         model = AttentionMLP(
#             input_dim=args.input_dim,
#             hidden_dim=args.hidden_dim,
#             output_dim=args.number_of_classes,
#             dropout_p=args.dropout_p,
#             autoencoder_layer_sizes=args.autoencoder_layer_sizes,
#         )
#     elif args.baseline == "repset":
#         model = ApproxRepSet(
#             input_dim=args.input_dim,
#             n_hidden_sets=args.n_hidden_sets,
#             n_elements=args.n_elements,
#             n_classes=args.number_of_classes,
#             autoencoder_layer_sizes=args.autoencoder_layer_sizes,
#         )
#     else:
#         model = None
#     return model


# def create_mil_model_with_dict(args):
#     if args['baseline'] == "MaxMLP":
#         model = MaxMLP(
#             input_dim=args["input_dim"],
#             hidden_dim=args["hidden_dim"],
#             output_dim=args["number_of_classes"],
#             dropout_p=args["dropout_p"],
#             autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
#         )
#     elif args['baseline'] == "MeanMLP":
#         model = MeanMLP(
#             input_dim=args["input_dim"],
#             hidden_dim=args["hidden_dim"],
#             output_dim=args["number_of_classes"],
#             dropout_p=args["dropout_p"],
#             autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
#         )
#     elif args['baseline'] == "AttentionMLP":
#         model = AttentionMLP(
#             input_dim=args["input_dim"],
#             hidden_dim=args["hidden_dim"],
#             output_dim=args["number_of_classes"],
#             dropout_p=args["dropout_p"],
#             autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
#         )
#     elif args['baseline'] == "repset":
#         model = ApproxRepSet(
#             input_dim=args["input_dim"],
#             n_hidden_sets=args["n_hidden_sets"],
#             n_elements=args["n_elements"],
#             n_classes=args["number_of_classes"],
#             autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
#         )
#     elif args['baseline'] == "SimpleMLP":
#         model = MaxMLP(
#             input_dim=args["input_dim"],
#             hidden_dim=args["hidden_dim"],
#             output_dim=args["number_of_classes"],
#             dropout_p=args["dropout_p"],
#         )
#     else:
#         model = None
#     return model


def init_weights_policy(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class ActorNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(ActorNetwork, self).__init__()
        self.state_dim = kwargs['state_dim']
        self.actor  = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.actor.apply(init_weights_policy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.state_dim) # (batch_size * bag_size, state_dim)
        action_scores = self.actor(x) # (batch_size * bag_size, 1)

        if original_shape.dim() > 2:
            action_scores = action_scores.view(original_shape[0], original_shape[1]) # (batch_size, bag_size)
        else: # If input was (N, state_dim)
            action_scores = action_scores.squeeze(-1)

        action_probs = torch.sigmoid(action_scores) # Output scores between 0 and 1
        return action_probs

class CriticNetwork(nn.Module): # RL Critic for Value Function V(s)
    def __init__(self, **kwargs):
        super(CriticNetwork, self).__init__()
        self.state_dim = kwargs['state_dim']
        self.hdim = kwargs['hdim'] # Hidden dimension for the RL critic
        self.ff1 = nn.Linear(self.state_dim, self.hdim)
        self.critic_layer = nn.Linear(self.hdim, 1) # Outputs a single value (state value)
        self.nl = nn.Tanh() # Activation for the hidden layer

        self.ff1.apply(init_weights_policy)
        self.critic_layer.apply(init_weights_policy) # Output layer should also be initialized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.state_dim)

        c_in = self.nl(self.ff1(x))
        out = self.critic_layer(c_in) # No sigmoid for V(s) usually

        if original_shape.dim() > 2:
             out = out.view(original_shape[0], original_shape[1])
        else:
            out = out.squeeze(-1)
        return out


def get_loss_fn(task_type: str, task_name: str = "") -> nn.Module: # Added task_name for clarity
    if task_type == 'classification':
        return nn.CrossEntropyLoss()
    elif task_type == 'regression':
        return nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss for task '{task_name}' type '{task_type}' not implemented.")

def sample_action(action_probs: torch.Tensor, n_to_select: int, device: torch.device,
                  random_selection: bool = False, algorithm: str = "with_replacement") -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, bag_size = action_probs.shape

    if n_to_select <= 0 : # Handle n_to_select being non-positive
        return torch.empty((batch_size, 0), dtype=torch.long, device=device), torch.zeros(batch_size, device=device)

    if n_to_select > bag_size:
        # print(f"Warning: n_to_select ({n_to_select}) > bag_size ({bag_size}). Selecting all {bag_size} instances.")
        n_to_select = bag_size

    # Ensure action_probs are valid for Categorical and log
    clamped_action_probs = torch.clamp(action_probs, min=1e-8, max=1.0) # Clamp for stability

    if algorithm == "static":
        if random_selection:
            actions_list = [torch.randperm(bag_size, device=device)[:n_to_select] for _ in range(batch_size)]
            action = torch.stack(actions_list) if actions_list else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        else:
            if bag_size == 0: # Cannot sort empty tensor
                 action = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            else:
                _, action = torch.topk(clamped_action_probs, k=min(n_to_select, bag_size), dim=1)

        if action.numel() == 0: # No actions selected
            log_prob = torch.zeros(batch_size, device=device)
        else:
            log_prob = torch.log(clamped_action_probs.gather(1, action)).sum(dim=1)


    elif algorithm == "with_replacement":
        # Normalize probabilities for Categorical if they don't sum to 1 (though sigmoid output is not a distribution)
        # For selection scores, Categorical can still work if scores are positive.
        # If using raw scores, it's fine. If they are meant as probabilities, ensure they sum to 1 per instance (not per bag here).
        # Since action_probs are sigmoid outputs (0 to 1) per instance, directly using them in Categorical is okay.
        m = Categorical(logits=clamped_action_probs) # Using logits is safer if they are not normalized probabilities

        if random_selection:
            actions_list = [torch.randperm(bag_size, device=device)[:n_to_select] for _ in range(batch_size)]
            action = torch.stack(actions_list) if actions_list else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        else:
            if n_to_select == 0:
                action = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            else:
                action_samples = []
                for _ in range(n_to_select): # Sample one by one for "with_replacement" behavior
                    action_samples.append(m.sample())
                action = torch.stack(action_samples, dim=1) if action_samples else torch.empty((batch_size, 0), dtype=torch.long, device=device)

        if action.numel() == 0:
            log_prob = torch.zeros(batch_size, device=device)
        else:
             # log_prob for Categorical is log_prob of individual samples. Summing them implies independence.
            log_prob = m.log_prob(action).sum(dim=1)


    elif algorithm == "without_replacement":
        if random_selection:
            actions_list = [torch.randperm(bag_size, device=device)[:n_to_select] for _ in range(batch_size)]
            action = torch.stack(actions_list) if actions_list else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        else:
            if bag_size == 0 or n_to_select == 0:
                 action = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            else:
                # Ensure weights for multinomial are positive and sum to 1 if required by some backends.
                # For torch.multinomial, they just need to be non-negative.
                weights_for_sampling = clamped_action_probs
                # Handle rows that sum to zero after clamping (e.g., all original probs were zero)
                row_sums = weights_for_sampling.sum(dim=1, keepdim=True)
                # Add small epsilon to rows that sum to 0 to allow uniform sampling from them
                weights_for_sampling[row_sums.squeeze() == 0] = 1.0 / bag_size

                action = torch.multinomial(weights_for_sampling, n_to_select, replacement=False)

        if action.numel() == 0:
            log_prob = torch.zeros(batch_size, device=device)
        else:
            # log_prob for actions sampled without replacement is complex.
            # Approximating by sum of log_probs of chosen actions from original distribution.
            log_prob = torch.log(clamped_action_probs.gather(1, action)).sum(dim=1)
    else:
        raise NotImplementedError(f"Sampling algorithm '{algorithm}' not implemented.")
    return action, log_prob

def select_from_action(action: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch_size, n_selected = action.shape
    if x.dim() < 3:
        raise ValueError(f"Input tensor x must have at least 3 dimensions (batch, bag, features), but got {x.dim()}")
    if n_selected == 0 : # If no actions selected, return empty tensor with correct feature dim
        return torch.empty((batch_size, 0, x.shape[2]), device=x.device, dtype=x.dtype)
    if action.max() >= x.shape[1] or action.min() < 0:
        print(f"Warning: Action indices out of bounds. Action max: {action.max()}, x.shape[1]: {x.shape[1]}. Clamping.")
        action = torch.clamp(action, 0, x.shape[1] - 1)

    batch_idx = torch.arange(batch_size, device=action.device).unsqueeze(1).expand_as(action)
    selected_x = x[batch_idx, action]
    return selected_x


class PolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(PolicyNetwork, self).__init__()
        self.device = kwargs['device']

        # Ensure task_model is passed and is a MultiTaskBaseMIL instance (or similar structure)
        if not hasattr(kwargs['task_model'], 'task_configs'):
             raise ValueError("task_model passed to PolicyNetwork must have a 'task_configs' attribute.")
        self.task_model: MultiTaskBaseMIL = kwargs['task_model']
        self.task_configs = self.task_model.task_configs

        state_dim_for_policy = kwargs['state_dim']
        self.actor = ActorNetwork(state_dim=state_dim_for_policy)
        self.critic = CriticNetwork(state_dim=state_dim_for_policy, hdim=kwargs['hdim'])

        self.learning_rate_task_model = kwargs['learning_rate'] # LR for MIL model
        self.min_clip = kwargs.get('min_clip', None)
        self.max_clip = kwargs.get('max_clip', None)
        self.sample_algorithm = kwargs.get('sample_algorithm', 'with_replacement')
        self.no_autoencoder_for_rl = kwargs.get('no_autoencoder_for_rl', False) # For policy's view of state

        self.task_model_optimizer = optim.AdamW(self.task_model.parameters(), lr=self.learning_rate_task_model)

        self.task_loss_fns = nn.ModuleDict()
        for config in self.task_configs:
            self.task_loss_fns[config['name']] = get_loss_fn(config['type'], config['name'])

        self.saved_actions = []
        self.rewards = []

    def forward(self, batch_bag_instance_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.no_autoencoder_for_rl:
            instance_features_for_policy = batch_bag_instance_embeddings
        else:
            self.task_model.base_network.to(batch_bag_instance_embeddings.device)
            if self.task_model.base_network.layer_sizes:
                with torch.no_grad(): # Policy observes fixed features from g_theta
                    instance_features_for_policy = self.task_model.base_network(batch_bag_instance_embeddings).detach()
            else:
                instance_features_for_policy = batch_bag_instance_embeddings

        action_probs = self.actor(instance_features_for_policy)
        exp_reward_per_instance = self.critic(instance_features_for_policy)
        # Ensure exp_reward is calculated correctly even if bag_size is 1 or features_for_policy is 2D
        if exp_reward_per_instance.dim() > 1 and exp_reward_per_instance.shape[1] > 0 :
             exp_reward = torch.mean(exp_reward_per_instance, dim=1)
        elif exp_reward_per_instance.dim() == 1: # Only one instance per bag, or already averaged
            exp_reward = exp_reward_per_instance
        else: # Empty bag case for critic output
            exp_reward = torch.zeros(action_probs.shape[0], device=self.device)

        return action_probs, instance_features_for_policy, exp_reward

    def reset_reward_action(self):
        self.saved_actions, self.rewards = [], []

    def normalize_rewards(self, eps=1e-8): # Slightly larger eps for stability
        if not self.rewards: return
        rewards_np = np.array(self.rewards, dtype=np.float32)
        R_mean = np.mean(rewards_np)
        R_std = np.std(rewards_np)
        # Ensure R_std is not too small to avoid division by zero or large numbers
        if abs(R_std) < eps :
            self.rewards = [0.0] * len(rewards_np) # If std is essentially zero, normalized rewards are zero
        else:
            self.rewards = [(float(r) - R_mean) / R_std for r in rewards_np]

    def select_instances_from_bag_embeddings(self, batch_bag_instance_embeddings: torch.Tensor,
                                             bag_size_k_selected: int,
                                             random_selection: bool=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            # Get features for the policy network (actor)
            if self.no_autoencoder_for_rl:
                instance_features_for_policy = batch_bag_instance_embeddings
            else:
                self.task_model.base_network.to(batch_bag_instance_embeddings.device)
                if self.task_model.base_network.layer_sizes:
                     instance_features_for_policy = self.task_model.base_network(batch_bag_instance_embeddings)
                else:
                     instance_features_for_policy = batch_bag_instance_embeddings

            action_probs = self.actor(instance_features_for_policy) # Get probabilities/scores from actor

            actions_taken, action_log_probs = sample_action(
                action_probs, bag_size_k_selected, self.device,
                random=random_selection, algorithm=self.sample_algorithm
            )
            # Select the actual instance embeddings (raw ones, before base_network) for the task_model
            selected_instance_embeddings = select_from_action(actions_taken, batch_bag_instance_embeddings)
        return selected_instance_embeddings, actions_taken, action_log_probs

    def select_from_dataloader(self, dataloader, bag_size_k_selected: int, random: bool = False) -> List[tuple]:
        self.eval()
        selected_data_for_task_model = []
        with torch.no_grad():
            for batch_x_original_bag, batch_y_multi_labels_dict, _, instance_lvl_labels in dataloader:
                batch_x_original_bag = batch_x_original_bag.to(self.device)

                if batch_x_original_bag.shape[1] == 0: # Skip empty bags
                    continue

                if self.no_autoencoder_for_rl:
                    instance_features_for_policy = batch_x_original_bag
                else:
                    self.task_model.base_network.to(batch_x_original_bag.device)
                    if self.task_model.base_network.layer_sizes:
                        instance_features_for_policy = self.task_model.base_network(batch_x_original_bag)
                    else:
                        instance_features_for_policy = batch_x_original_bag

                action_probs = self.actor(instance_features_for_policy)
                actions, _ = sample_action(action_probs, bag_size_k_selected, self.device,
                                           random=random, algorithm=self.sample_algorithm)

                if actions.numel() == 0 and bag_size_k_selected > 0 : # If tried to select but got no actions
                    # This can happen if the bag was smaller than k_selected and sample_action handled it by returning empty
                    # Or if all bags were empty initially.
                    # Create empty tensors with correct feature dim for consistency if needed downstream,
                    # but usually, such batches might be skipped or handled carefully.
                    # For now, if actions are empty, selected_x will also be empty if created correctly.
                     batch_x_selected_for_task_model = torch.empty((batch_x_original_bag.shape[0], 0, batch_x_original_bag.shape[2]),
                                                                  device='cpu', dtype=batch_x_original_bag.dtype)

                else:
                    batch_x_selected_for_task_model = select_from_action(actions, batch_x_original_bag).cpu()

                selected_data_for_task_model.append((batch_x_selected_for_task_model,
                                                     batch_y_multi_labels_dict,
                                                     None, instance_lvl_labels))
        return selected_data_for_task_model

    def train_task_model_minibatch(self, batch_x_selected_instances: torch.Tensor,
                                   batch_y_multitask_labels: Dict[str, torch.Tensor]) -> tuple[float, Dict[str, float]]:
        if batch_x_selected_instances.shape[1] == 0: # No instances selected
            return 0.0, {task_config['name']: 0.0 for task_config in self.task_configs}

        self.task_model.train()
        self.task_model_optimizer.zero_grad()
        predictions_dict = self.task_model(batch_x_selected_instances) # MIL model forward pass

        total_loss = torch.tensor(0.0, device=self.device)
        task_losses_items = {}
        for task_config in self.task_configs:
            task_name = task_config['name']
            y_pred_task = predictions_dict[task_name]
            y_true_task = batch_y_multitask_labels[task_name].to(self.device)

            # Ensure y_true_task is not empty if labels are per-bag
            if y_true_task.numel() == 0:
                task_losses_items[task_name] = 0.0
                continue

            if task_config['type'] == "regression":
                y_pred_task = y_pred_task.squeeze(-1) if y_pred_task.dim() > 1 and y_pred_task.shape[-1] == 1 else y_pred_task
                y_true_task = y_true_task.squeeze().float()
            else: # classification
                y_true_task = y_true_task.squeeze().long()

            if y_pred_task.shape[0] != y_true_task.shape[0]: # Mismatch after squeeze, skip
                print(f"Warning: Shape mismatch for task {task_name} after squeeze. Pred: {y_pred_task.shape}, True: {y_true_task.shape}. Skipping loss.")
                task_losses_items[task_name] = 0.0
                continue


            loss = self.task_loss_fns[task_name](y_pred_task, y_true_task)
            task_losses_items[task_name] = loss.item()
            # Consider task weights here if you plan to use them: total_loss += task_weight * loss
            total_loss += loss

        if total_loss.requires_grad: # Only backward if there was a computable loss
            total_loss.backward()
            self.task_model_optimizer.step()
            return total_loss.item(), task_losses_items
        else: # No gradient, e.g. all true labels were empty or all selected bags were empty.
            return 0.0, task_losses_items

    def compute_reward_and_metrics(self, eval_data_selected_instances: List[tuple]
                                  ) -> tuple[float, float, Dict[str, Dict[str, Any]], Dict[str, Dict[str, torch.Tensor]]]: # Added type for detailed_preds_trues
        if not eval_data_selected_instances: # Handle empty eval data
            return 0.0, 0.0, {cfg['name']: {'loss': 0.0} for cfg in self.task_configs}, {}

        self.task_model.eval()
        all_task_outputs = {cfg['name']: {'y_preds_metric_input': [], 'y_trues_metric_input': [], 'losses': [], 'type': cfg['type']} for cfg in self.task_configs}

        for batch_x_sel, batch_y_multi, _, _ in eval_data_selected_instances:
            if batch_x_sel.shape[1] == 0: continue # Skip if no instances in selected batch
            batch_x_sel = batch_x_sel.to(self.device)

            with torch.no_grad():
                task_model_outputs_dict = self.task_model(batch_x_sel)

            for task_config in self.task_configs:
                task_name = task_config['name']
                if task_name not in batch_y_multi or batch_y_multi[task_name].numel() == 0:
                    all_task_outputs[task_name]['losses'].append(0.0) # Or handle as appropriate
                    continue

                y_pred_raw_cpu = task_model_outputs_dict[task_name].cpu()
                y_true_cpu = batch_y_multi[task_name].cpu().squeeze()

                all_task_outputs[task_name]['y_trues_metric_input'].append(y_true_cpu)

                y_pred_for_loss_device = task_model_outputs_dict[task_name]
                y_true_for_loss_device = batch_y_multi[task_name].to(self.device).squeeze()

                if task_config['type'] == "regression":
                    all_task_outputs[task_name]['y_preds_metric_input'].append(y_pred_raw_cpu.squeeze(-1) if y_pred_raw_cpu.dim() > 1 else y_pred_raw_cpu)
                    y_pred_for_loss_device = y_pred_for_loss_device.squeeze(-1) if y_pred_for_loss_device.dim() > 1 else y_pred_for_loss_device
                    y_true_for_loss_device = y_true_for_loss_device.float()
                else:
                    all_task_outputs[task_name]['y_preds_metric_input'].append(F.softmax(y_pred_raw_cpu, dim=1))
                    y_true_for_loss_device = y_true_for_loss_device.long()

                # Ensure shapes are compatible for loss
                if y_pred_for_loss_device.shape[0] == y_true_for_loss_device.shape[0] and y_true_for_loss_device.numel() > 0:
                    loss = self.task_loss_fns[task_name](y_pred_for_loss_device, y_true_for_loss_device)
                    all_task_outputs[task_name]['losses'].append(loss.item())
                else:
                    all_task_outputs[task_name]['losses'].append(0.0) # Or log a warning

        final_metrics_per_task = {}
        combined_reward_components = []
        overall_avg_loss_sum = 0
        num_tasks_with_loss = 0
        detailed_preds_trues_for_return = {}

        for task_name, data in all_task_outputs.items():
            if not data['y_trues_metric_input'] or not data['y_preds_metric_input']:
                final_metrics_per_task[task_name] = {'loss': 0, f"{'r2' if data['type'] == 'regression' else 'f1_macro'}": 0}
                if data['type'] == 'classification': final_metrics_per_task[task_name]['auc_macro'] = 0
                detailed_preds_trues_for_return[task_name] = {'y_pred_full': torch.empty(0), 'y_true_full': torch.empty(0)}
                continue

            y_trues_all = torch.cat(data['y_trues_metric_input'])
            y_preds_all_metric_input = torch.cat(data['y_preds_metric_input'])

            if y_trues_all.numel() == 0: # No true labels for this task in any batch
                final_metrics_per_task[task_name] = {'loss': np.mean(data['losses']) if data['losses'] else 0}
                detailed_preds_trues_for_return[task_name] = {'y_pred_full': y_preds_all_metric_input, 'y_true_full': y_trues_all}
                continue


            detailed_preds_trues_for_return[task_name] = {'y_pred_full': y_preds_all_metric_input, 'y_true_full': y_trues_all}
            avg_loss_task = np.mean(data['losses']) if data['losses'] else 0.0
            final_metrics_per_task[task_name] = {'loss': avg_loss_task}
            if data['losses']: # Only average if there were actual losses calculated
                overall_avg_loss_sum += avg_loss_task
                num_tasks_with_loss += 1

            metric_val = 0.0
            try:
                if data['type'] == 'regression':
                    # Ensure y_preds_all_metric_input is 1D for R2 score
                    y_p_squeezed = y_preds_all_metric_input.squeeze()
                    y_p_clamped = torch.clamp(y_p_squeezed, min=self.min_clip or -float('inf'), max=self.max_clip or float('inf'))
                    if y_trues_all.numel() > 1 : # R2 needs at least 2 samples
                         metric_val = r2_score(y_trues_all.numpy(), y_p_clamped.numpy())
                    final_metrics_per_task[task_name]['r2'] = metric_val
                elif data['type'] == 'classification':
                    # Ensure y_preds_all_metric_input has shape (N, num_classes)
                    if y_preds_all_metric_input.dim() == 1 : # If it was squeezed to 1D by mistake, unsqueeze
                        if y_preds_all_metric_input.numel() == y_trues_all.numel() * self.task_configs[[tc['name'] for tc in self.task_configs].index(task_name)]['output_dim']:
                             y_preds_all_metric_input = y_preds_all_metric_input.view(y_trues_all.numel(), -1)
                        else: # Cannot reshape, skip metric
                             metric_val = 0.0; auc_val = 0.0;
                             final_metrics_per_task[task_name].update({'f1_macro': metric_val, 'auc_macro': auc_val})
                             combined_reward_components.append(metric_val)
                             continue


                    y_pred_classes = torch.argmax(y_preds_all_metric_input, dim=1)
                    metric_val = f1_score(y_trues_all.numpy(), y_pred_classes.numpy(), average='macro', zero_division=0)
                    final_metrics_per_task[task_name]['f1_macro'] = metric_val

                    num_classes_task = y_preds_all_metric_input.shape[1]
                    if num_classes_task == 2:
                        auc_val = roc_auc_score(y_trues_all.numpy(), y_preds_all_metric_input[:, 1].numpy(), average='macro')
                    elif num_classes_task > 2 :
                        auc_val = roc_auc_score(y_trues_all.numpy(), y_preds_all_metric_input.numpy(), average='macro', multi_class='ovr')
                    else:
                        auc_val = 0.0 # Should not happen if output_dim > 1 for classification
                    final_metrics_per_task[task_name]['auc_macro'] = auc_val
            except Exception as e:
                print(f"Error calculating metric for task {task_name}: {e}")
                metric_val = 0.0 # Default metric if error occurs
                if data['type'] == 'regression': final_metrics_per_task[task_name]['r2'] = metric_val
                else: final_metrics_per_task[task_name].update({'f1_macro': metric_val, 'auc_macro': 0.0})


            combined_reward_components.append(metric_val)

        final_scalar_reward = np.mean(combined_reward_components) if combined_reward_components else 0.0
        avg_total_loss = overall_avg_loss_sum / num_tasks_with_loss if num_tasks_with_loss > 0 else 0.0

        return final_scalar_reward, avg_total_loss, final_metrics_per_task, detailed_preds_trues_for_return

    def expected_reward_loss(self, pool_data_selected_instances: List[List[tuple]],
                           average_metric_type='macro') -> tuple[float, float, float]:
        if not pool_data_selected_instances: # Handle empty pool
            return 0.0, 0.0, 0.0

        pool_rewards = [] # Store scalar reward for each item in the pool
        pool_losses = []  # Store average loss for each item in the pool

        # For ensemble reward calculation:
        # {task_name: {'y_true': FULL_DATASET_Y_TRUE_FOR_TASK,
        #             'y_pred_raw_outputs_from_each_pool_item': [[BATCH1_PREDS, BATCH2_PREDS,...]_pool_item1, ...],
        #             'type': TASK_TYPE}}
        ensemble_collector = {
            tc['name']: {'y_true': [],
                         'y_pred_raw_outputs_from_each_pool_item': [[] for _ in range(len(pool_data_selected_instances))],
                         'type': tc['type']}
            for tc in self.task_configs
        }
        # Flag to ensure y_true is concatenated only once from the first pool item
        all_trues_concatenated_for_task = {tc['name']: False for tc in self.task_configs}

        for pool_idx, single_pool_item_batches in enumerate(pool_data_selected_instances):
            if not single_pool_item_batches : continue # Skip empty pool item

            # --- Calculate reward & loss for this single pool item (for `mean_reward_across_pool`) ---
            # This is like calling compute_reward_and_metrics on one full pass of selected data
            item_scalar_reward, item_avg_loss, _, _ = self.compute_reward_and_metrics(single_pool_item_batches)
            pool_rewards.append(item_scalar_reward)
            pool_losses.append(item_avg_loss)

            # --- Collect raw predictions from this pool item for final ensemble calculation ---
            with torch.no_grad():
                for batch_x_sel, batch_y_multi_dict, _, _ in single_pool_item_batches:
                    if batch_x_sel.shape[1] == 0: continue
                    batch_x_sel = batch_x_sel.to(self.device)
                    task_model_outputs_dict = self.task_model(batch_x_sel) # Raw outputs

                    for task_config in self.task_configs:
                        task_name = task_config['name']
                        if task_name not in batch_y_multi_dict or batch_y_multi_dict[task_name].numel() == 0:
                            continue

                        y_pred_raw_batch_cpu = task_model_outputs_dict[task_name].cpu()
                        ensemble_collector[task_name]['y_pred_raw_outputs_from_each_pool_item'][pool_idx].append(y_pred_raw_batch_cpu)

                        if not all_trues_concatenated_for_task[task_name]:
                            y_true_batch_cpu = batch_y_multi_dict[task_name].cpu().squeeze()
                            ensemble_collector[task_name]['y_true'].append(y_true_batch_cpu)

            # After processing all batches of the first pool item, concatenate all its y_true
            if pool_idx == 0:
                for task_name in self.task_configs:
                    task_name = task_name['name'] # Get name from config dict
                    if ensemble_collector[task_name]['y_true']:
                        ensemble_collector[task_name]['y_true'] = torch.cat(ensemble_collector[task_name]['y_true'])
                    all_trues_concatenated_for_task[task_name] = True

        # --- Calculate final ensemble reward ---
        ensemble_task_metric_values = []
        for task_config in self.task_configs:
            task_name = task_config['name']
            task_type = task_config['type']

            if not ensemble_collector[task_name]['y_true'].numel(): continue

            y_true_full_dataset = ensemble_collector[task_name]['y_true']

            # Aggregate predictions for this task from all pool items
            # Each element in list is preds from one pool item (concatenated over its batches)
            preds_from_each_pool_item_for_this_task = []
            for i in range(len(pool_data_selected_instances)):
                if ensemble_collector[task_name]['y_pred_raw_outputs_from_each_pool_item'][i]:
                    preds_from_each_pool_item_for_this_task.append(
                        torch.cat(ensemble_collector[task_name]['y_pred_raw_outputs_from_each_pool_item'][i])
                    )

            if not preds_from_each_pool_item_for_this_task: continue

            # Stack predictions from different pool strategies: (pool_size, num_total_samples_in_dataset, num_classes_or_1)
            # Ensure all tensors in preds_from_each_pool_item_for_this_task have the same num_total_samples
            # This should be true if y_true_full_dataset corresponds to the full dataset length
            # and each pool item processed all of it.
            try:
                stacked_preds_raw = torch.stack(preds_from_each_pool_item_for_this_task)
            except RuntimeError as e:
                print(f"Error stacking predictions for task {task_name} in ensemble: {e}")
                print(f"Sizes: {[p.shape for p in preds_from_each_pool_item_for_this_task]}")
                continue


            # Average raw predictions (logits/values) across the pool: (num_total_samples, num_classes_or_1)
            avg_pooled_preds_raw = torch.mean(stacked_preds_raw, dim=0)

            ensemble_metric = 0.0
            try:
                if task_type == 'regression':
                    avg_pooled_preds_clamped = torch.clamp(avg_pooled_preds_raw.squeeze(),
                                                           min=self.min_clip or -float('inf'),
                                                           max=self.max_clip or float('inf'))
                    if y_true_full_dataset.numel() > 1:
                        ensemble_metric = r2_score(y_true_full_dataset.numpy(), avg_pooled_preds_clamped.numpy())
                elif task_type == 'classification':
                    avg_pooled_probs = F.softmax(avg_pooled_preds_raw, dim=1)
                    avg_pooled_pred_classes = torch.argmax(avg_pooled_probs, dim=1)
                    ensemble_metric = f1_score(y_true_full_dataset.numpy(), avg_pooled_pred_classes.numpy(),
                                               average=average_metric_type, zero_division=0)
            except Exception as e:
                print(f"Error calculating ensemble metric for task {task_name}: {e}")
                ensemble_metric = 0.0

            ensemble_task_metric_values.append(ensemble_metric)

        final_ensemble_reward = np.mean(ensemble_task_metric_values) if ensemble_task_metric_values else 0.0
        mean_reward_across_pool = np.mean(pool_rewards) if pool_rewards else 0.0
        mean_loss_across_pool = np.mean(pool_losses) if pool_losses else 0.0

        return mean_reward_across_pool, mean_loss_across_pool, final_ensemble_reward

# --- Utility functions for model creation (Refined for MultiTask) ---
def create_mil_model_with_dict(args_dict: Dict[str, Any]) -> nn.Module:
    class ArgsNamespace: # Simple way to access dict keys as attributes
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args_ns = ArgsNamespace(**args_dict)

    # --- Crucial: Ensure task_configs is present and correctly formatted ---
    if not hasattr(args_ns, 'task_configs') or not args_ns.task_configs:
        # Fallback: Attempt to create a single-task config from old args for backward compatibility
        # This part is tricky and should ideally be avoided by passing explicit task_configs
        print("Warning: 'task_configs' not found or empty in args_dict. Attempting single-task fallback.")
        output_dim = args_dict.get('number_of_classes', args_dict.get('output_dim', 1))
        # Infer task_type based on output_dim or an explicit old task_type arg
        inferred_task_type = args_dict.get('task_type', 'classification' if output_dim > 1 else 'regression')

        args_ns.task_configs = [{'name': args_dict.get('label', 'default_task'),
                                'type': inferred_task_type,
                                'output_dim': output_dim,
                                'head_hidden_dim': args_dict.get('hidden_dim', 128),
                                'head_dropout_p': args_dict.get('dropout_p', 0.1)}]

    # Ensure 'input_dim_instance' is present for MIL models
    # This is the dimension of raw instance embeddings fed to BaseNetwork
    if not hasattr(args_ns, 'input_dim_instance'):
        # Try to get it from 'input_dim' if that was used, or default
        args_ns.input_dim_instance = args_dict.get('input_dim', args_dict.get('input_dim_instance', 768))
        print(f"Warning: 'input_dim_instance' not in args_dict, using/defaulting to {args_ns.input_dim_instance}")

    # Defaults for other args if not present (used by create_mil_model)
    if not hasattr(args_ns, 'hidden_dim'): args_ns.hidden_dim = 128 # For default_head_hidden_dim
    if not hasattr(args_ns, 'dropout_p'): args_ns.dropout_p = 0.1   # For default_head_dropout_p
    if args_ns.baseline == "repset": # Specific defaults for RepSet
        if not hasattr(args_ns, 'n_hidden_sets'): args_ns.n_hidden_sets = 8
        if not hasattr(args_ns, 'n_elements'): args_ns.n_elements = 8
        if not hasattr(args_ns, 'fc1_hidden_dim'): args_ns.fc1_hidden_dim = 32
    if args_ns.baseline == "AttentionMLP":
        if not hasattr(args_ns, 'attention_size'): args_ns.attention_size = 64
        if not hasattr(args_ns, 'attention_dropout_p'): args_ns.attention_dropout_p = 0.1


    return create_mil_model(args_ns)

def create_mil_model(args) -> nn.Module: # args is an object (e.g., Namespace)
    if not hasattr(args, 'task_configs') or not args.task_configs:
        raise ValueError("'task_configs' must be provided in args for create_mil_model.")
    if not hasattr(args, 'input_dim_instance'):
        raise ValueError("'input_dim_instance' (raw instance embedding dimension) must be provided in args.")

    if args.baseline == "SimpleMLP":
        if len(args.task_configs) != 1:
            raise ValueError("SimpleMLP is single-task, expects 1 task_config.")
        tc = args.task_configs[0]
        # SimpleMLP expects aggregated features as input.
        # Here, args.input_dim_instance might represent aggregated dim if SimpleMLP is used.
        return SimpleMLP(input_dim=args.input_dim_instance,
                           hidden_dim=tc.get('head_hidden_dim', getattr(args, 'hidden_dim', 128)),
                           output_dim=tc['output_dim'],
                           dropout_p=tc.get('head_dropout_p', getattr(args, 'dropout_p', 0.5))) # SimpleMLP had 0.5 default

    mil_class_map = {
        "MeanMLP": MultiTaskMeanMLP, "MaxMLP": MultiTaskMaxMLP,
        "AttentionMLP": MultiTaskAttentionMLP, "repset": MultiTaskApproxRepSet
    }
    if args.baseline in mil_class_map:
        model_class = mil_class_map[args.baseline]
        common_mil_args = {
            "input_dim_instance": args.input_dim_instance, # Raw instance embedding dim
            "task_configs": args.task_configs,
            "autoencoder_layer_sizes": getattr(args, 'autoencoder_layer_sizes', None),
            "default_head_hidden_dim": getattr(args, 'hidden_dim', 128),
            "default_head_dropout_p": getattr(args, 'dropout_p', 0.1)
        }
        if args.baseline == "AttentionMLP":
            common_mil_args.update({
                "is_linear_attention": getattr(args, 'is_linear_attention', True),
                "attention_size": getattr(args, 'attention_size', 64),
                "attention_dropout_p": getattr(args, 'attention_dropout_p', 0.1)
            })
        elif args.baseline == "repset":
            common_mil_args.update({
                "n_hidden_sets": args.n_hidden_sets,
                "n_elements": args.n_elements,
                "fc1_hidden_dim": getattr(args, 'fc1_hidden_dim', 32)
            })
        return model_class(**common_mil_args)
    else:
        raise ValueError(f"Unknown or unsupported MIL baseline for MTL: {args.baseline}")

# --- Baselines (random, majority - these are for single-label classification) ---
# These would need to be adapted or called per-task for multi-task evaluation.
class StratifiedRandomBaseline:
    def __init__(self, class_counts: Dict[Any, int]):
        self.class_labels = list(class_counts.keys())
        counts = list(class_counts.values())
        total_count = sum(counts)
        self.probs = [count / total_count for count in counts] if total_count > 0 else []

    def __call__(self, size: int) -> np.ndarray:
        if not self.probs: return np.array([]) # Handle empty class_counts
        return np.random.choice(self.class_labels, size=size, p=self.probs)

class MajorityBaseline:
    def __init__(self, class_counts: Dict[Any, int]):
        self.class_labels = list(class_counts.keys())
        counts = list(class_counts.values())
        self.majority_class = self.class_labels[np.argmax(counts)] if counts else None

    def __call__(self, size: int) -> np.ndarray:
        if self.majority_class is None: return np.array([])
        return np.full(size, self.majority_class)