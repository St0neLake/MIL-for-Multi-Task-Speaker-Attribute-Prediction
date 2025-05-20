from abc import ABC, abstractmethod
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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


class SimpleMLP(nn.Module, ABC):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout_p: float = 0.5,
    ):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

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
        x = self.mlp(x)  # Apply the MLP
        return x

class BaseMLP(nn.Module, ABC):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dims_dict: dict[str, int],
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(BaseMLP, self).__init__()
        self.input_dim_for_heads = input_dim # This is the dimension of the aggregated features
        self.hidden_dim = hidden_dim
        self.output_dims_dict = output_dims_dict
        self.dropout_p = dropout_p  # register the droupout probability as a buffer

        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in self.output_dims_dict.items():
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(self.input_dim_for_heads, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(self.hidden_dim, num_classes),
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_network(x)
        aggregated_x = self.aggregate(x)  # Aggregate the data

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(aggregated_x)
        return outputs

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


class MeanMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dims_dict: dict[str, int],
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        dim_after_autoencoder = autoencoder_layer_sizes[-1] if autoencoder_layer_sizes else input_dim
        super(MeanMLP, self).__init__(
            dim_after_autoencoder,
            hidden_dim,
            output_dims_dict,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )

    def aggregate(self, x):
        return torch.mean(x, dim=1)  # Compute the mean along the bag_size dimension


class MaxMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dims_dict: dict[str, int],
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        dim_after_autoencoder = autoencoder_layer_sizes[-1] if autoencoder_layer_sizes else input_dim
        super(MaxMLP, self).__init__(
            dim_after_autoencoder,
            hidden_dim,
            output_dims_dict,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )

    def aggregate(self, x):
        return torch.max(
            x, dim=1
        ).values  # Compute the max along the bag_size dimension


class AttentionMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dims_dict: dict[str, int],
            dropout_p: float = 0.5,
            is_linear_attention: bool = True,
            attention_size: int = 64,
            attention_dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        dim_after_autoencoder = autoencoder_layer_sizes[-1] if autoencoder_layer_sizes else input_dim
        self.attention_input_dim = dim_after_autoencoder # Dimension for the attention mechanism input
        super(AttentionMLP, self).__init__(
            dim_after_autoencoder,
            hidden_dim,
            output_dims_dict,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )
        self.attention = None
        self.is_linear_attention = is_linear_attention
        self.attention_size = attention_size
        self.attention_dropout_p = attention_dropout_p

        self.init_attention()
        # self.initialize_weights()

    def init_attention(self):
        if self.is_linear_attention:
            self.attention_mechanism = nn.Linear(self.attention_input_dim, 1)
        else:
            self.attention_mechanism = torch.nn.Sequential(
                torch.nn.Linear(self.attention_input_dim, self.attention_size), # MODIFIED
                torch.nn.Dropout(p=self.attention_dropout_p),
                torch.nn.Tanh(),
                torch.nn.Linear(self.attention_size, 1),
            )

        # Ensure these new layers are also initialized
        for m in self.attention_mechanism.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def aggregate(self, x):
        attention_weights = self.attention_mechanism(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        return torch.sum(x * attention_weights, dim=1)


class ApproxRepSet(nn.Module):
    def __init__(
            self,
            input_dim,
            n_hidden_sets,
            n_elements,
            output_dims_dict: dict[str, int],
            autoencoder_layer_sizes=None,
    ):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        dim_after_autoencoder = autoencoder_layer_sizes[-1] if autoencoder_layer_sizes else input_dim
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        # Wc operates on the output of the autoencoder
        self.Wc = nn.Parameter(torch.FloatTensor(dim_after_autoencoder, n_hidden_sets * n_elements))

        self.fc1 = nn.Linear(n_hidden_sets, 32) # This layer's output (32) is input to task heads
        self.relu = nn.ReLU()

        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in output_dims_dict.items():
            self.task_heads[task_name] = nn.Linear(32, num_classes) # Input is 32 from self.fc1

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wc.data)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias.data)

        for task_name in self.task_heads: # Initialize new heads
            nn.init.xavier_uniform_(self.task_heads[task_name].weight.data)
            if self.task_heads[task_name].bias is not None:
                 nn.init.zeros_(self.task_heads[task_name].bias.data)

    def forward(self, x):  # x: (batch_size, bag_size, d)
        t = self.base_network(x)  # t: (batch_size, bag_size, d)
        t = self.relu(
            torch.matmul(t, self.Wc)
        )  # t: (batch_size, bag_size, n_hidden_sets * n_elements)
        t = t.view(
            t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets
        )  # t: (batch_size, bag_size, n_elements, n_hidden_sets)
        t, _ = torch.max(t, dim=2)  # t: (batch_size, bag_size, n_hidden_sets)
        t = torch.sum(t, dim=1)  # t: (batch_size, n_hidden_sets)
        t = self.relu(self.fc1(t))  # t: (batch_size, 32)
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(t)
        return outputs


class StratifiedRandomBaseline:
    """
    class_counts: dict of class counts having the labels as keys and the counts as values.
    It is compatible with the value_counts() method of pandas.
    """

    def __init__(self, class_counts):
        self.class_labels = list(class_counts.keys())
        counts = list(class_counts.values())
        total_count = sum(counts)
        self.probs = [count / total_count for count in counts]

    def __call__(self, size):
        choices = np.random.choice(self.class_labels, size=size, p=self.probs)
        return choices


class MajorityBaseline:
    """
    class_counts: dict of class counts having the labels as keys and the counts as values.
    It is compatible with the value_counts() method of pandas.
    """

    def __init__(self, class_counts):
        self.class_labels = list(class_counts.keys())
        counts = list(class_counts.values())
        self.majority_class = self.class_labels[np.argmax(counts)]

    def __call__(self, size):
        choices = np.full(size, self.majority_class)
        return choices

def random_model(train_dataframe, test_dataframe, args, logger):
    # Get the class counts
    class_counts = train_dataframe["labels"].value_counts().to_dict()
    # Initialize the random baseline
    random_baseline = StratifiedRandomBaseline(class_counts)
    # Get the predictions
    predictions = random_baseline(size=len(test_dataframe["labels"].tolist()))
    # Get the ground truth
    ground_truth = test_dataframe["labels"].values
    # Get the precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=ground_truth, y_pred=predictions, average="macro"
    )
    # Get the accuracy
    accuracy = (predictions == ground_truth).sum() / len(ground_truth)
    if not args.no_wandb:
        # Log the metrics
        wandb.init(
            tags=[
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
                f"DATASET_{args.dataset}",
                f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
                f"RANDOM_SEED_{args.random_seed}"
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.dataset}_{args.baseline}_{args.label}",
        )
        wandb.log(
            {
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
            }
        )
    logger.info(f"test/accuracy: {accuracy}")
    logger.info(f"test/precision: {precision}")
    logger.info(f"test/recall: {recall}")
    logger.info(f"test/f1: {f1}")

    # Confusion matrix
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    # Log the confusion matrix
    logger.info(f"Confusion matrix:\n{cm}")

def majority_model(train_dataframe, test_dataframe, args, logger):
    # Get the class counts
    class_counts = train_dataframe["labels"].value_counts().to_dict()
    # Initialize the majority baseline
    majority_baseline = MajorityBaseline(class_counts)
    # Get the predictions
    predictions = majority_baseline(size=len(test_dataframe["labels"]))
    # Get the ground truth
    ground_truth = test_dataframe["labels"].values
    # Get the precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=ground_truth, y_pred=predictions, average="macro"
    )
    # Get the accuracy
    accuracy = (predictions == ground_truth).sum() / len(ground_truth)
    if not args.no_wandb:
        # Log the metrics
        wandb.init(
            tags=[
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
                f"DATASET_{args.dataset}",
                f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
                f"RANDOM_SEED_{args.random_seed}"
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"{args.dataset}_{args.baseline}_{args.label}",
        )
        wandb.log(
            {
                "test/accuracy": accuracy,
                "test/precision": precision,
                "test/recall": recall,
                "test/f1": f1,
            }
        )
    logger.info(f"test/accuracy: {accuracy}")
    logger.info(f"test/precision: {precision}")
    logger.info(f"test/recall: {recall}")
    logger.info(f"test/f1: {f1}")

    # Confusion matrix
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    # Log the confusion matrix
    logger.info(f"Confusion matrix:\n{cm}")

def create_mil_model(args):
    mil_input_dim = args.input_dim

    if args.baseline == "MaxMLP":
        model = MaxMLP(
            input_dim=args.mil_input_dim ,
            hidden_dim=args.hidden_dim,
            output_dims_dict=args.output_dims_dict,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "MeanMLP":
        model = MeanMLP(
            input_dim=args.mil_input_dim ,
            hidden_dim=args.hidden_dim,
            output_dims_dict=args.output_dims_dict,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "AttentionMLP":
        model = AttentionMLP(
            input_dim=args.mil_input_dim ,
            hidden_dim=args.hidden_dim,
            output_dims_dict=args.output_dims_dict,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "repset":
        model = ApproxRepSet(
            input_dim=args.mil_input_dim ,
            n_hidden_sets=args.n_hidden_sets,
            n_elements=args.n_elements,
            output_dims_dict=args.output_dims_dict,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    else:
        model = None
    return model


def create_mil_model_with_dict(config: dict):
    mil_input_dim = config["input_dim"]
    if config['baseline'] == "MaxMLP":
        model = MaxMLP(
            input_dim=mil_input_dim,
            hidden_dim=config["hidden_dim"],
            output_dims_dict=config["output_dims_dict"],
            dropout_p=config["dropout_p"],
            autoencoder_layer_sizes=config.get("autoencoder_layer_sizes"),
        )
    elif config['baseline'] == "MeanMLP":
        model = MeanMLP(
            input_dim=mil_input_dim,
            hidden_dim=config["hidden_dim"],
            output_dims_dict=config["output_dims_dict"],
            dropout_p=config["dropout_p"],
            autoencoder_layer_sizes=config.get("autoencoder_layer_sizes"),
        )
    elif config['baseline'] == "AttentionMLP":
        model = AttentionMLP(
            input_dim=mil_input_dim,
            hidden_dim=config["hidden_dim"],
            output_dims_dict=config["output_dims_dict"],
            dropout_p=config["dropout_p"],
            autoencoder_layer_sizes=config.get("autoencoder_layer_sizes"),
            is_linear_attention=config.get("is_linear_attention", True),
            attention_size=config.get("attention_size", 64)
        )
    elif config['baseline'] == "repset":
        model = ApproxRepSet(
            input_dim=mil_input_dim,
            n_hidden_sets=config["n_hidden_sets"],
            n_elements=config["n_elements"],
            output_dims_dict=config["output_dims_dict"],
            autoencoder_layer_sizes=config.get("autoencoder_layer_sizes"),
        )
    elif config['baseline'] == "SimpleMLP":
        model = SimpleMLP(
            input_dim=mil_input_dim,
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dims_dict"],
            dropout_p=config["dropout_p"],
        )
    else:
        model = None
    return model


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ActorNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(ActorNetwork, self).__init__()
        # self.args = args
        self.state_dim = kwargs['state_dim']
        # self.actor = nn.Linear(self.state_dim, 2)
        self.actor  = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            # nn.Linear(32, 2),
            nn.Linear(32, 1),
        )
        self.actor.apply(init_weights)
        # nn.init.xavier_uniform_(self.actor.weight)

    def forward(self, x):
        # action_probs = F.softmax(self.actor(x), dim=-1)
        action_probs = F.sigmoid(self.actor(x))
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(CriticNetwork, self).__init__()
        # self.args = args
        self.state_dim = kwargs['state_dim']
        self.hdim = kwargs['hdim']
        self.ff1 = nn.Linear(self.state_dim, self.hdim)
        nn.init.xavier_uniform_(self.ff1.weight)

        self.critic_layer = nn.Linear(self.hdim, 1)
        nn.init.xavier_uniform_(self.critic_layer.weight)

        self.nl = nn.Tanh()

    def forward(self, x):
        c_in = self.nl(self.ff1(x))
        out = torch.sigmoid(self.critic_layer(c_in))
        # out = torch.mean(out)
        return out


def get_loss_fn(task_type):
    if task_type == 'classification':
        return nn.CrossEntropyLoss()
    elif task_type == 'regression':
        return nn.MSELoss()
    else:
        NotImplementedError

def sample_action(action_probs, n, device, random=False, algorithm="with_replacement"):
    if algorithm == "static":
        # print("static")
        return sample_static_action(action_probs, n, device, random=random)
    elif algorithm == "with_replacement":
        # print("with_replacement")
        return sample_action_with_replacement(action_probs, n, device, random=random)
    elif algorithm == "without_replacement":
        # print("without_replacement")
        return sample_action_without_replacement(action_probs, n, device, random=random)
    else:
        NotImplementedError

def sample_action_with_replacement(action_probs, n, device, random=False):
    # with replacement
    m = Categorical(action_probs)
    if random:
        action = torch.randint(0, action_probs.shape[1], (n, action_probs.shape[0])).to(device)
    else:
        action = m.sample((n,))

    log_prob = m.log_prob(action).sum(dim=0)
    # from IPython import embed; embed(); exit()
    return action.T, log_prob

def sample_action_without_replacement(action_probs, n, device, random=False):
    # multinomial sampling without replacement
    # sample_weights = action_probs[:, :, 1]
    sample_weights = action_probs
    if random:
        action = torch.empty((action_probs.shape[0], n), dtype=torch.long)
        for i in range(action_probs.shape[0]):
            action[i] = torch.randperm(action_probs.shape[1])[:n]
        action = action.to(device)
    else:
        action = torch.multinomial(sample_weights, n)
    log_prob = torch.log(sample_weights.gather(1, action))
    log_prob = log_prob.mean(dim=1)
    return action, log_prob


def sample_static_action(action_probs, n, device, random=False):
    # action_sort = action_probs[:, :, 1].sort(descending=True)
    if random:
        action = torch.empty((action_probs.shape[0], n), dtype=torch.long)
        for i in range(action_probs.shape[0]):
            action[i] = torch.randperm(action_probs.shape[1])[:n]
        action = action.to(device)
        log_prob = torch.gather(action_probs, 1, action)
    else:
        action_sort = action_probs.sort(descending=True)
        action = action_sort.indices[:, :n]
        log_prob = torch.log(action_sort.values[:, :n])
    log_prob = torch.mean(log_prob, dim=1)
    return action, log_prob


def select_from_action(action, x):
    return x[torch.arange(action.shape[0]).unsqueeze(1), action]


class PolicyNetwork(nn.Module):
    def __init__(self, **kwargs):
        super(PolicyNetwork, self).__init__()
        # self.args = args
        self.actor = ActorNetwork(state_dim=kwargs['state_dim'])
        self.critic = CriticNetwork(state_dim=kwargs['state_dim'], hdim=kwargs['hdim'])
        self.task_model = kwargs['task_model']
        self.learning_rate = kwargs['learning_rate']
        self.device = kwargs['device']
        self.task_type_global = kwargs['task_type']
        self.min_clip = kwargs['min_clip']
        self.max_clip = kwargs['max_clip']
        self.sample_algorithm = kwargs.get('sample_algorithm', 'with_replacement')
        self.no_autoencoder = kwargs.get('no_autoencoder', False)

        # Loss functions per task
        self.task_names = list(self.task_model.task_heads.keys())
        self.loss_fns = nn.ModuleDict({
            task_name: get_loss_fn(self.task_type_global) # Assuming task_type_global applies to all
            for task_name in self.task_names
        })

        try:
            self.task_optim = optim.AdamW(self.task_model.parameters(), lr=self.learning_rate)
        except:
            self.task_optim = None

        self.saved_actions = []
        self.rewards = []

    def forward(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()

        exp_reward = self.critic(batch_rep)
        action_probs = self.actor(batch_rep)
        action_probs = action_probs.squeeze(-1)
        exp_reward = torch.mean(exp_reward, dim=1)
        return action_probs, batch_rep, exp_reward

    def reset_reward_action(self):
        self.saved_actions, self.rewards = [], []

    # TODO: make it vectorize
    def normalize_rewards(self, eps=1e-5):
        R_mean = np.mean(self.rewards)
        R_std = np.std(self.rewards)
        for i, r in enumerate(self.rewards):
            self.rewards[i] = float((r - R_mean) / (R_std + eps))

    def select_from_dataloader(self, dataloader, bag_size, random=False):
        with torch.no_grad():
            data = []
            for batch_x, batch_y, indices, instance_labels in dataloader:
                batch_x = batch_x.to(self.device)
                # select batch_x
                action_probs, _, _ = self.forward(batch_x)
                action, _ = sample_action(action_probs, bag_size, self.device, random=random, algorithm=self.sample_algorithm)
                batch_x = select_from_action(action, batch_x)
                batch_x = batch_x.cpu()
                data.append((batch_x, batch_y, indices, instance_labels))
        return data

    def compute_reward(self, eval_data):
        with torch.no_grad():
            task_data_ys_all = {task: [] for task in self.task_names}
            task_pred_ys_all = {task: [] for task in self.task_names}
            # task_prob_ys_all = {task: [] for task in self.task_names} # If needed for other metrics like AUC

            all_batch_combined_loss_items = []

            for batch_x, batch_y_dict, _, _ in eval_data:
                batch_x = batch_x.to(self.device)
                batch_y_dict_device = {k: v.to(self.device) for k, v in batch_y_dict.items()}

                pred_out_dict, combined_loss_item, individual_batch_losses = self.eval_minibatch(batch_x, batch_y_dict_device)
                all_batch_combined_loss_items.append(combined_loss_item)

                for task_name in self.task_names:
                    pred_out_task = pred_out_dict[task_name]
                    batch_y_task_cpu = batch_y_dict[task_name]

                    pred_y_task = torch.argmax(pred_out_task, dim=1) # Assuming classification

                    task_pred_ys_all[task_name].append(pred_y_task.detach().cpu())
                    task_data_ys_all[task_name].append(batch_y_task_cpu)
                    # task_losses[task_name].append(individual_batch_losses[task_name]) # Accumulate individual losses if needed

            # Calculate F1 for each task and average for combined reward
            task_f1_scores = {}
            all_task_preds_final_for_reward = {}
            all_task_labels_final_for_reward = {}

            for task_name in self.task_names:
                pred_Y_task = torch.cat(task_pred_ys_all[task_name], dim=0)
                data_Y_task = torch.cat(task_data_ys_all[task_name], dim=0)

                all_task_preds_final_for_reward[task_name] = pred_Y_task
                all_task_labels_final_for_reward[task_name] = data_Y_task

                task_f1_scores[task_name] = f1_score(data_Y_task.data.numpy(), pred_Y_task.data.numpy(), average='macro', zero_division=0)

            combined_scalar_reward = np.mean(list(task_f1_scores.values()))
            mean_overall_combined_loss = np.mean(all_batch_combined_loss_items)

        # Return:
        # 1. combined_scalar_reward (for RL agent)
        # 2. mean_overall_combined_loss (for logging/evaluation)
        # 3. all_task_preds_final_for_reward (dict: task -> tensor of predictions for this eval_data pass)
        # 4. all_task_labels_final_for_reward (dict: task -> tensor of labels for this eval_data pass)
        return combined_scalar_reward, mean_overall_combined_loss, all_task_preds_final_for_reward, all_task_labels_final_for_reward


    def compute_metrics_and_details(self, eval_data):
        with torch.no_grad():
            task_data_ys_all = {task: [] for task in self.task_names}
            task_pred_ys_all = {task: [] for task in self.task_names}
            task_prob_ys_all = {task: [] for task in self.task_names}
            batch_losses_all = [] # Stores the combined loss from each batch

            for batch_x, batch_y_dict, _, _ in eval_data:
                batch_x = batch_x.to(self.device)
                batch_y_dict_device = {k: v.to(self.device) for k, v in batch_y_dict.items()}

                pred_out_dict, combined_loss_item, _ = self.eval_minibatch(batch_x, batch_y_dict_device)
                batch_losses_all.append(combined_loss_item)

                for task_name in self.task_names:
                    pred_out_task = pred_out_dict[task_name]

                    prob_y_task = torch.softmax(pred_out_task, dim=1)
                    pred_y_task = torch.argmax(pred_out_task, dim=1)

                    task_pred_ys_all[task_name].append(pred_y_task.detach().cpu())
                    task_prob_ys_all[task_name].append(prob_y_task.detach().cpu())
                    task_data_ys_all[task_name].append(batch_y_dict[task_name].cpu()) # Original CPU labels

            metrics_summary = {'loss': np.mean(batch_losses_all)} # Average of combined losses
            all_probs_final = {}
            all_labels_final = {}
            all_preds_final = {}

            for task_name in self.task_names:
                data_Y_task = torch.cat(task_data_ys_all[task_name], dim=0).numpy()
                pred_Y_task = torch.cat(task_pred_ys_all[task_name], dim=0).numpy()
                prob_Y_task = torch.cat(task_prob_ys_all[task_name], dim=0).numpy()

                all_labels_final[task_name] = data_Y_task.tolist()
                all_preds_final[task_name] = pred_Y_task.tolist()
                all_probs_final[task_name] = prob_Y_task.tolist()

                f1_macro = f1_score(data_Y_task, pred_Y_task, average='macro', zero_division=0)
                f1_micro = f1_score(data_Y_task, pred_Y_task, average='micro', zero_division=0)
                acc = np.mean(data_Y_task == pred_Y_task) # accuracy_score

                # AUC
                if prob_Y_task.shape[1] == 2: # Binary classification
                    auc_score = roc_auc_score(data_Y_task, prob_Y_task[:, 1], average='macro')
                else: # Multi-class
                    auc_score = roc_auc_score(data_Y_task, prob_Y_task, average='macro', multi_class='ovr')

                metrics_summary[f'{task_name}/f1'] = f1_macro
                metrics_summary[f'{task_name}/f1_micro'] = f1_micro
                metrics_summary[f'{task_name}/auc'] = auc_score
                metrics_summary[f'{task_name}/accuracy'] = acc

        # Return: metrics_summary (dict of metrics),
        # all_probs_final (dict task -> list of prob lists),
        # all_labels_final (dict task -> list of labels),
        # all_preds_final (dict task -> list of preds)
        return metrics_summary, all_probs_final, all_labels_final, all_preds_final

    def train_minibatch(self, batch_x, batch_y):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        total_loss = 0
        individual_losses = {}
        for task_name, preds_task in batch_out.items():
            loss = self.loss_fns[task_name](preds_task.squeeze(), batch_out[task_name].squeeze())
            total_loss += loss
            individual_losses[task_name] = loss.item()

        if self.task_optim: # Check if optimizer is defined
            self.task_optim.zero_grad()
            total_loss.backward()
            self.task_optim.step()
        return total_loss.item() # Return combined loss

    def eval_minibatch(self, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        total_loss = 0
        individual_losses = {}
        for task_name, preds_task in batch_out.items():
            loss = self.loss_fns[task_name](preds_task.squeeze(), batch_out[task_name].squeeze())
            total_loss += loss
            individual_losses[task_name] = loss.item()

        return batch_out, total_loss.item(), individual_losses # Return dict of outs, combined loss

    def create_pool_data(self, dataloader, bag_size, pool_size, random=False):
        pool = []
        for _ in range(pool_size):
            data = self.select_from_dataloader(dataloader, bag_size, random=random)
            pool.append(data)
        return pool

    def expected_reward_loss(self, pool_data, average='macro', verbos=False):
        reward_pool, loss_pool = [], []

        task_ensembled_preds = {task_name: [] for task_name in self.task_names}
        task_labels_for_ensemble = {task_name: None for task_name in self.task_names}

        first_call = True
        for data_idx, data_item in enumerate(pool_data): # pool_data is a list of (list of batches)
            reward, loss, preds_dict, labels_dict = self.compute_reward(data_item) # compute_reward now returns dicts
            reward_pool.append(reward)
            loss_pool.append(loss)

            for task_name in self.task_names:
                task_ensembled_preds[task_name].append(preds_dict[task_name])
                if first_call: # Assuming labels are the same across the pool for a given task
                    task_labels_for_ensemble[task_name] = labels_dict[task_name]
            first_call = False

        ensemble_f1_scores_per_task = {}
        for task_name in self.task_names:
            if task_labels_for_ensemble[task_name] is not None and task_ensembled_preds[task_name]:
                stacked_preds_for_task = torch.stack(task_ensembled_preds[task_name], dim=0) # Shape: (pool_size, num_samples)
                ensembled_final_preds_task, _ = torch.mode(stacked_preds_for_task, dim=0)

                labels_task = task_labels_for_ensemble[task_name]
                ensemble_f1_scores_per_task[task_name] = f1_score(labels_task.data.numpy(),
                                                                  ensembled_final_preds_task.data.numpy(),
                                                                  average=average, zero_division=0)
            else:
                ensemble_f1_scores_per_task[task_name] = 0.0 # Default if no preds/labels

        ensemble_reward = np.mean(list(ensemble_f1_scores_per_task.values()))

        mean_reward = np.mean(reward_pool)
        mean_loss = np.mean(loss_pool)
        return mean_reward, mean_loss, ensemble_reward

    def predict(self, data_batches):
        self.task_model.eval()
        task_prob_ys_all = {task: [] for task in self.task_names}
        with torch.no_grad():
            for batch_x, _, _, _ in data_batches: # y is not used for prediction here
                batch_x = batch_x.to(self.device)
                pred_out_dict = self.task_model(batch_x)
                for task_name in self.task_names:
                    prob_y_task = torch.softmax(pred_out_dict[task_name], dim=1)
                    task_prob_ys_all[task_name].append(prob_y_task.detach().cpu())

        concatenated_probs = {}
        for task_name in self.task_names:
            concatenated_probs[task_name] = torch.cat(task_prob_ys_all[task_name], dim=0)
        return concatenated_probs

    def predict_pool(self, pool_data):
        all_task_probs_from_pool = {task: [] for task in self.task_names}

        for data_item in pool_data:
            prob_dict_one_selection = self.predict(data_item) # Returns dict: task -> probs for this selection
            for task_name in self.task_names:
                all_task_probs_from_pool[task_name].append(prob_dict_one_selection[task_name])

        ensembled_final_preds = {}
        for task_name in self.task_names:
            # Stack probs for this task: (pool_size, num_samples, num_classes)
            stacked_probs_task = torch.stack(all_task_probs_from_pool[task_name], dim=0)
            mean_probs_task = torch.mean(stacked_probs_task, dim=0) #Shape: (num_samples, num_classes)
            ensembled_final_preds[task_name] = torch.argmax(mean_probs_task, dim=1)

        return ensembled_final_preds

    def ensemble_predict(self, pool_data):
        preds_pool = []
        for data in pool_data:
            _, _, preds, labels = self.compute_reward(data)
            preds_pool.append(preds)
        preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2)
        return preds_pool, labels
