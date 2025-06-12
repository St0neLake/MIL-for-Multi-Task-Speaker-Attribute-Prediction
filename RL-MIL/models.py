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


class MeanMLP(BaseMLP):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(MeanMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
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
            output_dim,
            dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(MaxMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
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
            output_dim,
            dropout_p: float = 0.5,
            is_linear_attention: bool = True,
            attention_size: int = 64,
            attention_dropout_p: float = 0.5,
            autoencoder_layer_sizes=None,
    ):
        super(AttentionMLP, self).__init__(
            input_dim,
            hidden_dim,
            output_dim,
            dropout_p,
            autoencoder_layer_sizes=autoencoder_layer_sizes,
        )
        self.attention = None
        self.is_linear_attention = is_linear_attention
        self.attention_size = attention_size
        self.attention_dropout_p = attention_dropout_p

        self.init_attention()
        self.initialize_weights()

    def init_attention(self):
        if self.is_linear_attention:
            self.attention = nn.Linear(self.input_dim, 1)
        else:
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.attention_size),
                torch.nn.Dropout(p=self.attention_dropout_p),
                torch.nn.Tanh(),
                torch.nn.Linear(self.attention_size, 1),
            )

    def aggregate(self, x):
        attention = self.attention(x)
        attention = F.softmax(attention, dim=1)
        return torch.sum(x * attention, dim=1)


class ApproxRepSet(nn.Module):
    def __init__(
            self,
            input_dim,
            n_hidden_sets,
            n_elements,
            n_classes,
            autoencoder_layer_sizes=None,
    ):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements

        self.autoencoder_layer_sizes = autoencoder_layer_sizes
        self.base_network = BaseNetwork(self.autoencoder_layer_sizes)

        self.Wc = nn.Parameter(torch.FloatTensor(input_dim, n_hidden_sets * n_elements))

        self.fc1 = nn.Linear(n_hidden_sets, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.Wc.data)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)

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
        out = self.fc2(t)  # t: (batch_size, n_classes)
        return out


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
    if args.baseline == "MaxMLP":
        model = MaxMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.number_of_classes,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "MeanMLP":
        model = MeanMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.number_of_classes,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "AttentionMLP":
        model = AttentionMLP(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.number_of_classes,
            dropout_p=args.dropout_p,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    elif args.baseline == "repset":
        model = ApproxRepSet(
            input_dim=args.input_dim,
            n_hidden_sets=args.n_hidden_sets,
            n_elements=args.n_elements,
            n_classes=args.number_of_classes,
            autoencoder_layer_sizes=args.autoencoder_layer_sizes,
        )
    else:
        model = None
    return model


def create_mil_model_with_dict(args):
    if args['baseline'] == "MaxMLP":
        model = MaxMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "MeanMLP":
        model = MeanMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "AttentionMLP":
        model = AttentionMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "repset":
        model = ApproxRepSet(
            input_dim=args["input_dim"],
            n_hidden_sets=args["n_hidden_sets"],
            n_elements=args["n_elements"],
            n_classes=args["number_of_classes"],
            autoencoder_layer_sizes=args["autoencoder_layer_sizes"],
        )
    elif args['baseline'] == "SimpleMLP":
        model = MaxMLP(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            output_dim=args["number_of_classes"],
            dropout_p=args["dropout_p"],
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
        self.task_type = kwargs['task_type']
        self.min_clip = kwargs['min_clip']
        self.max_clip = kwargs['max_clip']
        self.sample_algorithm = kwargs.get('sample_algorithm', 'with_replacement')
        self.no_autoencoder = kwargs.get('no_autoencoder', False)
        
        try:
            self.task_optim = optim.AdamW(self.task_model.parameters(), lr=self.learning_rate)
        except:
            self.task_optim = None
        self.loss_fn = get_loss_fn(self.task_type)

        self.saved_actions = []
        self.rewards = []

    def forward(self, batch_x):
        if self.no_autoencoder:
            batch_rep = batch_x
        else:
            batch_rep = self.task_model.base_network(batch_x).detach()
        
        # batch_size, bag_size, embedding_size = batch_rep.shape
        # batch_rep = batch_rep.view(batch_size * bag_size, embedding_size)

        exp_reward = self.critic(batch_rep)
        action_probs = self.actor(batch_rep)
        action_probs = action_probs.squeeze(-1)
        # action_probs = action_probs.view(batch_size, bag_size)
        # exp_reward = exp_reward.view(batch_size, bag_size)
        # batch_rep = batch_rep.view(batch_size, bag_size, embedding_size)

        # action_probs = action_probs[:, :, 1]
        # action_probs = F.softmax(action_probs, dim=-1)
        # from IPython import embed; embed(); exit()

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
            data_ys, pred_ys, losses, prob_ys = [], [], [], []
            for batch_x, batch_y, _, _ in eval_data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_out, loss = self.eval_minibatch(batch_x, batch_y)
                if self.task_type == 'regression':
                    prob_y = pred_out
                    pred_y = torch.clamp(pred_out, min=self.min_clip, max=self.max_clip)
                elif self.task_type == 'classification':
                    prob_y = torch.softmax(pred_out, dim=1)
                    pred_y = torch.argmax(pred_out, dim=1)
                    
                pred_ys.append(pred_y.detach().cpu())
                prob_ys.append(prob_y.detach().cpu())
                data_ys.append(batch_y.detach().cpu())
                losses.append(loss)
            pred_Y = torch.cat(pred_ys, dim=0)
            data_Y = torch.cat(data_ys, dim=0)
            prob_Y = torch.cat(prob_ys, dim=0)
            if self.task_type == 'classification':
                reward = f1_score(data_Y.data, pred_Y.data, average='macro')
            elif self.task_type == 'regression':   
                reward = r2_score(data_Y.data, pred_Y.data)
        return reward, np.mean(losses), prob_Y, data_Y

    def compute_metrics_and_details(self, eval_data):
        with torch.no_grad():
            data_ys, pred_ys, losses, prob_ys = [], [], [], []
            for batch_x, batch_y, _, _ in eval_data:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_out, loss = self.eval_minibatch(batch_x, batch_y)
                if self.task_type == 'regression':
                    prob_y = pred_out
                    pred_y = torch.clamp(pred_out, min=self.min_clip, max=self.max_clip)
                elif self.task_type == 'classification':
                    prob_y = torch.softmax(pred_out, dim=1)
                    pred_y = torch.argmax(pred_out, dim=1)
                    
                pred_ys.append(pred_y.detach().cpu())
                prob_ys.append(prob_y.detach().cpu())
                data_ys.append(batch_y.detach().cpu())
                losses.append(loss)
            pred_Y = torch.cat(pred_ys, dim=0)
            data_Y = torch.cat(data_ys, dim=0)
            prob_Y = torch.cat(prob_ys, dim=0)
            metrics = {'loss': np.mean(losses)}
            if self.task_type == 'classification':
                f1_macro = f1_score(data_Y.data, pred_Y.data, average='macro')
                f1_micro = f1_score(data_Y.data, pred_Y.data, average='micro')
                if prob_Y.shape[1] == 2:
                    auc = roc_auc_score(data_Y.data, prob_Y.data[:, 1], average='macro')
                else:
                    auc = roc_auc_score(data_Y.data, prob_Y.data, average='macro', multi_class='ovr')
                metrics.update({
                    'f1': f1_macro,
                    'f1_micro': f1_micro,
                    'auc': auc,
                })
            elif self.task_type == 'regression':   
                reward = r2_score(data_Y.data, pred_Y.data)
                metrics.update({
                    'r2': reward,
                })
        return metrics, prob_Y.tolist(), data_Y.tolist(), pred_Y.tolist()
    
    def train_minibatch(self, batch_x, batch_y):
        self.task_model.train()
        batch_out = self.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        self.task_optim.zero_grad()
        batch_loss.backward()
        self.task_optim.step()
        return batch_loss.item()

    def eval_minibatch(self, batch_x, batch_y):
        self.task_model.eval()
        batch_out = self.task_model(batch_x)
        batch_loss = self.loss_fn(batch_out.squeeze(), batch_y.squeeze())
        return batch_out, batch_loss.item()

    def create_pool_data(self, dataloader, bag_size, pool_size, random=False):
        pool = []
        for _ in range(pool_size):
            data = self.select_from_dataloader(dataloader, bag_size, random=random)
            pool.append(data)
        return pool

    def expected_reward_loss(self, pool_data, average='macro', verbos=False):
        reward_pool, loss_pool, preds_pool = [], [], []
        for data in pool_data:
            reward, loss, preds, labels = self.compute_reward(data)
            reward_pool.append(reward)
            loss_pool.append(loss)
            preds_pool.append(preds)
        if self.task_type == 'classification':
            preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2).argmax(dim=1)
            ensemble_reward = f1_score(labels.data, preds_pool.data, average=average)
        elif self.task_type == 'regression':
            preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2).squeeze()
            preds_pool = torch.clamp(preds_pool, min=self.min_clip, max=self.max_clip)
            ensemble_reward = r2_score(labels.data, preds_pool.data)
        mean_reward = np.mean(reward_pool)
        mean_loss = np.mean(loss_pool)
        return mean_reward, mean_loss, ensemble_reward
    
    def predict_pool(self, pool_data):
        probs_pool = []
        for data in pool_data:
            prob_Y = self.predict(data)
            probs_pool.append(prob_Y)
        preds_pool = torch.stack(probs_pool, dim=2).mean(dim=2).argmax(dim=1)
        return preds_pool

    def predict(self, data):
        with torch.no_grad():
            prob_ys = []
            for batch_x in data:
                batch_x = batch_x.to(self.device)
                pred_out = self.task_model(batch_x)
                prob_y = torch.softmax(pred_out, dim=1)
                prob_ys.append(prob_y.detach().cpu())
            prob_Y = torch.cat(prob_ys, dim=0)
        return prob_Y
    
    def ensemble_predict(self, pool_data):
        preds_pool = []
        for data in pool_data:
            _, _, preds, labels = self.compute_reward(data)
            preds_pool.append(preds)
        preds_pool = torch.stack(preds_pool, dim=2).mean(dim=2)
        return preds_pool, labels
