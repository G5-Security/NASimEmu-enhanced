import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool
from config import config
from net import Net
from nasimemu.nasim.envs.host_vector import HostVector

import torch, numpy as np
import torch_geometric

from numba import jit

from torch.nn import *
from torch_geometric.data import Data, Batch
from torch_scatter import scatter

from rl import a2c, ppo

from graph_nns import *
from .net_utils import *

import wandb
def wandb_log_safe(data: dict, **kwargs):
    # Only log if W&B is initialized
    run = getattr(wandb, "run", None)
    if run is not None:
        try:
            wandb.log(data, **kwargs)
        except Exception:
            pass


class SimpleHRL(Net):
    def __init__(self):
        super().__init__()
        observation_dim = config.node_dim + config.pos_enc_dim
        
        # Manager Network - sets high-level goals
        self.manager_gat = GATConv(observation_dim, 64)
        self.manager_goal = nn.Linear(64, 32)  # Goal representation
        
        # Worker Network - executes low-level actions
        self.worker_gat = GATConv(observation_dim + 32, 128)  # State + goal
        self.worker_policy = nn.Linear(128, config.action_dim)
        
        # Value Network - estimates state value
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr)
        self.to(config.device)
        
        # Store current goal
        self.current_goal = None
        self.force_continue = False

    @staticmethod
    def prepare_batch(s_batch):
        node_feats, edge_index, node_index, pos_index = zip(*s_batch)
        node_feats = [torch.tensor(x, dtype=torch.float32, device=config.device) for x in node_feats]
        edge_index = [torch.tensor(x, dtype=torch.int64, device=config.device) for x in edge_index]
        data = [Data(x=node_feats[i], edge_index=edge_index[i]) for i in range(len(s_batch))]
        data_lens = [x.num_nodes for x in data]
        batch = Batch.from_data_list(data)
        batch_ind = batch.batch.to(config.device)
        node_index = np.concatenate(node_index)
        pos_index = torch.tensor(np.concatenate(pos_index)).to(config.device)
        return data, data_lens, batch, batch_ind, node_index, pos_index

    def reset_state(self, batch_mask=None):
        # Reset the current goal
        self.current_goal = None

    def forward(self, s_batch, only_v=False, force_action=None, reset_hidden=False):
        data, data_lens, batch, batch_ind, node_index, pos_index = self.prepare_batch(s_batch)
        x = batch.x
        
        # Positional encoding
        pos_enc = positional_encoding(pos_index, config.pos_enc_dim)
        x = torch.cat([x, pos_enc], dim=1)
        
        # Manager processes the graph and sets a goal
        manager_features = self.manager_gat(x, batch.edge_index)
        manager_pooled = global_mean_pool(manager_features, batch_ind)
        
        # Create goal (or use existing one if not time to update)
        if self.current_goal is None or reset_hidden:
            self.current_goal = self.manager_goal(manager_pooled)
        
        # Expand goal to match node count
        goal_per_node = self.current_goal[batch_ind]
        
        # Worker combines state and goal
        worker_input = torch.cat([x, goal_per_node], dim=1)
        worker_features = self.worker_gat(worker_input, batch.edge_index)
        
        # Value estimation
        value = self.value_net(manager_pooled)
        
        if only_v:
            return value
        
        # Action selection
        action_logits = self.worker_policy(worker_features)
        
        # Mask subnet nodes
        subnet_mask = batch.x[:, 0] == 1
        action_logits[subnet_mask] = -float('inf')
        
        # Sampling
        action_probs = torch_geometric.utils.softmax(
            action_logits.flatten(), 
            torch.repeat_interleave(batch_ind, config.action_dim))
        data_lens_a = [n_nodes * config.action_dim for n_nodes in data_lens]
        
        if force_action is not None:
            action_selected = force_action
        else:
            action_selected = segmented_sample(action_probs, data_lens_a)
        
        cum_lens = np.cumsum([0] + data_lens_a[:-1])
        a_index = torch.tensor(cum_lens, device=config.device) + action_selected
        a_prob = action_probs[a_index].view(-1, 1)
        
        # Map to environment actions
        n_index = a_index.cpu().numpy() // config.action_dim
        targets = node_index[n_index].reshape(-1, 2)
        a_id = (action_selected % config.action_dim).cpu().numpy()
        
        # Handle termination
        if not self.force_continue:
            terminate = (value.detach() <= 0).flatten()
            a_id[terminate.cpu().numpy()] = -1
            a_prob[terminate] = 0.5
        
        wandb_log_safe({
            "hidden_state_mean": memory_out.mean().item(),
            "action_logits_mean": action_logits.mean().item(),
            "value_estimate": value.mean().item(),
            "subnet_nodes": subnet_mask.sum().item() / len(subnet_mask)
        }, commit=False)

        return list(zip(targets, a_id)), value, a_prob, action_selected

    def set_force_continue(self, force):
        self.force_continue = force

    def update(self, trace, target_net=None, hidden_s0=None):
        """PPO update method"""
        sx, a, a_cnt, r, sx_, d = zip(*trace)
        
        # Prepare states and next states
        s = np.empty((config.ppo_t, config.batch), dtype=object)
        s[:, :] = sx
        s_ = np.empty((config.ppo_t, config.batch), dtype=object)
        s_[:, :] = sx_
        
        # Convert rewards and dones to arrays
        r = np.vstack(r)
        d = np.vstack(d)
        
        # Concatenate actions
        a = torch.cat(a)
        
        # Use current network as target if none provided
        if target_net is None:
            target_net = self
        
        # Compute value for next states
        v_ = target_net(s_[-1], only_v=True)
        v_ = v_.detach().flatten()
        
        # Compute value target
        v_target = compute_v_target(r, v_, d, config.gamma, config.ppo_t, 
                                   config.batch, config.use_a_t)
        
        # Flatten states and value targets
        s = np.concatenate(s)
        v_target = v_target.flatten()
        a_cnt = torch.tensor(np.concatenate(a_cnt), dtype=torch.bool, device=self.device)
        
        # Log metrics
        wandb.log({
            "v_value": v_.mean().item(),
            "v_target": v_target.mean().item()
        }, commit=False)
        
        # Perform PPO update
        return ppo(
            s, a, a_cnt, d, v_target, self, config.gamma,
            config.alpha_v, config.alpha_h, config.ppo_k, config.ppo_eps,
            config.use_a_t, config.v_range, False, None, True
        )

    def _update(self, loss):
        """Perform a single gradient update step"""
        self.opt.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), config.opt_max_norm
        )
        
        # Update weights
        self.opt.step()
        
        # Log gradient norm
        wandb_log_safe({"grad_norm": grad_norm.item()}, commit=False)
        return grad_norm