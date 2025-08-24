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


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=1)
    
    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

# nasim_net_feudal.py
class GraphMemoryModule(nn.Module):
    def __init__(self, node_dim, memory_size, num_heads):
        super().__init__()
        self.memory_size = memory_size
        self.input_proj = nn.Linear(node_dim + memory_size, memory_size)
        self.attn = nn.MultiheadAttention(memory_size, num_heads)
        self.gru = nn.GRUCell(memory_size, memory_size)
        
    def forward(self, state, abstract_goal):
        # state: [num_nodes, node_dim]
        # abstract_goal: [num_nodes, memory_size]
        memory_input = torch.cat([state, abstract_goal], dim=1)
        projected_input = self.input_proj(memory_input)  # Project to memory_size
        
        # Reshape for MultiheadAttention: (seq_len, batch_size, embed_dim)
        # We treat each node as a separate sequence element
        projected_input = projected_input.unsqueeze(1)  # [num_nodes, 1, memory_size]
        projected_input = projected_input.transpose(0, 1)  # [1, num_nodes, memory_size]
        
        # Apply attention
        updated_memory, _ = self.attn(projected_input, projected_input, projected_input)
        
        # Reshape back: [num_nodes, memory_size]
        updated_memory = updated_memory.squeeze(0)
        new_memory = self.gru(updated_memory)
        return new_memory

class MetaSubgoalNetwork(nn.Module):
    def __init__(self, input_dim, num_subgoals):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_subgoals)
        )  # Fixed missing parenthesis
    
    def forward(self, context):
        return self.net(context)

class FeudalGTM(Net):
    def __init__(self):
        super().__init__()
        observation_dim = config.node_dim + config.pos_enc_dim
        num_subgoals = 5
        
        # Manager Network
        self.manager_gat = GraphAttentionLayer(observation_dim, 128)
        self.manager_lstm = nn.LSTM(128, 64, batch_first=True)
        
        # Graph Temporal Memory
        self.gtm = GraphMemoryModule(
            node_dim=observation_dim,
            memory_size=256,
            num_heads=4
        )
        
        # Worker Network
        self.worker = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, config.action_dim)
        )
        
        # Subgoal Module
        self.subgoal_predictor = MetaSubgoalNetwork(256, num_subgoals)
        
        # Value Function
        self.value_function = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        
        # Goal projection layer
        self.manager_goal_projection = nn.Linear(64, 256)
        
        self.opt = torch.optim.AdamW(self.parameters(), lr=config.opt_lr)
        self.to(config.device)
        
        # State tracking
        self.memory_state = None
        self.lstm_hidden = None
        self.lstm_cell = None
        self.force_continue = False  # Added initialization

    @staticmethod
    def prepare_batch(s_batch):
        # Identical to NASimNetGNN_MAct's implementation
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
        if batch_mask is None:
            self.memory_state = None
            self.lstm_hidden = None
            self.lstm_cell = None
        else:
            # Vectorized reset for finished episodes
            if not isinstance(batch_mask, torch.Tensor):
                batch_mask = torch.tensor(batch_mask, dtype=torch.bool, device=config.device)
            
            # Reset memory_state for finished episodes
            if self.memory_state is not None:
                reset_mask = batch_mask[self.batch_ind]  # Expand to node-level
                self.memory_state[reset_mask] = 0
            
            # Reset LSTM states for finished episodes
            if self.lstm_hidden is not None:
                self.lstm_hidden[:, batch_mask, :] = 0
            if self.lstm_cell is not None:
                self.lstm_cell[:, batch_mask, :] = 0

    def forward(self, s_batch, only_v=False, force_action=None, reset_hidden=False):
        data, data_lens, batch, batch_ind, node_index, pos_index = self.prepare_batch(s_batch)
        x = batch.x
        self.batch_ind = batch_ind  # Store for reset_state
        
        # Save current state if we're in replay mode
        if reset_hidden:
            saved_lstm_hidden = self.lstm_hidden
            saved_lstm_cell = self.lstm_cell
            saved_memory_state = self.memory_state
        
        # Positional encoding
        pos_enc = positional_encoding(pos_index, config.pos_enc_dim)
        x = torch.cat([x, pos_enc], dim=1)
        
        # Manager processing
        x_gat = self.manager_gat(x, batch.edge_index)
        x_pooled = global_mean_pool(x_gat, batch_ind)
        
        # Handle hidden state reset for replay
        if reset_hidden:
            self.lstm_hidden = None
            self.lstm_cell = None
            self.memory_state = None
        
        # Initialize LSTM state if needed
        if self.lstm_hidden is None:
            self.lstm_hidden = torch.zeros(1, len(data_lens), 64, device=config.device)
            self.lstm_cell = torch.zeros(1, len(data_lens), 64, device=config.device)
        
        # LSTM processing
        lstm_input = x_pooled.unsqueeze(1)
        lstm_out, (self.lstm_hidden, self.lstm_cell) = self.manager_lstm(
            lstm_input, 
            (self.lstm_hidden, self.lstm_cell)
        )
        abstract_goal = lstm_out.squeeze(1)
        
        # Project and expand manager goal
        projected_goal = self.manager_goal_projection(abstract_goal)
        goal_per_node = projected_goal[batch_ind]
        
        # Initialize memory if needed
        if self.memory_state is None:
            self.memory_state = torch.zeros(x.size(0), 256, device=config.device)
        
        # Graph Temporal Memory with per-node goal
        context = self.gtm(x, goal_per_node)
        self.memory_state = context  # Update memory state
        
        # Value function
        context_pooled = global_mean_pool(context, batch_ind)
        value = self.value_function(context_pooled)
        
        # Restore original state if we were in replay mode
        if reset_hidden:
            self.lstm_hidden = saved_lstm_hidden
            self.lstm_cell = saved_lstm_cell
            self.memory_state = saved_memory_state
        
        
        if only_v:
            return value
        
        # Action selection
        action_logits = self.worker(context)
        
        # Mask subnet nodes
        subnet_mask = batch.x[:, 0] == 1
        action_logits[subnet_mask] = -float('inf')
        
        # Sampling (identical to NASimNetGNN_MAct)
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
        
        return list(zip(targets, a_id)), value, a_prob, action_selected

    def set_force_continue(self, force):
        self.force_continue = force

    # Added update method for FeudalGTM
    
    def update(self, trace, target_net=None, hidden_s0=None):
        """PPO update method for FeudalGTM network"""
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
        
        # Perform PPO update with positional arguments
        return ppo(
            s,  # states
            a,  # actions
            a_cnt,  # advantages
            d,  # dones
            v_target,  # v_targets
            self,  # net
            config.gamma,
            config.alpha_v,
            config.alpha_h,
            config.ppo_k,
            config.ppo_eps,
            config.use_a_t,
            config.v_range,
            False,
            None,
            True
            #reset_hidden=True  # Add this flag for replay
        )

    # Added helper method for gradient updates
    def _update(self, loss):
        """Perform a single gradient update step for FeudalGTM"""
        self.opt.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), config.opt_max_norm
        )
        
        # Update weights
        self.opt.step()
        
        # Log gradient norm
        wandb.log({"grad_norm": grad_norm.item()}, commit=False)
        return grad_norm