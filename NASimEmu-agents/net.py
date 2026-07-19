import os
import warnings

import torch, numpy as np
from torch.nn import *
from config import config

class Net(Module):
    # Subclasses that bind a semantic ontology to part of their architecture
    # (e.g. NASimNetDHRL's goal_bank <-> llm_teacher.goal_ontology.GOAL_NAMES)
    # override this so a checkpoint records which ontology it was trained
    # against -- master plan Sec 15.13: an old latent-goal checkpoint must
    # never be silently loaded into a semantic-goal model with a mismatched
    # meaning. None here means "this architecture has no ontology."
    ONTOLOGY_VERSION = None

    def __init__(self):
        super().__init__()

        self.device = torch.device(config.device)
        self.lr = config.opt_lr
        self.alpha_h = config.alpha_h

    def save(self, file='model.pt', training_state=None):
        """Save model weights and optional trainer state.

        ``training_state`` deliberately lives outside the module state dict:
        old checkpoints remain loadable, while ``main.py`` can attach the
        optimizer, target-network, RNG, schedule, and progress metadata needed
        for a step-aware restart.
        """
        payload = {
            'state_dict': self.state_dict(),
            'ontology_version': self.ONTOLOGY_VERSION,
        }
        if training_state is not None:
            payload['training_state'] = training_state
        # A power loss during torch.save must not destroy the previous usable
        # checkpoint. Write beside it and atomically replace only after the
        # complete payload has reached the filesystem interface.
        tmp_file = f'{file}.tmp'
        torch.save(payload, tmp_file)
        os.replace(tmp_file, file)

    def load(self, file='model.pt'):
        loaded = torch.load(file, map_location=self.device)

        is_versioned = isinstance(loaded, dict) and 'state_dict' in loaded and 'ontology_version' in loaded
        if not is_versioned:
            # Pre-existing checkpoint saved before ontology versioning was
            # added (e.g. trained_models/*.pt) -- a bare state_dict, not the
            # {'state_dict', 'ontology_version'} wrapper. Not a general
            # backwards-compat shim: this is the one-time exception needed so
            # prior trained work isn't stranded, not a promise to keep
            # supporting the old format going forward.
            warnings.warn(
                f"{file} is an unversioned checkpoint (predates ontology versioning) -- "
                f"loading it as-is. If this is a NASimNetDHRL checkpoint trained against "
                f"a different goal ontology than llm_teacher.goal_ontology's current version, "
                f"this load cannot detect that mismatch."
            )
            self.load_state_dict(loaded)
            return None

        ckpt_version = loaded['ontology_version']
        if ckpt_version is not None and self.ONTOLOGY_VERSION is not None and ckpt_version != self.ONTOLOGY_VERSION:
            raise ValueError(
                f"{file} was saved with ontology_version={ckpt_version}, but this model's "
                f"current ONTOLOGY_VERSION is {self.ONTOLOGY_VERSION}. Loading it would silently "
                f"bind goal_bank slots to the wrong semantic meaning -- refusing to load."
            )
        self.load_state_dict(loaded['state_dict'])
        return loaded.get('training_state')

    def copy_weights(self, other, rho):
        params_other = list(other.parameters())
        params_self  = list(self.parameters())

        for i in range( len(params_other) ):
            val_self  = params_self[i].data
            val_other = params_other[i].data
            val_new   = rho * val_other + (1-rho) * val_self

            params_self[i].data.copy_(val_new)

    def set_lr(self, lr):
        self.lr = lr

        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def set_alpha_h(self, alpha_h):
        self.alpha_h = alpha_h

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def reset_state(self, batch_mask=None):
        pass

    def clone_state(self, other):
        pass
