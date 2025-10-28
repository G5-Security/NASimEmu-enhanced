"""Curriculum learning manager for progressive difficulty in training.

This module implements a curriculum learning system that gradually increases
the realism and difficulty of the environment during training, while ensuring
evaluation always uses the most difficult settings.
"""

import copy


class CurriculumManager:
    """Manages difficulty progression during training via curriculum stages.
    
    The curriculum manager controls the realism parameters (IDS, scan noise,
    network reliability, service dynamics) based on the current training epoch.
    During evaluation mode, it always returns the most difficult stage settings.
    
    Attributes
    ----------
    config : dict
        The curriculum configuration from the scenario
    training_mode : bool
        Whether the environment is in training or evaluation mode
    current_epoch : int
        The current epoch number during training
    """
    
    def __init__(self, curriculum_config, training_mode=True):
        """Initialize the curriculum manager.
        
        Parameters
        ----------
        curriculum_config : dict
            Dictionary containing curriculum stages and settings
        training_mode : bool, optional
            Whether in training mode (True) or evaluation mode (False).
            In evaluation mode, always uses the final/hardest stage.
            (default=True)
        """
        self.config = curriculum_config
        self.training_mode = training_mode
        self.current_epoch = 0
        
        # Validate and sort stages by start_epoch
        if 'stages' in self.config:
            self.config['stages'] = sorted(
                self.config['stages'], 
                key=lambda x: x.get('start_epoch', 0)
            )
    
    def update_epoch(self, epoch):
        """Update the current epoch number.
        
        This should be called at the start of each epoch during training.
        
        Parameters
        ----------
        epoch : int
            The current epoch number
        """
        self.current_epoch = epoch
    
    def get_current_stage(self):
        """Determine the current curriculum stage based on epoch number.
        
        In evaluation mode, always returns the final (most difficult) stage.
        In training mode, returns the stage corresponding to current epoch.
        
        Returns
        -------
        dict
            The stage configuration dictionary
        """
        if not self.training_mode:
            return self._get_final_stage()
        
        # Find the appropriate stage for current epoch
        if 'stages' not in self.config or len(self.config['stages']) == 0:
            return self._get_default_stage()
        
        for stage in self.config['stages']:
            start = stage.get('start_epoch', 0)
            end = stage.get('end_epoch', float('inf'))
            
            if start <= self.current_epoch < end:
                return stage
        
        # If no stage matches, return the last stage
        return self.config['stages'][-1]
    
    def _get_final_stage(self):
        """Get the final (most difficult) stage for evaluation.
        
        Returns
        -------
        dict
            The final stage configuration
        """
        if 'stages' not in self.config or len(self.config['stages']) == 0:
            return self._get_default_stage()
        
        return self.config['stages'][-1]
    
    def _get_default_stage(self):
        """Get a default stage with no realism enabled.
        
        Returns
        -------
        dict
            Default stage configuration
        """
        return {
            'name': 'default',
            'start_epoch': 0,
            'end_epoch': float('inf'),
            'ids': {'enabled': False},
            'scan_noise': {
                'service_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
                'os_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
                'process_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0}
            },
            'network_reliability': {
                'timeout_probability': 0.0,
                'affected_actions': []
            },
            'service_dynamics': {
                'churn_probability': 0.0,
                'affected_services': []
            }
        }
    
    def get_realism_params(self):
        """Get the current realism parameters based on the active stage.
        
        Returns
        -------
        dict
            Dictionary containing:
            - ids_config: IDS configuration
            - scan_noise: Scan noise configuration
            - network_reliability: Network reliability configuration
            - service_dynamics: Service dynamics configuration
        """
        stage = self.get_current_stage()
        
        return {
            'ids_config': stage.get('ids', {'enabled': False}),
            'scan_noise': stage.get('scan_noise', {}),
            'network_reliability': stage.get('network_reliability', {}),
            'service_dynamics': stage.get('service_dynamics', {})
        }
    
    def get_stage_info(self):
        """Get information about the current stage.
        
        Returns
        -------
        dict
            Dictionary with stage information:
            - name: Stage name
            - epoch: Current epoch
            - start_epoch: Stage start epoch
            - end_epoch: Stage end epoch
            - training_mode: Whether in training mode
        """
        stage = self.get_current_stage()
        
        return {
            'name': stage.get('name', 'unknown'),
            'epoch': self.current_epoch,
            'start_epoch': stage.get('start_epoch', 0),
            'end_epoch': stage.get('end_epoch', float('inf')),
            'training_mode': self.training_mode
        }
    
    def is_enabled(self):
        """Check if curriculum learning is enabled.
        
        Returns
        -------
        bool
            True if curriculum is enabled in config
        """
        return self.config.get('enabled', False)
    
    def get_stage_transition_epochs(self):
        """Get list of epoch numbers where stage transitions occur.
        
        Returns
        -------
        list of int
            Sorted list of epoch numbers where stages transition
        """
        if 'stages' not in self.config or len(self.config['stages']) == 0:
            return []
        
        transitions = set()
        for stage in self.config['stages']:
            start = stage.get('start_epoch', 0)
            transitions.add(start)
        
        return sorted(transitions)

