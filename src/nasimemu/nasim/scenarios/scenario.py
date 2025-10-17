import math, numpy as np
from pprint import pprint

import nasimemu.nasim.scenarios.utils as u


class Scenario:

    def __init__(self, scenario_dict, name=None, generated=False, permute_subnets=True):

        self.scenario_dict = scenario_dict
        self.name = name
        self.generated = generated
        self._e_map = None
        self._pe_map = None

        if permute_subnets:
            self._permute_subnets()

        # this is used for consistent positioning of
        # host state and obs in state and obs matrices
        self.host_num_map = {}
        for host_num, host_addr in enumerate(self.hosts):
            self.host_num_map[host_addr] = host_num

    def _permute_subnets(self):
        # create the permutation sequence
        perm = np.concatenate([[0], np.random.permutation(np.arange(1, len(self.subnets)))])
        orig = np.arange(len(self.subnets))
        # print(f'perm={list(perm)}')
        # print(f'orig={list(orig)}')

        # permute the subnets' sizes
        subnets = np.array(self.scenario_dict[u.SUBNETS])
        subnets[perm] = subnets[orig]
        self.scenario_dict[u.SUBNETS] = subnets

        # permute topology
        topology = np.array(self.scenario_dict[u.TOPOLOGY])

        topology[perm, :] = topology[orig, :] # rows
        topology[:, perm] = topology[:, orig] # cols

        self.scenario_dict[u.TOPOLOGY] = topology

        # permute sensitive_hosts
        sensitive_hosts = {}
        for ((subnet, host_addr), value) in self.scenario_dict[u.SENSITIVE_HOSTS].items():
            sensitive_hosts[(perm[subnet], host_addr)] = value

        self.scenario_dict[u.SENSITIVE_HOSTS] = sensitive_hosts

        # permute host_configs
        hosts = {}
        for ((subnet, host_addr), host_config) in self.scenario_dict[u.HOSTS].items():
            # alter the host's address
            host_config.address = (perm[subnet], host_addr)

            # alter the host firewall
            fw_dict = {}
            for ((fw_subnet, fw_host_addr), fw_config) in host_config.firewall.items():
                fw_dict[(perm[fw_subnet], fw_host_addr)] = fw_config
            host_config.firewall = fw_dict

            # add to the correct spot
            hosts[(perm[subnet], host_addr)] = host_config

        self.scenario_dict[u.HOSTS] = hosts

        # permute firewall
        firewall = {}
        for ((fw_from, fw_to), fw_config) in self.scenario_dict[u.FIREWALL].items():
            firewall[(perm[fw_from], perm[fw_to])] = fw_config

        self.scenario_dict[u.FIREWALL] = firewall

    @property
    def step_limit(self):
        return self.scenario_dict.get(u.STEP_LIMIT, None)

    @property
    def services(self):
        return self.scenario_dict[u.SERVICES]

    @property
    def num_services(self):
        return len(self.services)

    @property
    def os(self):
        return self.scenario_dict[u.OS]

    @property
    def num_os(self):
        return len(self.os)

    @property
    def processes(self):
        return self.scenario_dict[u.PROCESSES]

    @property
    def num_processes(self):
        return len(self.processes)

    @property
    def access_levels(self):
        return u.ROOT_ACCESS

    @property
    def exploits(self):
        return self.scenario_dict[u.EXPLOITS]

    @property
    def privescs(self):
        return self.scenario_dict[u.PRIVESCS]

    @property
    def exploit_map(self):
        """A nested dictionary for all exploits in scenario.

        I.e. {service_name: {
                 os_name: {
                     name: e_name,
                     cost: e_cost,
                     prob: e_prob,
                     access: e_access
                 }
             }
        """
        if self._e_map is None:
            e_map = {}
            for e_name, e_def in self.exploits.items():
                srv_name = e_def[u.EXPLOIT_SERVICE]
                if srv_name not in e_map:
                    e_map[srv_name] = {}
                srv_map = e_map[srv_name]

                os = e_def[u.EXPLOIT_OS]
                if os not in srv_map:
                    srv_map[os] = {
                        "name": e_name,
                        u.EXPLOIT_SERVICE: srv_name,
                        u.EXPLOIT_OS: os,
                        u.EXPLOIT_COST: e_def[u.EXPLOIT_COST],
                        u.EXPLOIT_PROB: e_def[u.EXPLOIT_PROB],
                        u.EXPLOIT_ACCESS: e_def[u.EXPLOIT_ACCESS]
                    }
            self._e_map = e_map
        return self._e_map

    @property
    def privesc_map(self):
        """A nested dictionary for all privilege escalation actions in scenario.

        I.e. {process_name: {
                 os_name: {
                     name: pe_name,
                     cost: pe_cost,
                     prob: pe_prob,
                     access: pe_access
                 }
             }
        """
        if self._pe_map is None:
            pe_map = {}
            for pe_name, pe_def in self.privescs.items():
                proc_name = pe_def[u.PRIVESC_PROCESS]
                if proc_name not in pe_map:
                    pe_map[proc_name] = {}
                proc_map = pe_map[proc_name]

                os = pe_def[u.PRIVESC_OS]
                if os not in proc_map:
                    proc_map[os] = {
                        "name": pe_name,
                        u.PRIVESC_PROCESS: proc_name,
                        u.PRIVESC_OS: os,
                        u.PRIVESC_COST: pe_def[u.PRIVESC_COST],
                        u.PRIVESC_PROB: pe_def[u.PRIVESC_PROB],
                        u.PRIVESC_ACCESS: pe_def[u.PRIVESC_ACCESS]
                    }
            self._pe_map = pe_map
        return self._pe_map

    @property
    def subnets(self):
        return self.scenario_dict[u.SUBNETS]

    @property
    def topology(self):
        return self.scenario_dict[u.TOPOLOGY]

    @property
    def sensitive_hosts(self):
        return self.scenario_dict[u.SENSITIVE_HOSTS]

    @property
    def sensitive_addresses(self):
        return list(self.sensitive_hosts.keys())

    @property
    def firewall(self):
        return self.scenario_dict[u.FIREWALL]

    @property
    def hosts(self):
        return self.scenario_dict[u.HOSTS]

    @property
    def address_space(self):
        return list(self.hosts.keys())

    @property
    def service_scan_cost(self):
        return self.scenario_dict[u.SERVICE_SCAN_COST]

    @property
    def os_scan_cost(self):
        return self.scenario_dict[u.OS_SCAN_COST]

    @property
    def subnet_scan_cost(self):
        return self.scenario_dict[u.SUBNET_SCAN_COST]

    @property
    def process_scan_cost(self):
        return self.scenario_dict[u.PROCESS_SCAN_COST]

    @property
    def address_space_bounds(self):
        if "address_space_bounds" in self.scenario_dict:
            return self.scenario_dict["address_space_bounds"]

        else:
            return len(self.subnets), max(self.subnets)

    @property
    def host_value_bounds(self):
        """The min and max values of host in scenario

        Returns
        -------
        (float, float)
            (min, max) tuple of host values
        """
        min_value = math.inf
        max_value = -math.inf
        for host in self.hosts.values():
            min_value = min(min_value, host.value)
            max_value = max(max_value, host.value)
        return (min_value, max_value)

    @property
    def host_discovery_value_bounds(self):
        """The min and max discovery values of hosts in scenario

        Returns
        -------
        (float, float)
            (min, max) tuple of host values
        """
        min_value = math.inf
        max_value = -math.inf
        for host in self.hosts.values():
            min_value = min(min_value, host.discovery_value)
            max_value = max(max_value, host.discovery_value)
        return (min_value, max_value)

    @property
    def scan_noise(self):
        """Scan noise configuration for realism"""
        return self.scenario_dict.get('scan_noise', {
            'service_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
            'os_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0},
            'process_scan': {'false_positive_rate': 0.0, 'false_negative_rate': 0.0}
        })

    @property
    def service_dynamics(self):
        """Service dynamics configuration for realistic service churn"""
        return self.scenario_dict.get('service_dynamics', {
            'churn_probability': 0.0,
            'affected_services': [],
            'restart_delay': 10,
            'churn_types': [
                {
                    'type': 'crash_restart',
                    'probability': 1.0,
                    'down_duration': [5, 15]
                }
            ]
        })
    
    @property
    def intrusion_detection(self):
        """Intrusion detection system (IDS) configuration"""
        return self.scenario_dict.get('intrusion_detection', {
            'enabled': False,
            'detection_decay': 0.98,
            'base_thresholds': [0.7, 0.8],
            'response_types': {
                'quarantine': 0.2,
                'patch': 0.4,
                'monitor': 0.4
            },
            'failed_exploit_multiplier': 3.0,
            'detection_increase': {
                'subnet_scan': 0.02,
                'service_scan': 0.05,
                'os_scan': 0.03,
                'process_scan': 0.03,
                'exploit_failed': 0.15,
                'exploit_success': 0.08,
                'privesc_failed': 0.20,
                'privesc_success': 0.10,
            }
        })

    @property
    def network_reliability(self):
        """Network reliability configuration for realistic timeouts"""
        return self.scenario_dict.get('network_reliability', {
            'timeout_probability': 0.0,
            'affected_actions': [],
            'retry_cost': 0,
            'timeout_types': [
                {
                    'type': 'packet_loss',
                    'probability': 1.0,
                    'duration': 1
                }
            ]
        })

    def display(self):
        pprint(self.scenario_dict)

    def get_action_space_size(self):
        num_exploits = len(self.exploits)
        num_privescs = len(self.privescs)
        # OSScan, ServiceScan, SubnetScan, ProcessScan
        num_scans = 4
        actions_per_host = num_exploits + num_privescs + num_scans
        return len(self.hosts) * actions_per_host

    def get_state_space_size(self):
        # compromised, reachable, discovered
        host_aux_bin_features = 3
        num_bin_features = (
            host_aux_bin_features
            + self.num_os
            + self.num_services
            + self.num_processes
        )
        # access
        num_tri_features = 1
        host_states = 2**num_bin_features * 3**num_tri_features
        return len(self.hosts) * host_states

    def get_state_dims(self):
        # compromised, reachable, discovered, value, discovery_value, access
        host_aux_features = 6
        host_state_size = (
            self.address_space_bounds[0]
            + self.address_space_bounds[1]
            + host_aux_features
            + self.num_os
            + self.num_services
            + self.num_processes
        )
        return len(self.hosts), host_state_size

    def get_observation_dims(self):
        state_dims = self.get_state_dims()
        return state_dims[0]+1, state_dims[1]

    def get_description(self):
        description = {
            "Name": self.name,
            "Type": "generated" if self.generated else "static",
            "Subnets": len(self.subnets),
            "Hosts": len(self.hosts),
            "OS": self.num_os,
            "Services": self.num_services,
            "Processes": self.num_processes,
            "Exploits": len(self.exploits),
            "PrivEscs": len(self.privescs),
            "Actions": self.get_action_space_size(),
            "Observation Dims": self.get_observation_dims(),
            "States": self.get_state_space_size(),
            "Step Limit": self.step_limit
        }
        return description
