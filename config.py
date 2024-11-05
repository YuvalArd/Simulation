import json

class Config:
    """
    Configuration handler for simulation parameters.

    Attributes:
        file (str): Path to the configuration JSON file.
        N (int): Number of samples.
        P (int): Number of simulated neurons.
        slope (float): Slope parameter for singular values.
        subspace_sizes (dict): Sizes of various subspaces.
        subspace_weights (dict): Weight configurations for subspaces.
        weight_step (float): Step size for iterating weights.
        partial_pcs (bool): Indicator for partial principal components.
        pc_part (float): Fraction of principal components to retain if partial_pcs is True.
        mix (bool): Indicator for mixing indexes.
        indexes (dict): Mapping of indexes for mixing.
        shift (bool): Indicator for shifting singular values.
        shift_list (list): List of shifts to apply.
    """

    def __init__(self, file_name):
        """
        Initialize the Config object by loading parameters from a JSON file.

        Args:
            file_name (str): Path to the configuration JSON file.
        """
        self.file = file_name
        self.N = 0
        self.P = 0
        self.slope = 0
        self.subspace_sizes = {}
        self.subspace_weights = {}
        self.weight_step = 0.1
        self.partial_pcs = False
        self.pc_part = 0
        self.mix = False
        self.indexes = {}
        self.shift = False
        self.shift_list = []
        self.load_config()

    def read_config(self):
        """
        Read and parse the JSON configuration file.

        Returns:
            dict: Parsed configuration parameters.
        """
        with open(self.file, 'r') as config_file:
            return json.load(config_file)

    def load_config(self):
        """
        Load configuration parameters into the Config object's attributes.
        """
        config = self.read_config()
        self.N, self.P, self.slope = config.get('consts').values()
        self.subspace_weights = config.get('subspace_weights')
        if config.get('weight_step'):
            self.weight_step = config.get('weight_step')
        self.subspace_sizes = config.get('subspace_sizes')
        if config.get('partial_pcs') is not None:
            self.partial_pcs = True
            self.pc_part = config.get('partial_pcs')
        index_change = config.get('index_change')
        if index_change:
            self.mix = index_change.get('mix', False)
            self.indexes = {'from': index_change.get('from', []), 'to': index_change.get('to', [])}
            self.shift = index_change.get('shift', False)
            self.shift_list = index_change.get('shift_list')
