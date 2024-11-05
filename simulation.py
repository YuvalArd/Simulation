from config import Config
import analysis_functions as af
import numpy as np
import itertools
import scipy
import json
import copy

CONFIG_FILE = 'config.json'
OUTPUT_FILE = 'output.json'

SRC_INDEXES = {
    'public_video': 0,
    'private_video_src': 1,
    'private_video_dst': 2,
    'communication': 3,
    'private_video': 4,
    'private_src': 5,
    'private_dst': 6
}

class Simulation:
    """
    Handles the simulation processes for regression analysis based on configured parameters.

    Attributes:
        config (Config): Configuration object containing simulation parameters.
        weight_functions (dict): Mapping of weight function names to their implementations.
        space_sources (dict): Sources for different spaces (e.g., video, src, dst).
    """
    def __init__(self):
        self.config = Config(CONFIG_FILE)
        """" Mapped functions can be extended, each new function can except arguments from the configuration and
             has to return a np.ndarray of size P """
        self.weight_functions = {"linear": self.linear_weights}
        self.space_sources = {'video': ['public_video', 'private_video_src', 'private_video_dst', 'private_video'],
                              'src': ['public_video', 'private_video_src', 'communication', 'private_src'],
                              'dst': ['public_video', 'private_video_dst', 'communication', 'private_dst']}

    def linear_weights(self, args):
        """
        Generate linearly spaced weights based on provided arguments.

        Args:
            args (dict): Dictionary containing 'start' and 'end' keys for weight generation.

        Returns:
            np.ndarray: Array of linearly spaced weights.
        """
        return np.linspace(args["start"], args["end"], num=self.config.P)

    def generate_orthogonal_matrices(self, N, P, sources):
        """
        Generate orthogonal matrices for U and V sources.

        Args:
            N (int): Number of observations.
            P (int): Number of predictors.
            sources (int): Number of sources.

        Returns:
            tuple:
                - np.ndarray: Orthogonal U matrices.
                - np.ndarray: Orthogonal V matrices.
        """
        U = scipy.linalg.orth(np.random.normal(0, 1, (sources * N, P)))
        V = scipy.linalg.orth(np.random.normal(0, 1, (sources * P, P)))
        return U, V

    def generate_sources(self, N, P):
        """
        Generate orthogonal source matrices for simulation.

        Args:
            N (int): Number of observations.
            P (int): Number of predictors.

        Returns:
            tuple:
                - np.ndarray: U_sources with shape (sources, N, P).
                - np.ndarray: V_sources with shape (sources, P, P).
        """
        sources = len(SRC_INDEXES)
        U_sources = np.full((sources, N, P), np.nan)
        V_sources = np.full((sources, P, P), np.nan)
        U, V = self.generate_orthogonal_matrices(N, P, sources)
        for i in range(sources):
            U_sources[i], V_sources[i] = U[i * N:(i + 1) * N, :], V[i * P:(i + 1) * P, :]
        return U_sources, V_sources

    def create_singular_values(self):
        """
        Create power-law descending singular values based on configuration parameters of slope and PC fraction.

        Returns:
            np.ndarray: Array of singular values.
        """
        P, slope = self.config.P, self.config.slope
        if self.config.partial_pcs:
            s1 = np.arange(1.0, self.config.pc_part * P + 1) ** slope
            s2 = np.full((int((1 - self.config.pc_part) * P),), 0)
            return np.concatenate((s1, s2))
        return np.arange(1.0, P + 1) ** slope

    def arrange_singular_values(self, switch_index, private_index):
        """
        Rearrange singular values based on mixing or shifting configurations.

        Args:
            switch_index (int): Index for switching singular values.
            private_index (int): Index of the first private PC.

        Returns:
            list: Rearranged list of singular values.
        """
        S = self.create_singular_values().tolist()
        if self.config.mix:
            from_indexes = self.config.indexes['from']
            to_indexes = self.config.indexes['to'][switch_index]
            switch_values = dict(zip(from_indexes, to_indexes))
            sorted_values = dict(sorted(switch_values.items(), key=lambda item: item[1]))
            for i, (fr, to) in enumerate(sorted_values.items()):
                switch_values[fr] = S.pop(to - i)
            for i in sorted(switch_values.keys()):
                S = S[:i] + [switch_values[i]] + S[i:]
        if self.config.shift:
            shift = self.config.shift_list[switch_index]
            first_dims, S = S[:shift], S[shift:]
            S = S[:private_index] + first_dims + S[private_index:]
        return S

    def select_sources(self, sources, keys):
        """
        Select specific sources based on space keys.

        Args:
            sources (np.ndarray): List of source matrices.
            keys (list): List of keys to select from sources.

        Returns:
            list: Selected source matrices.
        """
        return [sources[SRC_INDEXES[key]] for key in keys]

    def add_pcs(self, sources, sizes):
        """
        Add principal components from sources based on given sizes.

        Args:
            sources (list): List of source matrices.
            sizes (list): List of sizes indicating the number of PCs to retain for each source.

        Returns:
            np.ndarray: Combined matrix.
        """
        space_pcs = []
        for i, src in enumerate(sources):
            if sizes[i] > 0:
                space_pcs.append(src[:, :sizes[i]])
        return np.hstack(space_pcs)

    def create_joined_pcs(self, sources, weights):
        """
        Create joined principal components by weighting and summing sources.

        Args:
            sources (list): List of source matrices.
            weights (list): List of weights corresponding to each source.

        Returns:
            np.ndarray: Joined principal components.
        """
        return np.sum(np.array([src * weight for src, weight in zip(sources, weights)]), axis=0)

    def create_space_joined_pcs(self, weights, U_sources, V_sources):
        """
        Construct the space matrix by combining U, V sources with weights and singular values.

        Args:
            weights (list): Dictionary of weights for each subspace.
            U_sources (list): Array of U source matrices.
            V_sources (list): Array of V source matrices.

        Returns:
            np.ndarray: Combined simulated matrix.
        """
        S = self.create_singular_values()
        U = self.create_joined_pcs(U_sources, weights)
        V = self.create_joined_pcs(V_sources, weights)
        return np.dot(np.dot(U, np.diag(S)), V.T)

    def create_space_discrete_pcs(self, sizes,  U_sources, V_sources, rearranged=-1):
        """
        Construct the space matrix with specified principal component sizes.

        Args:
            sizes (list): List of sizes for each subspace.
            U_sources (list): Array of U source matrices.
            V_sources (list): Array of V source matrices.
            rearranged (int, optional): Index indicating rearrangement. Default is -1 (no rearrangement).

        Returns:
            np.ndarray: Combined simulated matrix.
        """
        if rearranged >= 0:
            S = self.arrange_singular_values(rearranged, sum(sizes[:-1]))
        else:
            S = self.create_singular_values()
        U = self.add_pcs(U_sources, sizes)
        V = self.add_pcs(V_sources, sizes)
        return np.dot(np.dot(U, np.diag(S)), V.T)

    def calc_subspace_sizes(self, P, sizes):
        """
        Calculate sizes for different subspaces.

        Args:
            P (int): Total number of predictors.
            sizes (list): List of sizes for specific subspaces.

        Returns:
            tuple:
                - list: Area sizes including all subspace sizes for the source and target areas.
                - list: Video sizes including all subspace sizes for the video space.
        """
        area_sizes, video_sizes = list(sizes), list(sizes)
        area_sizes.append(P - sum(sizes))
        video_sizes[2] = video_sizes[1]
        video_sizes.append(P - sizes[0] - 2 * sizes[1])
        return area_sizes, video_sizes

    def create_spaces_discrete_pcs(self, P, sizes,  U_sources, V_sources, rearranged_index=-1):
        """
        Create matrices for all defined spaces based on subspace sizes.

        Args:
            P (int): Number of predictors.
            sizes (list): List of sizes for each subspace.
            U_sources (np.ndarray): Array of U source matrices.
            V_sources (np.ndarray): Array of V source matrices.
            rearranged_index (int, optional): Index for rearrangement. Default is -1 (no rearrangement).

        Returns:
            dict: Dictionary of matrices for each space.
        """
        spaces = {}
        area_sizes, video_sizes = self.calc_subspace_sizes(P, sizes)
        for index, (key, value) in enumerate(self.space_sources.items()):
            if key == 'video':
                space = self.create_space_discrete_pcs(video_sizes, self.select_sources(U_sources, value),
                                                       self.select_sources(V_sources, value))
            else:
                space = self.create_space_discrete_pcs(area_sizes, self.select_sources(U_sources, value),
                                                       self.select_sources(V_sources, value), rearranged_index)
            spaces[key] = space
        return spaces

    def create_spaces_joined_pcs(self, weights_values, U_sources, V_sources):
        """
        Create matrices for all defined spaces based on weights.

        Args:
            weights_values (dict): Weights for each subspace.
            U_sources (np.ndarray): Array of U source matrices.
            V_sources (np.ndarray): Array of V source matrices.

        Returns:
            dict: Dictionary of joined prediction matrices for each space.
        """
        spaces = {}
        weights = [weights_values['video_public'], weights_values['video_private'], weights_values['communication'],
                   weights_values['private']]

        for key, value in self.space_sources.items():
            curr_weights = weights.copy()
            if key == 'video':
                curr_weights[2] = curr_weights[1]
            spaces[key] = self.create_space_joined_pcs(curr_weights, self.select_sources(U_sources, value),
                                                       self.select_sources(V_sources, value))
        return spaces

    def iterate_weights(self):
        """
        Iterate through all possible combinations of subspace weights based on configuration.

        Yields:
            dict: Deep copy of the current weight configuration.
        """
        step = self.config.weight_step
        config = self.config.subspace_weights

        for vp_start in np.arange(config['video_public']['limits']['min'], config['video_public']['limits']['max'] + step, step):
            config["video_public"]["arguments"]["start"] = vp_start
            for vp_end in np.arange(0, vp_start + step, step):
                config["video_public"]["arguments"]["end"] = vp_end
                for vpr_start in np.arange(config['video_private']['limits']['min'], vp_start, step):
                    config["video_private"]["arguments"]["start"] = vpr_start
                    for vpr_end in np.arange(0, vpr_start + step, step):
                        config["video_private"]["arguments"]["end"] = vpr_end
                        for comm_start in np.arange(config['communication']['limits']['min'], vpr_start, step):
                            config["communication"]["arguments"]["start"] = comm_start
                            for comm_end in np.arange(comm_start, config['communication']['limits']['max'] + step, step):
                                config["communication"]["arguments"]["end"] = comm_end
                                for pri_start in np.arange(config['private']['limits']['min'], config['private']['limits']['max'] + step, step):
                                    config["private"]["arguments"]["start"] = pri_start
                                    for pri_end in np.arange(pri_start, config['private']['limits']['max'] + step, step):
                                        config["private"]["arguments"]["end"] = pri_end
                                yield copy.deepcopy(config)

    def predictions(self, spaces, N, P):
        """
        Generate predictions and R² metrics based on the current space configurations.

        Args:
            spaces (dict): Dictionary of matrices for each space.
            N (int): Number of observations.
            P (int): Number of predictors.

        Returns:
            dict: Dictionary containing R² metrics for video, communication-subspace, and residuals.
        """
        video_predictions = {space: np.full((N, P), np.nan) for space in ['src', 'dst']}

        # video prediction + video r2 for dst
        _, video_r2, _, video_predictions['dst'] = af.cv_regression(spaces['video'], spaces['dst'])

        # video prediction for src
        _, _, _, video_predictions['src'] = af.cv_regression(spaces['video'], spaces['src'])

        # dst by src
        _, space_to_space_r2, _, _ = af.cv_regression(spaces['src'], spaces['dst'])

        # dst by src no video
        _, space_to_space_no_video_r2, _, _ = af.cv_regression(spaces['src'] - video_predictions['src'],
                                                               spaces['dst'] - video_predictions['dst'])

        return {'video': video_r2, 'cs': space_to_space_r2, 'residual': space_to_space_no_video_r2}

    def check_results(self, video_r2, space_to_space_r2, space_to_space_no_video_r2):
        """
        Validate the simulation results based on predefined R² thresholds.

        Args:
           video_r2 (float): R² value for video predictions.
           space_to_space_r2 (float): R² value for cross-space predictions.
           space_to_space_no_video_r2 (float): R² value for residual cross-space predictions.

        Returns:
           bool: True if all R² values exceed thresholds and video R² exceeds cross-space R²; False otherwise.
        """
        if video_r2 > 0.01 and space_to_space_r2 > 0.01 and space_to_space_no_video_r2 > 0.01 and \
                video_r2 > space_to_space_r2:
            return True
        return False

    def write_results(self, results):
        """
        Write simulation results to the output JSON file.

        Args:
            results (list or dict): Results to be serialized and saved.
        """
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f)

    def run_simulation_joined_pcs(self, N, P, weights=None, sources=None):
        """
        Execute a single simulation using joined principal components.

        Args:
            N (int): Number of observations.
            P (int): Number of predictors.
            weights (dict, optional): Weight configurations for subspaces. Defaults to None.
            sources (tuple, optional): Tuple containing U_sources and V_sources. Defaults to None.

        Returns:
            dict: R² results from the predictions.
        """
        weights_values = {key: self.weight_functions[value["function"]](value["arguments"]) for key, value in
                          weights.items()}
        U_sources, V_sources = sources
        spaces = self.create_spaces_joined_pcs(weights_values, U_sources, V_sources)
        return self.predictions(spaces, N, P)

    def single_simulation_joined_pcs(self):
        """
        Perform a single simulation run with joined principal components and save the results.
        """
        N, P = self.config.N, self.config.P
        weights = self.config.subspace_weights
        U_sources, V_sources = self.generate_sources(N, P)
        results = self.run_simulation_joined_pcs(N, P, weights, (U_sources, V_sources))
        self.write_results(results)

    def multiple_simulations_joined_pcs(self, logger=None):
        """
        Conduct multiple simulations across all weight configurations and log the outcomes.

        Args:
            logger (logging.Logger, optional): Logger for recording simulation progress. Defaults to None.
        """
        N, P = self.config.N, self.config.P
        results = []
        U_sources, V_sources = self.generate_sources(N, P)

        for weights in self.iterate_weights():
            r2 = self.run_simulation_joined_pcs(N, P, weights, (U_sources, V_sources))
            if logger:
                if self.check_results(*r2):
                    logger.info(f"success - {weights}")
                    logger.info(f"success - {r2}")
                else:
                    logger.debug(f"fail - {weights}")
                    logger.debug(f"fail - {r2}")
            results.append([weights, r2])

        self.write_results(results)

    def run_simulation_discrete_pcs(self, N, P, sizes, U_sources, V_sources, results, rearranged=-1):
        """
        Execute a single simulation using discrete principal components.

        Args:
            N (int): Number of observations.
            P (int): Number of predictors.
            sizes (list): Sizes for each subspace.
            U_sources (np.ndarray): Array of U source matrices.
            V_sources (np.ndarray): Array of V source matrices.
            results (list): List to append the simulation results.
            rearranged (int, optional): Index for rearrangement. Default is -1 (no rearrangement).
        """
        spaces = self.create_spaces_discrete_pcs(P, sizes, U_sources, V_sources, rearranged)
        r2 = self.predictions(spaces, N, P)
        results.append([sizes, r2])

    def simulations_discrete_pcs(self):
        """
        Perform simulations using discrete principal components across all size combinations.
        """
        N, P = self.config.N, self.config.P
        U_sources, V_sources = self.generate_sources(N, P)
        video_public, video_private, communication = self.config.subspace_sizes.values()
        results = []

        for sizes in itertools.product(video_public, video_private, communication):
            self.run_simulation_discrete_pcs(N, P, sizes, U_sources, V_sources, results)
        self.write_results(results)

    def simulations_rearranged_discrete_pcs(self):
        """
        Perform simulations using discrete principal components with rearrangements (shifting/mixing).
        """
        N, P = self.config.N, self.config.P
        U_sources, V_sources = self.generate_sources(N, P)
        video_public, video_private, communication = self.config.subspace_sizes.values()
        results = []

        for sizes in itertools.product(video_public, video_private, communication):
            if self.config.shift:
                for i, _ in enumerate(self.config.shift_list):
                    self.run_simulation_discrete_pcs(N, P, sizes, U_sources, V_sources, results, i)
            if self.config.mix:
                for i, _ in enumerate(self.config.indexes['to']):
                    self.run_simulation_discrete_pcs(N, P, sizes, U_sources, V_sources, results, i)
        self.write_results(results)

