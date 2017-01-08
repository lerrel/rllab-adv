from rllab.algos.base import RLAlgorithm
from rllab.sampler import parallel_sampler
from rllab.sampler.base import BaseSampler
import rllab.misc.logger as logger
import rllab.plotter as plotter
from rllab.policies.base import Policy
from copy import deepcopy as copy


class BatchSampler(BaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.pro_policy, scope=self.algo.scope, adv_policy=self.algo.adv_policy)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_pro_params = self.algo.pro_policy.get_param_values()
        cur_adv_params = self.algo.adv_policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            pro_policy_params=cur_pro_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
            adv_policy_params=cur_adv_params
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            pro_policy,
            pro_baseline,
            adv_policy,
            adv_baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            sampler_cls=None,
            sampler_args=None,
            is_protagonist=True,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.pro_policy = pro_policy
        self.pro_baseline = pro_baseline
        self.adv_policy = adv_policy
        self.adv_baseline = adv_baseline
        self.scope = scope
        self.n_itr = n_itr
        self.current_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.is_protagonist = is_protagonist
        if sampler_cls is None:
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        if self.is_protagonist==True:
            self.policy = self.pro_policy
            self.baseline = self.pro_baseline
        else:
            self.policy = self.adv_policy
            self.baseline = self.adv_baseline
        self.start_worker()
        self.init_opt()

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def train(self):
        self.rews = []
        for itr in range(0, self.n_itr):
            #with logger.prefix('itr #%d | ' % itr):
            with logger.prefix(''):
                logger.log('itr #%d | ' % itr)
                all_paths = self.sampler.obtain_samples(itr)
                paths = self.get_agent_paths(all_paths, is_protagonist=self.is_protagonist)
                #from IPython import embed; embed()
                samples_data = self.sampler.process_samples(itr, paths)
                self.log_diagnostics(paths)
                self.optimize_policy(itr, samples_data)
                #logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                #logger.save_itr_params(itr, params)
                #logger.log("saved")
                #logger.dump_tabular(with_prefix=False)
                self.rews.append(self.get_average_reward(paths))
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

        #self.shutdown_worker()

    def get_agent_paths(self, paths, is_protagonist=True):
        cur_paths = copy(paths)
        for p in cur_paths:
            if is_protagonist==True:
                p['actions']=p.pop('pro_actions')
                del p['adv_actions']
                p['agent_infos']=p.pop('pro_agent_infos')
                del p['adv_agent_infos']
            else:
                alpha = -1.0
                p['actions']=p.pop('adv_actions')
                del p['pro_actions']
                p['rewards']=alpha*p['rewards']
                p['agent_infos']=p.pop('adv_agent_infos')
                del p['pro_agent_infos']
        return cur_paths

    def get_average_reward(self, paths):
        sum_r = 0.0
        for p in paths:
            sum_r +=p['rewards'].sum()
        return sum_r/len(paths)

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using theano / cgt, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
