import numpy as np

class Supervisor:
    def __init__(self, action_dim, teacher, learner):
        self.teacher = teacher
        self.learner = learner
        self.action_dim = action_dim
        self.cov = np.zeros((action_dim, action_dim))
        self.i_action = None
    
    def fit_cov(self, trajs):
        self.cov = np.zeros((self.action_dim, self.action_dim))
        for traj in trajs:
            sup_actions = np.array([i_action for _, _, i_action in traj])
            lnr_actions = np.array([self.learner.select_action(state, eval=True) for state, _, _ in traj])
            diff = sup_actions - lnr_actions
            self.cov += np.dot(diff.T, diff) / float(len(traj))
        self.cov /= float(len(trajs))
        
    def select_action(self, obs, *args, **kwargs):
        i_action = self.teacher.select_action(obs, eval_t=True)
        action = np.random.multivariate_normal(i_action, self.cov)
        self.i_action = i_action
        return action