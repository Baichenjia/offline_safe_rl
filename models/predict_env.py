import numpy as np
import torch

import wandb

class PredictEnv:
    def __init__(self, model, env_name):
        self.model = model
        self.env_name = env_name

    def _termination_fn(self, env_name, obs, act, next_obs):
        prefix = env_name.split('-')[0]
        if env_name == "Hopper-v2" or prefix == 'hopper' or prefix == 'Hopper':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  np.isfinite(next_obs).all(axis=-1) \
                        * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:,None]
            return done

        elif env_name == 'HalfCheetah-v2' or prefix == 'halfcheetah' or prefix == 'HalfCheetah':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            done = np.array([False]).repeat(len(obs))
            done = done[:, None]
            return done

        elif env_name == "Walker2d-v2" or prefix == 'walker2d' or prefix == 'Walker2d':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  (height > 0.8) \
                        * (height < 2.0) \
                        * (angle > -1.0) \
                        * (angle < 1.0)
            done = ~not_done
            done = done[:,None]
            return done

    def step(self, obs, act, deterministic=False, reward_penalty=1, algo='gambol'):
        assert len(obs.shape) == 2
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)

        ensemble_model_means[:,:,self.model.reward_size:] += obs
        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + torch.randn(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape

        model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)

        batch_idxes = np.arange(0, batch_size)
        samples = ensemble_samples[model_idxes, batch_idxes]

        # model_means = ensemble_model_means[model_idxes, batch_idxes]
        # model_stds = ensemble_model_stds[model_idxes, batch_idxes]
        # log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:self.model.reward_size], samples[:,self.model.reward_size:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        # MOPO soft penalty
        if reward_penalty != 0 and algo == 'mopo':
            penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
            penalty = np.expand_dims(penalty, 1)
            penalized_rewards = rewards - reward_penalty * penalty
            info = {}
        else:
            penalized_rewards = rewards
            info = {}
        return next_obs, penalized_rewards, terminals, info