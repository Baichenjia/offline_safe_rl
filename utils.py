import d4rl
import gym
import h5py
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader


def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def get_safetygym_dataset(h5path=None):
    dataset_file = h5py.File(h5path, 'r')
    data_dict = {}
    for k in get_keys(dataset_file):
        try:
            # first try loading as an array
            data_dict[k] = dataset_file[k][:]
        except ValueError as e:  # try loading as a scalar
            data_dict[k] = dataset_file[k][()]
    dataset_file.close()

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'info/cost', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key


    return data_dict


def qlearning_dataset(env, dataset, terminate_on_end=False):
    """Construct qlearning style dataset for model-learning."""
    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        # combine reward and cost in one vector
        cost = dataset['info/cost'][i].astype(np.float32)
        reward = np.array([reward, cost])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env.num_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def load_d4rl_dataset(env_name='halfcheetah-expert-v0'):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    # dataset['rewards'] = np.expand_dims(dataset['rewards'], axis=1)
    # dataset['terminals'] = np.expand_dims(dataset['terminals'], axis=1)
    return env, dataset


def load_normalized_dataset(env_name='hopper', dataset_name='medium-replay-v0'):
    x_train, y_train, x_test, y_test = np.load(f'data/{env_name}-{dataset_name}-normalized-data.npy', allow_pickle=True)
    return x_train,y_train,x_test, y_test


def multistep_dataset(env, h=2, terminate_on_end=False, **kwargs):
    dataset = env.get_dataset(**kwargs)
    N = dataset['rewards'].shape[0]

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - h):
        skip = False
        for j in range(i, i + h - 1):
            if bool(dataset['terminals'][j]) or dataset['timeouts'][j]:
                skip = True
        if skip:
            continue

        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + h]
        action = dataset['actions'][i:i + h].flatten()
        reward = dataset['rewards'][i + h - 1]
        done_bool = bool(dataset['terminals'][i + h - 1])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i + h - 1]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def format_samples_for_training(samples):
    obs = samples['observations']
    act = samples['actions']
    next_obs = samples['next_observations']
    rew = samples['rewards']
    delta_obs = next_obs - obs
    inputs = np.concatenate((obs, act), axis=-1)
    outputs = np.concatenate((rew.reshape(rew.shape[0], -1), delta_obs), axis=-1)

    # inputs = torch.from_numpy(inputs).float()
    # outputs = torch.from_numpy(outputs).float()

    return inputs, outputs


def create_data_loader(X,y, train_n=5000, test_n=6000, batch_size=64):
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:test_n], y[train_n:test_n]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


def batch_generator(index_array, batch_size):
    index_array = shuffle_rows(index_array)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= index_array.shape[1]:
            batch_count = 0
            index_array = shuffle_rows(index_array)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield index_array[:, start:end]