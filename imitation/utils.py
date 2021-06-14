from batch_utils import evaluate_policy
import numpy as np
import torch
import wandb

dart_step = 0

def train_learner(args, learner, trajs, env_sampler):
    training_losses = []
    states = np.array([state for traj in trajs for state, _, _ in traj])
    actions = np.array([i_action for traj in trajs for _, _, i_action in traj])
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(states), torch.FloatTensor(actions))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.policy_train_batch_size, shuffle=True)

    for epoch in range(100):
        for expert_states, expert_actions in dataloader:
            expert_states = expert_states.to(learner.device)
            expert_actions = expert_actions.to(learner.device)
            learner.policy_optim.zero_grad()
            _, _, mean = learner.policy.sample(expert_states)
            policy_loss = torch.nn.MSELoss()(mean, expert_actions)
            policy_loss.backward()
            learner.policy_optim.step()
            loss_val = policy_loss.cpu().detach().item()
            training_losses.append(loss_val)
        if (epoch + 1) % 10 == 0:
            rew, cost = evaluate_policy(args, env_sampler, learner, args.epoch_length)
            # REPLACE WITH WANDB
            wandb.log({
                "Dart/step": dart_step,
                "Dart/reward": rew,
                "Dart/cost": cost,
            })
            print(f"Epoch {epoch+1} Reward: {rew}, Cost: {cost}")

def split_data(data):
    l = len(data)
    gap = l // 50
    k = np.random.randint(0, gap)
    train = []
    holdout = data[k::gap]
    for i in range(gap):
        if i != k:
            train += data[i::gap]
    return train, holdout
