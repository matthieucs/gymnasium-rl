import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
from collections import deque

import torch
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical

matplotlib.use("Qt5Agg")  # use TkAgg for real-time plotting, change to "Qt5Agg" if it doesn't work

### Hyperparameters ###
NUM_ENV = 16
CAPTURE_VIDEO = True
RUN_NAME = f"ppo_lunarlander_{int(time.time())}"
GAMMA = 0.99
LAMBDA = 0.95
NUMBER_ITERATIONS = 1000
EPISODE_LENGTH = 512
NUMBER_EPOCHS = 4
NUM_MINIBATCH = 4
CLIP_COEF = 0.2
BATCH_SIZE = int(NUM_ENV * EPISODE_LENGTH)
MINIBATCH_SIZE = int(BATCH_SIZE // NUM_MINIBATCH)
NORM_ADV = True
VF_COEF = 0.5
ENT_COEF = 0.05
MAX_GRAD_NORM = 0.5
LEARNING_RATE = 2.5e-4

print(f"Batch size: {BATCH_SIZE}, Minibatch size: {MINIBATCH_SIZE}")

#### Creating the environments ####
def create_envs(idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                "LunarLander-v3",
                render_mode="rgb_array",
                continuous=False,
                gravity=-10.0,
                enable_wind=False,
                wind_power=15.0,
                turbulence_power=1.5)
            env =gym.wrappers.RecordVideo(
                    env, 
                    f"videos/{run_name}",
                    episode_trigger=lambda ep_id: ep_id % 50 == 0  # record every 50th episode
                )
        else:
            env = gym.make("LunarLander-v3",
                           continuous=False,
                           gravity=-10.0,
                           enable_wind=False,
                           wind_power=15.0,
                           turbulence_power=1.5)
        return env
    return thunk

envs = gym.vector.SyncVectorEnv(
    [create_envs(i, CAPTURE_VIDEO, RUN_NAME) for i in range(NUM_ENV)],
)
envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

#### Networks ####
device = torch.device('cpu')

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def probs(self, obs):
        logits = self.network(obs)
        return Categorical(logits=logits)

    def get_logprob(self, obs, action):
        return self.probs(obs).log_prob(action)

    def get_entropy(self, obs):
        return self.probs(obs).entropy()

    def forward(self, obs):
        return self.probs(obs).sample()


class CriticNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, obs):
        return self.network(obs)

### Initialization ###
actor = ActorNetwork(envs).to(device)
critic = CriticNetwork(envs).to(device)
optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()),
    lr=LEARNING_RATE,
    eps=1e-5
)

#### Real-time plotting setup ####
plt.ion()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("PPO Training — LunarLander", fontsize=14)

ax_reward      = axes[0, 0]
ax_clip_loss   = axes[0, 1]
ax_value_loss  = axes[0, 2]
ax_entropy     = axes[1, 0]
ax_kl          = axes[1, 1]
ax_expvar      = axes[1, 2]

# Rolling history
history = {
    "episodic_returns": deque(maxlen=200),  # raw episode returns as they come in
    "clip_loss":        [],
    "value_loss":       [],
    "entropy_loss":     [],
    "approx_kl":        [],
    "explained_var":    [],
    "iterations":       [],
    "mean_returns":     [],  # mean return per iteration (smoothed)
}

def update_plots():
    for ax in axes.flat:
        ax.cla()

    iters = history["iterations"]

    # Episodic returns — scatter all episodes + rolling mean
    if history["episodic_returns"]:
        returns_arr = list(history["episodic_returns"])
        ax_reward.plot(returns_arr, alpha=0.3, color="steelblue", linewidth=0.8)
        # rolling mean over last 20
        if len(returns_arr) >= 20:
            rolling = np.convolve(returns_arr, np.ones(20)/20, mode='valid')
            ax_reward.plot(range(19, len(returns_arr)), rolling, color="steelblue", linewidth=2)
    ax_reward.set_title("Episodic Return")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Return")
    ax_reward.grid(True, alpha=0.3)

    # Losses and metrics per iteration
    if iters:
        ax_clip_loss.plot(iters, history["clip_loss"], color="tomato", linewidth=1.5)
        ax_clip_loss.set_title("Policy (Clip) Loss")
        ax_clip_loss.set_xlabel("Iteration")
        ax_clip_loss.grid(True, alpha=0.3)

        ax_value_loss.plot(iters, history["value_loss"], color="orange", linewidth=1.5)
        ax_value_loss.set_title("Value Loss")
        ax_value_loss.set_xlabel("Iteration")
        ax_value_loss.grid(True, alpha=0.3)

        ax_entropy.plot(iters, history["entropy_loss"], color="mediumseagreen", linewidth=1.5)
        ax_entropy.set_title("Entropy")
        ax_entropy.set_xlabel("Iteration")
        ax_entropy.grid(True, alpha=0.3)

        ax_kl.plot(iters, history["approx_kl"], color="mediumpurple", linewidth=1.5)
        ax_kl.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, label="target KL")
        ax_kl.set_title("Approx KL Divergence")
        ax_kl.set_xlabel("Iteration")
        ax_kl.legend(fontsize=8)
        ax_kl.grid(True, alpha=0.3)

        ax_expvar.plot(iters, history["explained_var"], color="goldenrod", linewidth=1.5)
        ax_expvar.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="perfect")
        ax_expvar.set_title("Explained Variance")
        ax_expvar.set_xlabel("Iteration")
        ax_expvar.set_ylim(-1, 1.1)
        ax_expvar.legend(fontsize=8)
        ax_expvar.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.pause(0.01)

#### Rollout buffers ####
obs          = torch.zeros((EPISODE_LENGTH, NUM_ENV) + envs.single_observation_space.shape).to(device)
actions      = torch.zeros((EPISODE_LENGTH, NUM_ENV) + envs.single_action_space.shape).to(device)
logprobs     = torch.zeros((EPISODE_LENGTH, NUM_ENV)).to(device)
rewards      = torch.zeros((EPISODE_LENGTH, NUM_ENV)).to(device)
gae          = torch.zeros((EPISODE_LENGTH, NUM_ENV)).to(device)
dones        = torch.zeros((EPISODE_LENGTH, NUM_ENV)).to(device)
state_values = torch.zeros((EPISODE_LENGTH, NUM_ENV)).to(device)

#### Initial reset ####
current_obs, infos = envs.reset()
current_obs  = torch.Tensor(current_obs).to(device)
current_done = torch.zeros(NUM_ENV).to(device)

#### Training loop ####
for iteration in range(NUMBER_ITERATIONS):

    # Optimizer decay
    frac = 1.0 - iteration / NUMBER_ITERATIONS
    optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

    ## --- Rollout collection --- ##
    for timestep in range(EPISODE_LENGTH):
        obs[timestep]    = current_obs
        dones[timestep]  = current_done

        with torch.no_grad():
            current_actions      = actor(current_obs)
            current_logprob      = actor.get_logprob(current_obs, current_actions)
            current_state_values = critic(current_obs).flatten()  # BUG FIX: was never stored before

        actions[timestep]      = current_actions
        logprobs[timestep]     = current_logprob
        state_values[timestep] = current_state_values  # BUG FIX: this line was missing

        current_obs, current_rewards, terminations, truncations, infos = envs.step(current_actions.cpu().numpy())

        current_done = np.logical_or(terminations, truncations)
        rewards[timestep] = torch.tensor(current_rewards).to(device).view(-1)
        current_obs  = torch.Tensor(current_obs).to(device)
        current_done = torch.Tensor(current_done).to(device)

        if "episode" in infos:
        # episode['r'] is now an array of shape (NUM_ENV,)
        # _episode is a boolean mask of which envs finished
            for i, finished in enumerate(infos["_episode"]):
                if finished:
                    ep_return = float(infos["episode"]["r"][i])
                    history["episodic_returns"].append(ep_return)
                    #print(f"[iter {iteration}] episodic_return={ep_return:.1f}")


    ## --- GAE computation --- ##
    with torch.no_grad():
        t_1_state_values = critic(current_obs).reshape(1, -1)
        current_gae = 0
        for timestep in reversed(range(EPISODE_LENGTH)):
            if timestep == EPISODE_LENGTH - 1:
                is_terminal = 1.0 - current_done
            else:
                is_terminal = 1.0 - dones[timestep + 1]
                t_1_state_values = state_values[timestep + 1]
            delta_t = rewards[timestep] + is_terminal * GAMMA * t_1_state_values - state_values[timestep]
            current_gae = delta_t + is_terminal * GAMMA * LAMBDA * current_gae
            gae[timestep] = current_gae
        returns = gae + state_values

    ## --- Flatten buffers --- ##
    b_obs          = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs     = logprobs.reshape(-1)
    b_actions      = actions.reshape((-1,) + envs.single_action_space.shape)
    b_gae          = gae.reshape(-1)
    b_returns      = returns.reshape(-1)
    b_state_values = state_values.reshape(-1)

    ## --- PPO update --- ##
    b_inds     = np.arange(BATCH_SIZE)
    clipfracs  = []
    iter_clip_losses  = []
    iter_value_losses = []
    iter_entropies    = []
    iter_kls          = []

    for epoch in range(NUMBER_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end    = start + MINIBATCH_SIZE
            mb_inds = b_inds[start:end]

            newlogprob = actor.get_logprob(b_obs[mb_inds], b_actions.long()[mb_inds])
            entropy    = actor.get_entropy(b_obs[mb_inds])
            newvalue   = critic(b_obs[mb_inds]).flatten()

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio    = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl     = ((ratio - 1) - logratio).mean()
                clipfracs    += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

            mb_advantages = b_gae[mb_inds]
            if NORM_ADV:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            clip_loss1 = -mb_advantages * ratio
            clip_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            clip_loss  = torch.max(clip_loss1, clip_loss2).mean()

            # Value loss
            v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            # Total loss
            loss = clip_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                MAX_GRAD_NORM
            )
            optimizer.step()

            iter_clip_losses.append(clip_loss.item())
            iter_value_losses.append(v_loss.item())
            iter_entropies.append(entropy_loss.item())
            iter_kls.append(approx_kl.item())

    ## --- Logging --- ##
    y_pred, y_true = b_state_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    history["iterations"].append(iteration)
    history["clip_loss"].append(np.mean(iter_clip_losses))
    history["value_loss"].append(np.mean(iter_value_losses))
    history["entropy_loss"].append(np.mean(iter_entropies))
    history["approx_kl"].append(np.mean(iter_kls))
    history["explained_var"].append(explained_var)

    print(
        f"[iter {iteration:4d}] "
        f"lr={optimizer.param_groups[0]['lr']:.2e}  "
        f"clip_loss={history['clip_loss'][-1]:+.4f}  "
        f"v_loss={history['value_loss'][-1]:.4f}  "
        f"entropy={history['entropy_loss'][-1]:.4f}  "
        f"kl={history['approx_kl'][-1]:.4f}  "
        f"exp_var={explained_var:.3f}  "
        f"clipfrac={np.mean(clipfracs):.3f}"
    )

    # Update plots every 5 iterations
    if iteration % 5 == 0:
        update_plots()

# Final plot update and save
update_plots()
plt.ioff()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
print("Training complete. Plot saved to training_curves.png")
plt.show()

envs.close()
