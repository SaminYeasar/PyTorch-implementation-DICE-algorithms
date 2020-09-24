
"""Implementation of AlgaeDICE.

Based on the publication "AlgaeDICE: Policy Gradient from Arbitrary Experience"
by Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, Dale Schuurmans.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import torch
import keras_utils as keras_utils
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import distributions as pyd

ds = tfp.distributions



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def torch_to_tf_tensor(x):
    return tf.convert_to_tensor(x.detach().cpu().numpy())


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=False)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=False)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1


    def __init__(self, cache_size=1):
        super(TanhTransform,self).__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super(SquashedNormal, self).__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

# source https://github.com/kevinzakka/pytorch-goodies
def orthogonal_regularization(model, device):
    with torch.enable_grad():
        reg = 1e-6
        orth_loss = torch.zeros(1).to(device)
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0]).to(device)
                orth_loss = orth_loss + (reg * sym.abs().sum())
    return orth_loss

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_range, log_std_bounds=[-5, 2]):
        super(Actor, self).__init__()

        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(state_dim, 256, 2 * action_dim, hidden_depth=2)
        self.action_range = action_range

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist





class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.Q1 = mlp(state_dim + action_dim,
                      256, 1, 2)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)

        self.outputs['q1'] = q1

        return q1

class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DoubleCritic, self).__init__()

        self.Q1 = mlp(state_dim + action_dim,
                      256, 1, 2)
        self.Q2 = mlp(state_dim + action_dim,
                      256, 1, 2)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2





class ALGAE(object):
    """Class performing algae training."""

    def __init__(self,
                 state_dim,
                 action_dim,
                 action_range,
                 log_interval,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 alpha_init=1.0,
                 learn_alpha=True,
                 algae_alpha=1.0,
                 use_dqn=True,
                 use_init_states=True,
                 exponent=2.0):
        """Creates networks.

        Args:
          state_dim: State size.
          action_dim: Action size.
          log_interval: Log losses every N steps.
          actor_lr: Actor learning rate.
          critic_lr: Critic learning rate.
          alpha_init: Initial temperature value for causal entropy regularization.
          learn_alpha: Whether to learn alpha or not.
          algae_alpha: Algae regularization weight.
          use_dqn: Whether to use double networks for target value.
          use_init_states: Whether to use initial states in objective.
          exponent: Exponent p of function f(x) = |x|^p / p.
        """
        self.action_range = action_range
        self.actor = Actor(state_dim, action_dim, action_range).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.use_init_states = use_init_states

        if use_dqn:
            self.critic = DoubleCritic(state_dim, action_dim).to(device)
            self.critic_target = DoubleCritic(state_dim, action_dim).to(device)
        else:
            self.critic = Critic(state_dim, action_dim).to(device)
            self.critic_target = Critic(state_dim, action_dim).to(device)
        soft_update_params(self.critic, self.critic_target, tau=1.0)

        self._lambda = torch.tensor(0.0).to(device)
        self._lambda.requires_grad = True
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        initial_temperature = alpha_init
        self.log_alpha = torch.tensor(np.log(initial_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.learn_alpha = learn_alpha
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])

        self.log_interval = log_interval
        self.algae_alpha = algae_alpha
        self.use_dqn = use_dqn
        self.exponent = exponent
        self.device=device


        if self.exponent <= 1:
            raise ValueError('Exponent must be greather than 1, but received %f.' %
                           self.exponent)
        self.f = lambda resid: torch.pow(torch.abs(resid), self.exponent) / self.exponent
        clip_resid = lambda resid: torch.clamp(resid, 0.0, 1e6)
        self.fgrad = lambda resid: torch.pow(clip_resid(resid), self.exponent - 1)


        # save

        self.avg_actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
        self.avg_alpha_loss = tf.keras.metrics.Mean('alpha_loss', dtype=tf.float32)
        self.avg_actor_entropy = tf.keras.metrics.Mean('actor_entropy', dtype=tf.float32)
        self.avg_alpha = tf.keras.metrics.Mean('alpha', dtype=tf.float32)
        self.avg_lambda = tf.keras.metrics.Mean('lambda', dtype=tf.float32)
        self.avg_critic_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
        self.training = True
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)


    @property
    def alpha(self):
        return self.log_alpha.exp()


    def sample(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(device)
        obs = obs.unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample() if sample else dist.mean
            action = action.clamp(*self.action_range)
            assert action.ndim == 2 and action.shape[0] == 1
        return to_np(action[0])


    def critic_mix(self, s, a):
        if self.use_dqn:
            target_q1, target_q2 = self.critic_target(s, a)
            target_q = torch.min(target_q1, target_q2)
            q1, q2 = self.critic(s, a)
            return (q1 * 0.05 + target_q * 0.95), (q2 * 0.05 + target_q * 0.95)
        else:
            return (self.critic(s, a) * 0.05 + self.critic_target(s, a) * 0.95)

    def fit_critic(self, states, actions, next_states, rewards, masks, discount,
                   init_states):
        """Updates critic parameters.

        Args:
          states: A batch of states.
          actions: A batch of actions.
          next_states: A batch of next states.
          rewards: A batch of rewards.
          masks: A batch of masks indicating the end of the episodes.
          discount: An MDP discount factor.
          init_states: A batch of init states from the MDP.

        Returns:
          Critic loss.
        """
        with torch.no_grad():
            init_actions, _ = self.sample(init_states)
            next_actions, next_log_probs = self.sample(next_states)

        # ========== for double Q case ========== #

        if self.use_dqn:


            with torch.no_grad():
                target_q1, target_q2 = self.critic_mix(next_states, next_actions)
                #target_q1, target_q2 = self.critic_target(next_states, next_actions)
                target_q1 = target_q1 - self.alpha * next_log_probs
                target_q2 = target_q2 - self.alpha * next_log_probs

                target_q1 = (rewards + discount * masks * target_q1)
                target_q2 = (rewards + discount * masks * target_q2)

            q1, q2 = self.critic(states, actions)
            init_q1, init_q2 = self.critic(init_states, init_actions)

            if discount == 1:
                critic_loss1 = torch.mean(self.f(self._lambda + self.algae_alpha + target_q1 - q1) - self.algae_alpha * self._lambda)
                critic_loss2 = torch.mean(self.f(self._lambda + self.algae_alpha + target_q2 - q2) - self.algae_alpha * self._lambda)
            else:
                critic_loss1 = torch.mean(self.f(target_q1 - q1) + (1 - discount) * init_q1 * self.algae_alpha)
                critic_loss2 = torch.mean(self.f(target_q2 - q2) + (1 - discount) * init_q2 * self.algae_alpha)

            critic_loss = (critic_loss1 + critic_loss2)


        # ============ for single Q ============ #

        else:

            with torch.no_grad():
                target_q = self.critic_mix(next_states, next_actions)
                target_q = target_q - self.alpha * next_log_probs
                target_q = (rewards + discount * masks * target_q)      # r(s,a) + \gamma ( nu(s',a') - temp * log(\pi(a'|s')) )

            q = self.critic(states, actions)                      # nu(s,a)
            init_q = self.critic(init_states, init_actions)       # nu(s_0,a_0)

            if discount == 1:
                critic_loss = torch.mean(self.f(self._lambda + self.algae_alpha + target_q - q) - self.algae_alpha * self._lambda)
            else:

                # Equation 25: 2*alpha * (1-gamma) * E [nu(s_0,a_0)] + E [ clip(bellman residuals) ]
                # self.f clip the bellman residuals \delta (explained in "continous control" section in AlgaeDICE)
                critic_loss = torch.mean(self.f(target_q - q) + (1 - discount) * init_q * self.algae_alpha)


        # TODO : add self._lambda
        # TODO tf uses following : self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.variables + [self._lambda]))
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def fit_actor(self, states, actions, next_states, rewards, masks, discount,
                  target_entropy, init_states):
        """Updates critic parameters.

        Args:
          states: A batch of states.
          actions: A batch of actions.
          next_states: A batch of next states.
          rewards: A batch of rewards.
          masks: A batch of masks indicating the end of the episodes.
          discount: An MDP discount factor.
          target_entropy: Target entropy value for alpha.
          init_states: A batch of init states from the MDP.

        Returns:
          Actor and alpha losses.
        """

        init_actions, _ = self.sample(init_states)
        next_actions, next_log_probs = self.sample(next_states)

        # ========== for double Q case ========== #

        if self.use_dqn:

            target_q1, target_q2 = self.critic_mix(next_states, next_actions)
            #with torch.no_grad():
            #target_q1, target_q2 = self.critic_target(next_states, next_actions)

            target_q1 = target_q1 - self.alpha.detach() * next_log_probs
            target_q2 = target_q2 - self.alpha.detach() * next_log_probs
            target_q1 = rewards + discount * masks * target_q1
            target_q2 = rewards + discount * masks * target_q2

            q1, q2 = self.critic(states, actions)
            init_q1, init_q2 = self.critic(init_states, init_actions)

            if discount == 1:
                actor_loss1 = -torch.mean(self.fgrad(self._lambda + self.algae_alpha + target_q1 - q1).detach() * (target_q1 - q1))
                actor_loss2 = -torch.mean(self.fgrad(self._lambda + self.algae_alpha + target_q2 - q2).detach() * (target_q2 - q2))

            else:
                actor_loss1 = -torch.mean(self.fgrad(target_q1 - q1).detach() * (target_q1 - q1) + (1 - discount) * init_q1 * self.algae_alpha)
                actor_loss2 = -torch.mean(self.fgrad(target_q2 - q2).detach() * (target_q2 - q2) + (1 - discount) * init_q2 * self.algae_alpha)

            loss = (actor_loss1 + actor_loss2) / 2.0


        # ============ for single Q ============ #

        else:

            target_q = self.critic_mix(next_states, next_actions)
            target_q = target_q - self.alpha.detach() * next_log_probs
            target_q = rewards + discount * masks * target_q

            q = self.critic(states, actions)
            init_q = self.critic(init_states, init_actions)

            if discount == 1:
                loss = -torch.mean(self.fgrad(self._lambda + self.algae_alpha + target_q - q).detach() * (target_q - q))
            else:
                # for policy training cliped (delta or residuals) must be >0 (explained in "continous control" section in AlgaeDICE)
                # This line is an application of the change rule. Specifically:
                # Derivative of 1/k * (a - b)^k = (a - b)^{k-1} * [Derivative of a - b].
                loss = -torch.mean(self.fgrad(target_q - q).detach() * (target_q - q) + (1 - discount) * init_q * self.algae_alpha)


        actor_loss = loss + orthogonal_regularization(self.actor.trunk, self.device)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        alpha_loss = torch.mean(self.alpha * (-next_log_probs.data - target_entropy))

        if self.learn_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()


        return actor_loss, alpha_loss, -next_log_probs


    def update(self,
              replay_buffer,
              total_timesteps,
              discount=0.99,
              tau=0.005,
              target_entropy=0,
              actor_update_freq=2):
        """Performs a single training step for critic and actor.

        Args:
          replay_buffer_iter: An tensorflow graph iteratable object for sampling
            transitions.
          init_replay_buffer: An tensorflow graph iteratable object for sampling
            init states.
          discount: A discount used to compute returns.
          tau: A soft updates discount.
          target_entropy: A target entropy for alpha.
          actor_update_freq: A frequency of the actor network updates.

        Returns:
          Actor and alpha losses.
        """
        states, actions, rewards, next_states, masks = replay_buffer.sample()
        init_states = states



        # TODO: IMPORTANT TO CHECK
        # if self.use_init_states:
        #   init_states = next(init_replay_buffer)[0]
        # else:
        #   init_states = states


        # ========== critic update ========== #
        critic_loss = self.fit_critic(states, actions, next_states, rewards, masks,
                                      discount, init_states)
        step = 0
        self.avg_critic_loss(torch_to_tf_tensor(critic_loss))
        if tf.equal(total_timesteps % self.log_interval, 0):
            train_measurements = [('train/critic_loss', self.avg_critic_loss.result()),]
            for (label, value) in train_measurements:
                tf.summary.scalar(label, value, step=step)
            keras_utils.my_reset_states(self.avg_critic_loss)

        # ========= actor update & critic target update ========= #
        if tf.equal(total_timesteps % actor_update_freq, 0):
            actor_loss, alpha_loss, entropy = self.fit_actor(states, actions,
                                                             next_states, rewards,
                                                             masks, discount,
                                                             target_entropy,
                                                             init_states)
            soft_update_params(self.critic, self.critic_target, tau=tau)

            self.avg_actor_loss(torch_to_tf_tensor(actor_loss))
            self.avg_alpha_loss(torch_to_tf_tensor(alpha_loss))
            self.avg_actor_entropy(torch_to_tf_tensor(entropy))
            self.avg_alpha(torch_to_tf_tensor(self.alpha))
            self.avg_lambda(torch_to_tf_tensor(self._lambda))
            if tf.equal(total_timesteps % self.log_interval, 0):

                print('critic loss : {} | actor loss : {}'.format(critic_loss.data.cpu().numpy(),actor_loss.data.cpu().numpy()))
                train_measurements = [
                    ('train/actor_loss', self.avg_actor_loss.result()),
                    ('train/alpha_loss', self.avg_alpha_loss.result()),
                    ('train/actor entropy', self.avg_actor_entropy.result()),
                    ('train/alpha', self.avg_alpha.result()),
                    ('train/lambda', self.avg_lambda.result()),
                ]
                for (label, value) in train_measurements:
                    tf.summary.scalar(label, value, step=total_timesteps)
                keras_utils.my_reset_states(self.avg_actor_loss)
                keras_utils.my_reset_states(self.avg_alpha_loss)
                keras_utils.my_reset_states(self.avg_actor_entropy)
                keras_utils.my_reset_states(self.avg_alpha)
                keras_utils.my_reset_states(self.avg_lambda)


