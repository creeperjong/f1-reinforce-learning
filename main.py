import gym
import numpy as np
import argparse
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

class F110ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_beams = env.sim.agents[0].num_beams
        self.num_agents = env.num_agents

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_beams + 2,),
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        return obs, reward, done, _
    
    def reset(self, **kwargs):
        poses = kwargs.pop("poses", None)
        if poses is None:
            x0 = self.env.start_xs[0]
            y0 = self.env.start_ys[0]
            th0 = self.env.start_thetas[0]
            poses = np.array([[x0, y0, th0]], dtype=np.float32)
        obs, _, _, _ = self.env.reset(poses=poses)
        return self.observation(obs)

    def observation(self, obs):
        scan = obs["scans"][0]
        v_x = obs["linear_vels_x"][0:1]
        yaw = obs["ang_vels_z"][0:1]
        out = np.concatenate([scan, v_x, yaw]).astype(np.float32)

        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite observation")

        return out

class F110ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.s_min = env.params['s_min']
        self.s_max = env.params['s_max']
        self.v_min = env.params['v_min']
        self.v_max = env.params['v_max']

        self.num_agents = env.num_agents

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, 2),
            dtype=np.float32
        )

    def action(self, action):
        a = np.clip(action, -1.0, 1.0)

        steer = 0.5 * (a[..., 0] + 1.0) * (self.s_max - self.s_min) + self.s_min
        speed = 0.5 * (a[..., 1] + 1.0) * (self.v_max - self.v_min) + self.v_min

        return np.stack([steer, speed], axis=-1)

class F110RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_steer = 0.0
        self.prev_lap = 0

    def reset(self, **kwargs):
        self.prev_steer = 0.0
        self.prev_lap = 0
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # === Extract info ===
        scan = obs["scans"][0]
        vx = obs["linear_vels_x"][0]
        yaw_rate = obs["ang_vels_z"][0]

        # left/right lidar
        left = np.mean(scan[:200])
        right = np.mean(scan[-200:])
        front = np.mean(scan[440:640])

        # progress (simple version)
        progress_reward = vx * 0.05

        # lateral error (left-right imbalance)
        lateral_error = left - right
        lateral_penalty = abs(lateral_error) * 0.01

        # yaw penalty
        yaw_penalty = abs(yaw_rate) * 0.02

        # steering smoothness
        steer = action[0][0] if action.ndim == 2 else action[0]
        steer_penalty = abs(steer - self.prev_steer) * 0.1
        self.prev_steer = steer

        # curvature-aware safe speed
        safe_speed = np.clip(front * 0.5, 1.0, 6.0)
        speed_penalty = max(0, vx - safe_speed) * 0.1

        # collision penalty (additive)
        collision_penalty = 20.0 if np.any(obs["collisions"]) else 0.0

        # final reward
        shaped = (
            progress_reward
            - lateral_penalty
            - yaw_penalty
            - steer_penalty
            - speed_penalty
            - collision_penalty
        )

        final_reward = shaped

        processed = self.env.observation(obs)

        return processed, final_reward, done, info

def make_env(seed=0):
    def _init():
        env = gym.make("f110_gym:f110-v0", map="vegas", num_agents=1, seed=seed).unwrapped
        env = F110ObsWrapper(env)
        env = F110RewardWrapper(env)
        env = F110ActionWrapper(env)
        return env
    set_random_seed(seed)
    return _init

def train_ppo(total_timesteps=200000, model_path="ppo_f1tenth_vegas"):
    env = DummyVecEnv([make_env()])

    model = PPO("MlpPolicy", env, verbose=0)

    n_chunks = 100
    steps_per_chunk = total_timesteps // n_chunks

    print(f"Start PPO training for {total_timesteps} steps...")
    for _ in trange(n_chunks):
        model.learn(total_timesteps=steps_per_chunk, reset_num_timesteps=False)

    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    env.close()
    return f"{model_path}.zip"

def run_with_render(model_path, max_steps=50000):
    env = gym.make(
        "f110_gym:f110-v0",
        map="vegas",
        num_agents=1,
        seed=123,
        render_mode="human"
    ).unwrapped
    wrapped = F110ObsWrapper(env)
    wrapped = F110RewardWrapper(wrapped)
    wrapped = F110ActionWrapper(wrapped)

    model = PPO.load(model_path)

    obs = wrapped.reset()
    done = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = wrapped.step(action)

        env.render()

        if done:
            obs = wrapped.reset()

    env.close()

def debug_env():
    env_fn = make_env()
    env = env_fn()
    obs = env.reset()
    print("reset obs finite?", np.all(np.isfinite(obs)))

    for t in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if not np.all(np.isfinite(obs)):
            print(f"Non-finite obs at step {t}")
            print("action:", action)
            print("obs:", obs)
            break
        if done:
            obs = env.reset()

def evaluate_with_render(model_path, max_steps=5000):

    model = PPO.load(model_path)

    # 建立 env（使用 human render）
    env = gym.make(
        "f110_gym:f110-v0",
        map="vegas",
        num_agents=1,
        seed=999,
        render_mode="human"
    ).unwrapped

    wrapped = F110ObsWrapper(wrapped)
    wrapped = F110RewardWrapper(wrapped)
    wrapped = F110ActionWrapper(env)

    obs = wrapped.reset()
    done = False

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = wrapped.step(action)
        env.render()

        if done:
            obs = wrapped.reset()

    env.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="", help="Path to the trained PPO model.")
    return parser.parse_args()

# ================================
# Main
# ================================
if __name__ == "__main__":
    args = parse_args()
    if args.model_path:
        run_with_render(args.model_path, max_steps=10000)
        exit(0)
    else:
        model_path = train_ppo(
            total_timesteps=200000,
            model_path="ppo_f1tenth_vegas"
        )

        run_with_render(model_path, max_steps=10000)
