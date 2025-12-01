import gym
import numpy as np
import argparse
from tqdm import trange

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

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

        scan = obs["scans"][0]
        vx = obs["linear_vels_x"][0]
        vy = obs["linear_vels_y"][0]
        yaw_rate = obs["ang_vels_z"][0]

        # === 1. Forward reward ===
        forward_reward = vx * 0.3

        # === 2. Collision penalty ===
        if np.any(obs["collisions"]):
            return self.env.observation(obs), -200.0, True, info

        # === 3. Slip penalty (use vy only, not yaw_rate) ===
        slip_penalty = -abs(vy) * 1.8

        # === 4. Better curve detection using front beam shrinking ===
        front_min = np.min(scan[400:680])  # center front
        straight_reference = 7.0           # typical long straight front distance

        # smaller front_min → curve approaching
        curve_strength = np.clip((straight_reference - front_min), 0, 4.0)

        # steering-dependent safe speed (more realistic)
        steer = action[0][0] if action.ndim == 2 else action[0]
        steer_actual = steer * 0.5  # scale to realistic steering

        base_speed_limit = 5.5
        steer_limit = base_speed_limit / (1.0 + 3.0 * abs(steer_actual))

        curve_speed_limit = max(1.5, steer_limit - 0.3 * curve_strength)

        # speed penalty
        if vx > curve_speed_limit:
            curve_penalty = -(vx - curve_speed_limit) * 0.5
        else:
            curve_penalty = +(curve_speed_limit - vx) * 0.1

        # === 5. Steering smoothness (weaken to avoid blocking turning) ===
        smooth_penalty = -abs(steer - self.prev_steer) * 0.1
        self.prev_steer = steer

        # === 6. Remove old apex reward (corrupted by wall geometry) ===
        apex_reward = 0.0

        # === Final reward ===
        final_reward = (
            forward_reward +
            slip_penalty +
            curve_penalty +
            smooth_penalty +
            apex_reward
        )

        return self.env.observation(obs), final_reward, done, info



def make_env(rank, seed=0):
    def _init():
        env = gym.make(
            "f110_gym:f110-v0",
            map="vegas",
            num_agents=1,
            seed=seed + rank
        ).unwrapped

        env = F110ObsWrapper(env)
        env = F110RewardWrapper(env)
        env = F110ActionWrapper(env)

        return env

    return _init

def train_ppo(total_timesteps=2e5, model_path="ppo_f1tenth_vegas"):
    NUM_ENVS = 12
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=6144,
        verbose=1,
    )

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
        run_with_render(args.model_path)
        exit(0)
    else:
        model_path = train_ppo(
            model_path="ppo_f1tenth_vegas"
        )
        run_with_render(model_path)