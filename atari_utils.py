import numpy as np
import gym
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.int)


# class FireResetEnv(gym.Wrapper): ## not used
#     def __init__(self, env=None):
#         super(FireResetEnv, self).__init__(env)
#         # assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
#         # assert len(env.unwrapped.get_action_meanings()) >= 3

#     def step(self, action):
#         return self.env.step(action)

#     def reset(self):
#         self.env.reset()
#         obs, _, done, _ = self.env.step(1)
#         if done:
#             self.env.reset()
#         obs, _, done, _ = self.env.step(2)
#         if done:
#             self.env.reset()
#         return obs

# class MaxAndSkipEnv(gym.Wrapper):
#     def __init__(self, env=None, skip=4):
#         super(MaxAndSkipEnv, self).__init__(env)
#         # most recent raw observations (for max pooling across time steps)
#         self._obs_buffer = deque(maxlen=2)
#         self._skip = skip

#     def step(self, action):
#         total_reward = 0.0
#         done = None
#         for _ in range(self._skip):
#             obs, reward, done, info = self.env.step(action)
#             self._obs_buffer.append(obs)
#             total_reward += reward
#             if done:
#                 break
#         max_frame = np.max(np.stack(self._obs_buffer), axis=0)
#         return max_frame, total_reward, done, info

#     def reset(self):
#         self._obs_buffer.clear()
#         obs = self.env.reset()
#         self._obs_buffer.append(obs)
#         return obs

# class BufferWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_steps, dtype=np.int):
#         super(BufferWrapper, self).__init__(env)
#         self.dtype = dtype
#         old_space = env.observation_space
#         self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
#                                                 old_space.high.repeat(n_steps, axis=0), dtype=dtype)

#     def reset(self):
#         self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
#         return self.observation(self.env.reset())

#     def observation(self, observation):
#         # self.buffer[:-1] = self.buffer[1:] # not need, as here the job is to changes the shape of the observation from HWC (height, width, channel) to the CHW (channel, height, width). see https://towardsdatascience.com/deep-q-network-dqn-i-bce08bdf2af
#         print(f"shape of buffer = {np.shape(self.buffer)}, shape of buffer[0] = {np.shape(self.buffer[0])} and shape of obs = {np.shape(observation)}")
#         self.buffer[0] = observation
#         return self.buffer

# class ProcessFrame84(gym.ObservationWrapper): ## not used
#     def __init__(self, env=None):
#         super(ProcessFrame84, self).__init__(env)
#         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.int)

#     def observation(self, obs):
#         return ProcessFrame84.process(obs)

#     @staticmethod
#     def process(frame):
#         if frame.size == 210 * 160 * 3:
#             img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
#         elif frame.size == 250 * 160 * 3:
#             img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
#         else:
#             assert False, "Unknown resolution."
#         img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
#         resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
#         x_t = resized_screen[18:102, :]
#         x_t = np.reshape(x_t, [84, 84, 1])
#         return x_t.astype(np.uint8)




# class ImageToPyTorch(gym.ObservationWrapper): ## not used
#     def __init__(self, env):
#         super(ImageToPyTorch, self).__init__(env)
#         old_shape = self.observation_space.shape
#         self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], 
#                                 old_shape[0], old_shape[1]), dtype=np.int)

#     def observation(self, observation):
#         return np.moveaxis(observation, 2, 0)

