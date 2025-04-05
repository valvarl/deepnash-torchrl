import numpy as np
import stratego_cpp  # Скомпилированный модуль

class StrategoEnvWrapper:
    def __init__(self, config=None):
        self._cpp_env = stratego_cpp.StrategoEnv(config)
        
    def reset(self, seed=None):
        obs = self._cpp_env.reset(seed or 0)
        return self._convert_observation(obs)
    
    def step(self, action):
        obs, reward, terminated, truncated = self._cpp_env.step(action)
        return self._convert_observation(obs), reward, terminated, truncated
    
    def _convert_observation(self, cpp_obs):
        # Конвертация C++ observation в numpy array
        return np.array(cpp_obs, dtype=np.float32)
    
    @property
    def action_space(self):
        # Возвращаем пространство действий
        ...
    
    @property 
    def observation_space(self):
        # Возвращаем пространство наблюдений
        ...