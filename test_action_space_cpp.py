import numpy as np
from stratego.cpp import stratego_cpp as sp
from stratego.wrappers.cpp_masked_multi_descrete import MaskedMultiDiscreteCpp

action_space = MaskedMultiDiscreteCpp(sp.MaskedMultiDiscrete([10, 10]))

print(action_space.mask)
print(action_space.sample())
print(action_space.sample())
print(action_space.sample())
print(action_space.sample())

# action_space.set_mask([False] * 60 + [True] * 40)
action_mask = np.zeros((10, 10), dtype=bool)
action_mask[6:] = True
action_space.set_mask(action_mask)

print(action_space.mask)
print(action_space.sample())
print(action_space.sample())
print(action_space.sample())
print(action_space.sample())
