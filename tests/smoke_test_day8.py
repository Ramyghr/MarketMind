import sys
sys.path.insert(0, 'src')
import numpy as np
from ssl_model.augmentations import jitter, scaling, window_slice, augment

x = np.random.randn(60, 5).astype(np.float32)

for fn in [jitter, scaling, window_slice]:
    out = fn(x)
    assert out.shape == (60, 5), f'{fn.__name__} shape mismatch'
    print(f'{fn.__name__}: OK')

out = augment(x)
assert out.shape == (60, 5)
print('augment: OK')