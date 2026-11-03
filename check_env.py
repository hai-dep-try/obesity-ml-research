import numpy, pandas, sklearn, skopt, tensorflow as tf
from skopt import gp_minimize

print("numpy", numpy.__version__, "| sklearn", sklearn.__version__,
      "| skopt", skopt.__version__, "| tf", tf.__version__)

# Thử gp_minimize trên hàm đơn giản
r = gp_minimize(lambda x: (x[0] - 2) ** 2, [(-5.0, 5.0)],
                n_calls=12, acq_func="EI", random_state=42)
print("gp_minimize OK, x* ≈", round(r.x[0], 2))

# Thử Keras
m = tf.keras.Sequential([tf.keras.layers.Input((5,)),
                         tf.keras.layers.Dense(3, activation="softmax")])
print("Keras OK, params:", m.count_params())