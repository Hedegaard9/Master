import numpy as np
import ewma

data = np.array([0.1, 0.2, 0.3, 0.4, np.nan, 0.5, 0.6])
lambda_val = 0.94
start = 3

result = ewma.ewma_c(data, lambda_val, start)
print(result)
