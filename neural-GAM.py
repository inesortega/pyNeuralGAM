

# Generate training data --  try to approximate f(x) = x^2:
x1 = -50 + np.random.random((25000,1))*100
x2 = 50 + np.random.random((25000,1))*140
x3 = 25 + np.random.random((25000,1))*50

y = x1**x1 + 2*x2 + x3