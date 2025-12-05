import numpy as np

def sample_u_power(n, k=2, rng=None):
    """
    Sample n values on (0,10) from f(x) ∝ |x-5|^k using rejection sampling.
    k controls how deep the middle valley is (higher k -> lower center prob).
    """
    samples = []
    # envelope max f occurs at edges: |x-5| = 5, so f_max = 5**k (we use ratio f(x)/f_max)
    while len(samples) < n:
        x = rng.random() * 10.0            # candidate from Uniform(0,10)
        u = rng.random()
        accept_prob = (abs(x - 5.0) / 5.0) ** k
        if u < accept_prob:
            samples.append(x)
    return np.array(samples)

def generate_data_luben(n, rng):
    # Proportional number of points for dense and sparse regions
    num_dense = round(0.425 * n)  # half of the data points go to dense regions
    num_middle = round(
        0.15 * n
    )  # a small fraction of data points for the middle region

    # Generate x values for dense and sparse regions
    x_dense1 = rng.uniform(0, 1.5, num_dense)
    x_dense2 = rng.uniform(8, 10, num_dense)
    # using beta
    x_middle = (rng.beta(8, 8, num_middle) * (8 - 1.5)) + 1.5
    x_sparse = np.concatenate([x_dense1, x_dense2, x_middle])

    # True function to generate y based on x
    def true_function(x):
        y = 2 * np.sin(x) + rng.normal(0, 0.1, len(x))
        mask = (2 < x) & (x < 7.5)
        y[mask] += rng.normal(0, 2, np.sum(mask))
        return y

    # Generate y values
    y = true_function(x_sparse)

    # Return as a data frame
    return {"x": x_sparse, "y": y}

def generate_data(n, type='linear', heteroscedastic=False, epistemic_uncertainty=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    if type=='luben':
        return generate_data_luben(n, rng)
    
    if type=='butterfly':
        x = rng.normal(5, 2, 1000)
        y = [rng.normal(0, np.abs(i-5)) for i in x]
        return {'x':x, 'y':y}
    
    if epistemic_uncertainty:
        x = sample_u_power(n, 2, rng)
    else:
        x = rng.uniform(0, 10, n)
        
    if type=='quadratic':
        y = [i**2 - 10*i + 25 for i in x]
    elif type=='logistic':
        y = [20 / (1+np.exp(-(i-5))) for i in x]
    elif type=='linear':
        y = x
    
    if heteroscedastic:
        y = [i + rng.normal(0, i/4) for i in y]
    else:
        y = [i + rng.normal(0, 1) for i in y]
    
    return {'x':x, 'y':y}