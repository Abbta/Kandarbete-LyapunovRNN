

""""
* The definitions 1) calculate lambda after each layer and pick the last one is equivalent to 2) selecting the last time step to predict lambda
* Lambda only meaningful when t --> inf. We are measuring FTLE
* Antoher plausible definition is to let the network make continuous predictions (e.g. let network prediction k --> inf) then compare
with the underperturbed input network predictions when k --> inf. Basically the same as saying we let layers --> inf since the modules adds horizontally.


* vilka plots? lambda vs mse, lambda vs ltsm?
* fixa träningsloop.

* bara pert. all input data eller pert ic och använd den nya datan??  det först undersöker bara hur systemet reagerar på störning tränat på ett dynamiskt system
behöver inte vara fysikaliskt.



"""


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import random
from scipy.integrate import solve_ivp

layers = keras.layers


# Configurations for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



# Constants 
g = 9.81      
L1 = 1.0     
L2 = 1.0     
m1 = 1.0     
m2 = 1.0     

# initial states
θi1 = np.pi / 2
ωi1 = 0.3
θi2 = np.pi / 2
ωi2 = 0.1

y0 = [θi1, ωi1, θi2, ωi2]  # initial state vector: [θi1, ωi1, θi2, ωi2]
y0_pert = [θi1, ωi1, θi2, ωi2] + np.random.normal(0, 0.5, size=4)


# ---------------------- Signal Definitions ---------------------- #
def f_sin(t, mean, sigma):
    """Sinusoidal signal with Gaussian noise."""
    input_noise = np.random.normal(loc=mean, scale=sigma, size=len(t))
    return np.sin(2 * np.pi * t), input_noise




def f_pendulum_1_pert(t, mean, sigma):
    """Angle of first pendulum with optional Gaussian noise."""
    θ1, _ = solve_double_pendulum(t, y0_pert)
    input_noise = np.random.normal(mean, sigma, size=len(t))
    return θ1, input_noise



def f_pendulum_1(t, mean, sigma):
    """Angle of first pendulum with optional Gaussian noise."""
    θ1, _ = solve_double_pendulum(t, y0)
    input_noise = np.random.normal(mean, sigma, size=len(t))
    return θ1, input_noise

def f_pendulum_2(t, mean, sigma):
    """Angle of second pendulum with optional Gaussian noise."""
    _, θ2 = solve_double_pendulum(t, y0)
    input_noise = np.random.normal(mean, sigma, size=len(t))
    return θ2, input_noise


# ---------------------- Solve Double Pendulum ---------------------- #
def double_pendulum_derivs(t, y):
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1

    den1 = (2 * m1 + m2 - m2 * np.cos(2 * Δ))

    dθ1 = ω1
    dθ2 = ω2

    dω1 = (
        -g * (2 * m1 + m2) * np.sin(θ1)
        - m2 * g * np.sin(θ1 - 2 * θ2)
        - 2 * np.sin(Δ) * m2 * (
            ω2**2 * L2 + ω1**2 * L1 * np.cos(Δ)
        )
    ) / (L1 * den1)

    dω2 = (
        2 * np.sin(Δ) * (
            ω1**2 * L1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(θ1)
            + ω2**2 * L2 * m2 * np.cos(Δ)
        )
    ) / (L2 * den1)

    return [dθ1, dω1, dθ2, dω2]


def solve_double_pendulum(t, y0):
    """Solves the double pendulum ODE for given time array and initial state."""
    sol = solve_ivp(double_pendulum_derivs, (t[0], t[-1]), y0, t_eval=t, method='DOP853')
    return sol.y[0], sol.y[2]  # theta1(t), theta2(t)



# ---------------------- Lyapunov Calculator ---------------------- #
def calc_lyapunov(l, input_noise_norm, output_noise_norm):
    """Estimate Lyapunov exponent from input/output noise norms. l here is the network layer index."""
    return np.log(output_noise_norm / input_noise_norm) / l


def compute_lyapunov_exponent(t, end):
    """
    Estimate the largest Lyapunov exponent from two nearby initial conditions.
    """

    pert = np.random.normal(0, 0.01, size=4) # perturbed initial parameters
    
    # Integrate both trajectories
    sol1 = solve_ivp(double_pendulum_derivs, (0, end), y0, t_eval=t, method='DOP853')
    sol2 = solve_ivp(double_pendulum_derivs, (0, end), y0 + pert, t_eval=t, method='DOP853')

    delta = np.linalg.norm(sol2.y - sol1.y, axis=0) # Compute the distance between the two trajectories at each time step
    delta_initial = np.linalg.norm(pert) # Initial perturbation magnitude
    lyap = np.log(delta / delta_initial) # Lyapunov exponent estimate as a function of time

    return lyap / t # return computed lyapunov per time step


# ---------------------- Data Generation ---------------------- #
def get_data(f, mean, sigma, N, window_size, end):
    """Generate data windows for training. We only train on non-perturbed system."""
    n = N - window_size - 1
    t = np.linspace(0, end, N)
    func, input_noise = f(t, mean, sigma) 
    func_pert = func + input_noise # only used for inputting to the trained model 

    data = np.stack([func[i: i + window_size + 1] for i in range(n)]) # used to train the model (dont train in perturbed data)

    X, y = np.split(data, [-1], axis=1)
    X = X[..., np.newaxis]  # Add channel dimension for Keras

    print(f"Example input shape: {X[0].shape}")
    print(f"Example input-output pair:\n{X[0].squeeze()} -> {y[0]}\n")

    return X, y, func, func_pert, t, input_noise


# ---------------------- Model ---------------------- #
def define_model():
    """Define and compile the LSTM model."""
    z0 = layers.Input(shape=[None, 1])
    # z = layers.LSTM(16)(z0)
    # z = layers.Dense(1)(z)

    # Stacked LSTMs with Dropout
    z = layers.LSTM(128, return_sequences=True)(z0)

    z = layers.LSTM(32, return_sequences=True)(z)

    z = layers.LSTM(16)(z)

    # Dense layers with Dropout
    z = layers.Dense(32, activation='relu')(z)
    # z = layers.Dropout(0.2)(z)

    z = layers.Dense(16, activation='relu')(z)


    z = layers.Dense(1)(z)

    model = keras.models.Model(inputs=z0, outputs=z)
    model.compile(loss='mse', optimizer='adam')

    return model


# ---------------------- Training ---------------------- #
def train_model(model, X, y, epochs, batch_size):
    """Train model with early stopping and LR scheduler."""
    results = model.fit(X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=3, verbose=1, min_lr=1E-5),
            keras.callbacks.EarlyStopping(patience=4, verbose=1)
        ])
    return results


def plot_training(results):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 4))
    plt.plot(results.history['loss'], label='Train Loss')
    plt.plot(results.history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('Training Progress')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


# ---------------------- Prediction ---------------------- #
def predict_next_k(model, input_window, k):
    """Predict next k time steps using rolling window."""
    x = input_window[np.newaxis, :, np.newaxis]
    y = np.zeros(k)
    for i in range(k):
        y[i] = model.predict(x, verbose=0)
        x = np.roll(x, -1, axis=1)
        x[:, -1] = y[i]
    return y

# ---------------------- Plot ---------------------- #
def plot_prediction(window_size, func, t, i0, k, label='Clean', plot=False):
    """Predict and visualize sequence continuation."""
    input_window = func[i0: i0 + window_size]
    y_pred = predict_next_k(model, input_window, k)

    t_input = t[i0: i0 + window_size]
    t_pred = t[i0 + window_size: i0 + window_size + k]

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(t, func, label='Original Signal')
        plt.plot(t_input, input_window, color='C1', lw=3, label=f'{label} Prediction Input')
        plt.plot(t_pred, y_pred, color='C1', ls='--', label=f'{label} Predicted Output')
        plt.xlim(0, 10)
        plt.xlabel('Time $t$')
        plt.ylabel('Signal $f(t)$')
        plt.title(f'{label} Prediction')
        plt.grid(True)
        plt.legend()
        plt.show()

    return y_pred



def plot_lyapunov(t, lyapunov_exact):
    plt.figure(figsize=(10, 4))
    plt.plot(t, lyapunov_exact, label='FTLE (Numeric)')
    plt.xlabel('Time [s]')
    plt.ylabel(r'$\lambda(t)$')
    plt.title('Lyapunov Exponent Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------- Run Everything ---------------------- #
# ------------------------------------------------------------ #

# Parameters
plot = False

f = f_pendulum_2


N = 1000
window_size = 15
end = 50
mean = 0
sigma = 0.2

epochs = 20
batch_size = 32

i0 = 0 # must be 0 for exact lyapunov prediction to make sense
k = 300

# Prepare data
X, y, func, func_pert, t, input_noise = get_data(f, mean, sigma, N, window_size, end)
input_nose = input_noise[i0: i0 + window_size] # extract only the noise from input to prediction

# Train model
model = define_model()
results = train_model(model, X, y, epochs, batch_size)
#plot_training(results)

# Predict
output_clean = plot_prediction(window_size, func, t, i0, k, 'Clean', plot) # Clean input 
output_noisy = plot_prediction(window_size, func_pert, t, i0, k, 'Noisy', plot) # Noisy input



# f = f_pendulum_1_pert
# X, y, func, func_pert, t, input_noise = get_data(f, mean, sigma, N, window_size, end)
# output_noisy = plot_prediction(window_size, func, t, i0, k, 'Noisy', plot) # Noisy input




# Lyapunov calculations
output_noise = output_noisy - output_clean
input_noise_norm = np.linalg.norm(input_noise) / len(input_noise)     # average norm
output_noise_norm = np.linalg.norm(output_noise) / len(output_noise)  # average norm

lyapunov_estimate = calc_lyapunov(k, input_noise_norm, output_noise_norm) # l = k
lyapunov_exact = compute_lyapunov_exponent(t, end) 

print(f"Finite Time Lyapunov Exponent Estimate: {round(lyapunov_estimate, 3)} s⁻¹")

t_star = k

compare_index = t_star # agree at inf?
print(f"Finite Time Lyapunov Exponent Numeric: {round(lyapunov_exact[compare_index], 3)} s⁻¹")
plot_lyapunov(t, lyapunov_exact)


# ------------------------------------------------------------ #
# ------------------------------------------------------------ #









