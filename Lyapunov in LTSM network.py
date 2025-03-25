


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt
layers = keras.layers



def f_sin(t, mean, sigma):
    input_noice = np.random.normal(loc=mean, scale=sigma, size=len(t))
    return np.sin(2*np.pi * t), input_noice


def f_pendulum(t, theta0=np.pi/4, omega0=0.1, L=1.0, g=9.81):
    """
    Returns the analytical solution to the small-angle pendulum.
    
    Parameters:
    - t: time array
    - theta0: initial angle (radians)
    - omega0: initial angular velocity
    - L: length of pendulum (meters)
    - g: gravitational acceleration (m/s^2)

    Returns:
    - theta: angle over time (radians)
    """
    const = np.sqrt(g / L)
    theta = theta0 * np.cos(const * t) 
    return theta


def calc_lyapunov(l, input_noice_norm, output_noice_avg_norm):
    return np.log(output_noice_avg_norm/input_noice_avg_norm)/l




def get_data(f, mean, sigma, N,window_size, end):
    n = N - window_size - 1


    t = np.linspace(0, end, N)  # time steps
    func, input_noice = f(t, mean, sigma)  # signal
    func_pert = func + input_noice

    # Now we stack training point consisting of window_size+1 inputs.
    data = np.stack([func[i: i + window_size + 1] for i in range(n)]) 
    data_pert = np.stack([func_pert[i: i + window_size + 1] for i in range(n)]) 

    X, y = np.split(data, [-1], axis=1)

    
    # as always, another dimension is added to the input vector
    # because the KERAS library also allows for multiple inputs per time step.
    # In our case here, we have just one input value per time stamp.
    X = X[..., np.newaxis]

    print(f'Example input-output pair: {X[0]} and {y[0]}')

    return X, y, func, func_pert, t, input_noice


def define_model():
    # define sequential model with one output (next time step)
    z0 = layers.Input(shape=[None, 1])
    z = layers.LSTM(16)(z0)
    z = layers.Dense(1)(z)
    model = keras.models.Model(inputs=z0, outputs=z)
    #print(model.summary())

    model.compile(loss='mse', optimizer='adam')

    return model

def train_model(model, X, y, epochs, batch_size):

    results = model.fit(X, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(factor=0.67, patience=3, verbose=1, min_lr=1E-5),
            keras.callbacks.EarlyStopping(patience=4, verbose=1)])

    return results


def plot_training(results):
    plt.figure(1, (12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.ylabel('loss')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.tight_layout()
    plt.show()


def predict_next_k(model, input_window, k):
    """Predict next k steps ITERATIVELY for the given model and starting sequence """
    x = input_window[np.newaxis, :, np.newaxis]  # initial input
    y = np.zeros(k)
    for i in range(k):
        y[i] = model.predict(x, verbose=1)
        # create the new input including the last prediction
        x = np.roll(x, -1, axis=1)  # shift all inputs 1 step to the left
        x[:, -1] = y[i]  # add latest prediction to end
    return y


def plot_prediction(window_size, func, t, i0, k):
    """ Predict and plot the next k steps for an input starting at i0 """
    input_window = func[i0: i0 + window_size]  # starting window (input) to 
    y1 = predict_next_k(model, input_window, k)  # predict next k steps

    t0 = t[i0: i0 + window_size]
    t1 = t[i0 + window_size: i0 + window_size + k]


    plt.figure(figsize=(12, 4))
    plt.plot(t, func, label='data')
    plt.plot(t0, input_window, color='C1', lw=3, label='prediction')
    plt.plot(t1, y1, color='C1', ls='--')
    plt.xlim(0, 10)
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$f(t)$')
    plt.show()

    return y1


f = f_sin

N = 1000
window_size = 60
end = 10
mean = 0
sigma = 0.5


X, y, func, func_pert, t, input_noice = get_data(f, mean, sigma, N, window_size, end)
print("input noice", input_noice)
model = define_model()


epochs = 20
batch_size = 32

results = train_model(model, X, y, epochs, batch_size)
plot_training(results)



i0 = 60 # starting points for prediction
k = 100 # number of time steps to predict
output = plot_prediction(window_size, func, t, i0, k)
output_pert = plot_prediction(window_size, func_pert, t, i0, k)

output_noice = output_pert - output


input_noice_avg_norm = np.linalg.norm(input_noice)/window_size
output_noice_avg_norm = np.linalg.norm(output_noice)/k



exp = calc_lyapunov(window_size, input_noice_avg_norm, output_noice_avg_norm)
print(f"Lyapunov exponent: {exp}")









