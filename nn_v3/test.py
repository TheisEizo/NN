

for X, Y in train:
    


#%%
import numpy as np
import matplotlib.pyplot as plt
from nn_v3_func import util

sin_wave =np.sin(np.arange(200))
plt.plot(sin_wave[:50])

X = []
y = []

seq_len = 50
num_records = len(sin_wave) - seq_len

for i in range(num_records - 50):
    X.append(sin_wave[i:i+seq_len])
    y.append(sin_wave[i+seq_len])
    
X = np.array(X)
X = np.expand_dims(X, axis=2)

y = np.array(y)
y = np.expand_dims(y, axis=1)

X_val = []
y_val = []

for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i:i+seq_len])
    y_val.append(sin_wave[i+seq_len])
    
X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

y_val = np.array(y_val)
y_val = np.expand_dims(y_val, axis=1)

min_clip_value = -10
max_clip_value = 10

#%%
from neurons import Sigmoid, Tanh, Softmax
from layers import RecurrentFullCon
from cost import CrossEntropy
from networks import FF

NN = FF(
        [RecurrentFullCon(Tanh, Softmax, (50, 100, 1), 5)],
        CrossEntropy)
NN.SGD((X,y),(X_val, y_val), epochs=25, printout='Loss')

#%%
learning_rate = 0.0001    
nepoch = 25               
T = 50                   # length of sequence
hidden_dim = 100         
output_dim = 1

U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))

bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10

print(X.shape)
# train model
for i in range(y.shape[0]):
    x_i, y_i = X[i], y[i]
    layers = []
    prev_s = np.zeros((hidden_dim, 1))
    dU = np.zeros(U.shape)
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)
    
    dU_t = np.zeros(U.shape)
    dV_t = np.zeros(V.shape)
    dW_t = np.zeros(W.shape)
    
    dU_i = np.zeros(U.shape)
    dW_i = np.zeros(W.shape)
    
    # forward pass
    for t in range(T):
        new_input = np.zeros(x_i.shape)
        x_t = x_i[t]
        new_input[t] = x_t
        mulu = np.dot(U, new_input)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = Sigmoid.act(add)
        mulv = np.dot(V, s)
        layers.append({'s':s, 'prev_s':prev_s})
        prev_s = s
#%%
data = []
target =[]
t = np.linspace(0,np.pi*2,100)
freqs = np.linspace(-5,-5,740)

for freq in freqs:
    a = np.around(np.sin(freq*t)/2+1/2,2)*100
    a_hot = util.onehot(a)
    for i in range(len(a_hot)-5):
        data.append(a_hot[i:i+5])
        target.append(a_hot[i+5])
data = np.stack(data)
target = np.stack(target)
data.shape
#%%

# x = input
# o = output
# s = num of neurons

# U, V, W = weights 
# x -> U -> s -> V -> o
# s -> W -> s


#xso model
#W = ws_ss + bs_s
#U = ws_xs
#V = ws_so + bs_o

# s = tanh(    ws_xs*x + ws_ss*s+bs_s)
# o = softmax( ws_so*s + bs_o)





class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), 
                                   np.sqrt(1./word_dim), 
                                   (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), 
                                   np.sqrt(1./hidden_dim), 
                                   (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), 
                                   np.sqrt(1./hidden_dim), 
                                   (hidden_dim, hidden_dim))
    
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step…
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]
        
    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence…
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards…
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t])*(1 – (s[t]**2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 – s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

RNNNumpy.bptt = bptt
RNNNumpy.forward_propagation = forward_propagation
RNNNumpy.predict = predict
RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss
