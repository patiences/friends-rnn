import numpy as np
import os

def process_inputs():
    if not os.path.isfile('data/input.txt'):
        build_input_file()
    data = open('data/input.txt', 'r').read()
    uniq_chars = set(data)
    data_size, vocab_size = len(data), len(uniq_chars)

    print('data/input.txt has %d characters, %d unique.' % (data_size, vocab_size))

    char_to_idx = { ch:i for i,ch in enumerate(uniq_chars) }
    idx_to_char = { i:ch for i,ch in enumerate(uniq_chars) }

    return data, data_size, vocab_size, char_to_idx, idx_to_char

def build_input_file():
    """
    Concatenate all data/*.txt files into single data/input.txt file.
    """
    print("Building data/input.txt...")
    filenames = []
    for filename in os.listdir("data/"):
        if filename.endswith(".txt"):
            filenames.append("data/" + filename)

    with open("input.txt", "a") as outfile:
        for fname in filenames:
            print(fname)
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

class RNN:

    """
    This code is adapted from Andrej Karpathy's minimal char-rnn model,
    https://gist.github.com/karpathy/d4dee566867f8291f086
    """

    def __init__(self, learning_rate=1e-1, hidden_layer_size=100, max_iter=200):
        """
        learning_rate: constant multiplied to gradient during param update.
            A larger value means the parameters (weights) will change more on
            each iteration of gradient descent.

        hidden_layer_size: number of neurons in the hidden layer

        max_iter: maximum number of iterations, or to convergence,
            whichever is less
        """
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.max_iter = max_iter

    def fit(self, data, vocab_size, char_to_idx, idx_to_char, seq_length=25):
        """
        data: list of integers representing the input text

        vocab_size: the number of unique characters in the input text

        char_to_idx: a mapping from the unique characters in the input text to integers

        seq_length: the number of previous characters to consider
            when predicting the next character
        """

        self.vocab_size = vocab_size

        # Initialize model parameters, sample from Gaussian distribution
        self.Wxh = np.random.randn(self.hidden_layer_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_layer_size, self.hidden_layer_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_layer_size) * 0.01
        self.bias_h = np.zeros((self.hidden_layer_size, 1))            # hidden_layer_size by 1
        self.bias_y = np.zeros((self.vocab_size, 1))                   # vocab_size by 1

        num_iter, current_char_idx = 0, 0

        while num_iter < self.max_iter:
            # If we're at the start or about to go off the end, reset RNN memory and go back to start
            if current_char_idx == 0 or current_char_idx + seq_length + 1 >= len(data):
                hidden_prev = np.zeros((self.hidden_layer_size, 1))
                current_char_idx = 0

            # Grab indices for this block of characters
            start, end = current_char_idx, current_char_idx + seq_length
            inputs = [char_to_idx[ch] for ch in data[start:end]]
            targets = [char_to_idx[ch] for ch in data[start+1:end+1]]

            X, H, y, f = self._forward_pass(inputs, targets, hidden_prev)
            Gxh, Ghh, Ghy, gbias_h, gbias_y = self._backpropagate(inputs, targets, X, H, y)
            self._update_params(Gxh, Ghh, Ghy, gbias_h, gbias_y)

            # TODO progress updates, save checkpoints

            # Make predictions
            if num_iter % 50 == 0:
                # Choose a random character as seed
                seed = np.random.choice(range(self.vocab_size), p=y[0].ravel())
                prediction_idxs = self.predict(seed, hidden_prev, 100)
                print(''.join([idx_to_char[idx] for idx in prediction_idxs]))

            current_char_idx += seq_length
            num_iter += 1

    def _forward_pass(self, inputs, targets, hidden_prev):
        """
        Compute the loss by making a forward pass through the network.
        """
        seq_length = len(inputs)

        X = np.zeros((seq_length, self.vocab_size, 1))                 # seq_length of vocab_size by 1 (column) vector inputs, encoded as 1 of k
        H = np.zeros((seq_length, self.hidden_layer_size, 1))          # Hidden state
        y = np.zeros((seq_length, self.vocab_size, 1))                 # Outputs
        f = 0                                                          # initialize loss to 0

        for i in range(seq_length):
            char_idx = inputs[i] # Get back the (index representation of the) char
            # X[i][j] = 1 means char j occurred at index i in the sequence
            X[i][char_idx] = 1

            if i is not 0:
                # h by v * v by 1
                H[i] = np.tanh(np.dot(self.Wxh, X[i]) + np.dot(self.Whh, H[i-1]) + self.bias_h)      # Neuron in hidden layer, with tanh nonlinearity applied
            else:
                H[i] = np.tanh(np.dot(self.Wxh, X[i]) + np.dot(self.Whh, hidden_prev) + self.bias_h)  # Use hidden state from previous forward pass

            # P(y[i] | X[i], W)
            y[i] = np.dot(self.Why, H[i]) + self.bias_y                                         # vocab_size by 1

            y[i] = np.exp(y[i]) / np.sum(np.exp(y[i]))                                     # softmax probability

            f += -np.log(y[i][targets[i]])                                                 # Want to minimize the loss for the right character, targets[i]

        return X, H, y, f

    def _backpropagate(self, inputs, targets, X, H, y):
        """
        Compute gradients backwards through the network.
        """
        Gxh = np.zeros((self.hidden_layer_size, self.vocab_size))        # Gradient wrt w in Wxh
        Ghh = np.zeros((self.hidden_layer_size, self.hidden_layer_size)) # Gradient wrt w in Whh
        Ghy = np.zeros((self.vocab_size, self.hidden_layer_size))        # Gradient wrt w in Why
        gbias_h = np.zeros((self.hidden_layer_size, 1))                  # Gradient wrt biases for hidden layer
        gbias_y = np.zeros((self.vocab_size, 1))                         # Gradient wrt biases for output layer
        dHnext = np.zeros((self.hidden_layer_size, 1))                   # TODO what is this really

        for i in reversed(range(len(inputs))):
            # Backprop from y into Why
            dyi = np.copy(y[i])                                          # Get softmax probabilities from forward pass
            dyi[targets[i]] -= 1                                         # Partial derivative is P(k | X[i], w) - (y[i] = k)
            Ghy += np.dot(dyi, H[i].T)                                   # v by 1 * 1 by h -> v by h
            gbias_y += dyi

            dHi = np.dot(self.Why.T, dyi) + dHnext
            dHiRaw = (1 - H[i]*H[i]) * dHi
            # Backprop from Why into Whh
            Ghh += np.dot(dHiRaw, H[i-1].T)
            gbias_h += dHiRaw

            # Backprop from Whh into Wxh
            Gxh += np.dot(dHiRaw, X[i].T)

            # Update dHnext for next iteration (1 char to left)
            dHnext = np.dot(self.Whh.T, dHiRaw)

        return Gxh, Ghh, Ghy, gbias_h, gbias_y

    def _update_params(self, Gxh, Ghh, Ghy, gbias_h, gbias_y):
        """
        Use AdaGrad to update parameters.
        """

        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bias_h, self.bias_y],
                                 [Gxh, Ghh, Ghy, gbias_h, gbias_y]):
            epsilon = 1e-8
            mem = dparam**2
            param += - self.learning_rate * dparam / np.sqrt(mem + epsilon)

    def predict(self, seed_idx, hidden_state, length):
        """
        Generate a sequence of characters (integers) using the model
        at a specific point in training time.

        seed_idx: a seed letter for the first character
        hidden_state: a hidden state from the model (h by 1)
        length: number of characters to generate
        """
        # the previous character, initialized with seed_idx
        x_prev = np.zeros((self.vocab_size, 1))
        x_prev[seed_idx] = 1

        prediction_idxs = []

        for c in range(length):
            # Generate the next hidden state using this character
            hidden_state = np.tanh(np.dot(self.Wxh, x_prev) + np.dot(self.Whh, hidden_state) + self.bias_h)

            # Prediction vector
            y = np.dot(self.Why, hidden_state) + self.bias_y
            y = np.exp(y) / np.sum(np.exp(y))

            # Predict a character, using the probability distribution y
            idx = np.random.choice(range(self.vocab_size), p=y.ravel())
            prediction_idxs.append(idx)

            # reset previous character
            x_prev = np.zeros((self.vocab_size, 1))
            x_prev[idx] = 1

        return prediction_idxs

def main():
    data, data_size, vocab_size, char_to_idx, idx_to_char = process_inputs()
    model = RNN()
    model.fit(data=data, vocab_size=vocab_size, char_to_idx=char_to_idx, idx_to_char=idx_to_char)



if __name__ == "__main__":
    main()
