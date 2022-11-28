import numpy as np
def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)
def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.这一段是对两个数组进行相减然后求取平均值
  return ((y_true - y_pred) ** 2).mean()
class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # 第一层的权重，Weights
    self.w1_1_1 = np.random.normal()
    self.w1_1_2 = np.random.normal()
    self.w1_1_3 = np.random.normal()
    self.w1_1_4 = np.random.normal()
    self.w1_1_5 = np.random.normal()
    self.w1_1_6 = np.random.normal()
    self.w1_1_7 = np.random.normal()
    self.w1_1_8 = np.random.normal()

    self.w1_2_1 = np.random.normal()
    self.w1_2_2 = np.random.normal()
    self.w1_2_3 = np.random.normal()
    self.w1_2_4 = np.random.normal()
    self.w1_2_5 = np.random.normal()
    self.w1_2_6 = np.random.normal()
    self.w1_2_7 = np.random.normal()
    self.w1_2_8 = np.random.normal()

    self.w1_3_1 = np.random.normal()
    self.w1_3_2 = np.random.normal()
    self.w1_3_3 = np.random.normal()
    self.w1_3_4 = np.random.normal()
    self.w1_3_5 = np.random.normal()
    self.w1_3_6 = np.random.normal()
    self.w1_3_7 = np.random.normal()
    self.w1_3_8 = np.random.normal()

    self.w1_4_1 = np.random.normal()
    self.w1_4_2 = np.random.normal()
    self.w1_4_3 = np.random.normal()
    self.w1_4_4 = np.random.normal()
    self.w1_4_5 = np.random.normal()
    self.w1_4_6 = np.random.normal()
    self.w1_4_7 = np.random.normal()
    self.w1_4_8 = np.random.normal()

    self.w1_5_1 = np.random.normal()
    self.w1_5_2 = np.random.normal()
    self.w1_5_3 = np.random.normal()
    self.w1_5_4 = np.random.normal()
    self.w1_5_5 = np.random.normal()
    self.w1_5_6 = np.random.normal()
    self.w1_5_7 = np.random.normal()
    self.w1_5_8 = np.random.normal()

    self.w1_6_1 = np.random.normal()
    self.w1_6_2 = np.random.normal()
    self.w1_6_3 = np.random.normal()
    self.w1_6_4 = np.random.normal()
    self.w1_6_5 = np.random.normal()
    self.w1_6_6 = np.random.normal()
    self.w1_6_7 = np.random.normal()
    self.w1_6_8 = np.random.normal()

    self.w1_7_1 = np.random.normal()
    self.w1_7_2 = np.random.normal()
    self.w1_7_3 = np.random.normal()
    self.w1_7_4 = np.random.normal()
    self.w1_7_5 = np.random.normal()
    self.w1_7_6 = np.random.normal()
    self.w1_7_7 = np.random.normal()
    self.w1_7_8 = np.random.normal()

    self.w1_8_1 = np.random.normal()
    self.w1_8_2 = np.random.normal()
    self.w1_8_3 = np.random.normal()
    self.w1_8_4 = np.random.normal()
    self.w1_8_5 = np.random.normal()
    self.w1_8_6 = np.random.normal()
    self.w1_8_7 = np.random.normal()
    self.w1_8_8 = np.random.normal()

    # 第二层的权重，Weights
    self.w2_1 = np.random.normal()
    self.w2_2 = np.random.normal()
    self.w2_3 = np.random.normal()
    self.w2_4 = np.random.normal()
    self.w2_5 = np.random.normal()
    self.w2_6 = np.random.normal()
    self.w2_7 = np.random.normal()
    self.w2_8 = np.random.normal()

    # 截距项，Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()
    self.b5 = np.random.normal()
    self.b6 = np.random.normal()
    self.b7 = np.random.normal()
    self.b8 = np.random.normal()
    self.b9 = np.random.normal()

  def feedforward(self, x):
      # x is a numpy array with 2 elements.
      h1 = sigmoid(self.w1_1_1 * x[0] + self.w1_1_2 * x[1] + self.w1_1_3 * x[2] + self.w1_1_4 * x[3] + self.w1_1_5 * x[4] + self.w1_1_6 * x[5] + self.w1_1_7 * x[6] + self.w1_1_8 * x[7] + self.b1)
      h2 = sigmoid(self.w1_2_1 * x[0] + self.w1_2_2 * x[1] + self.w1_2_3 * x[2] + self.w1_2_4 * x[3] + self.w1_2_5 * x[4] + self.w1_2_6 * x[5] + self.w1_2_7 * x[6] + self.w1_2_8 * x[7] + self.b2)
      h3 = sigmoid(self.w1_3_1 * x[0] + self.w1_3_2 * x[1] + self.w1_3_3 * x[2] + self.w1_3_4 * x[3] + self.w1_3_5 * x[4] + self.w1_3_6 * x[5] + self.w1_3_7 * x[6] + self.w1_3_8 * x[7] + self.b3)
      h4 = sigmoid(self.w1_4_1 * x[0] + self.w1_4_2 * x[1] + self.w1_4_3 * x[2] + self.w1_4_4 * x[3] + self.w1_4_5 * x[4] + self.w1_4_6 * x[5] + self.w1_4_7 * x[6] + self.w1_4_8 * x[7] + self.b4)
      h5 = sigmoid(self.w1_5_1 * x[0] + self.w1_5_2 * x[1] + self.w1_5_3 * x[2] + self.w1_5_4 * x[3] + self.w1_5_5 * x[4] + self.w1_5_6 * x[5] + self.w1_5_7 * x[6] + self.w1_5_8 * x[7] + self.b5)
      h6 = sigmoid(self.w1_6_1 * x[0] + self.w1_6_2 * x[1] + self.w1_6_3 * x[2] + self.w1_6_4 * x[3] + self.w1_6_5 * x[4] + self.w1_6_6 * x[5] + self.w1_6_7 * x[6] + self.w1_6_8 * x[7] + self.b6)
      h7 = sigmoid(self.w1_7_1 * x[0] + self.w1_7_2 * x[1] + self.w1_7_3 * x[2] + self.w1_7_4 * x[3] + self.w1_7_5 * x[4] + self.w1_7_6 * x[5] + self.w1_7_7 * x[6] + self.w1_7_8 * x[7] + self.b7)
      h8 = sigmoid(self.w1_8_1 * x[0] + self.w1_8_2 * x[1] + self.w1_8_3 * x[2] + self.w1_8_4 * x[3] + self.w1_8_5 * x[4] + self.w1_8_6 * x[5] + self.w1_8_7 * x[6] + self.w1_8_8 * x[7] + self.b8)
      o1 = sigmoid(self.w2_1 * h1 + self.w2_2 * h2+self.w2_3 * h3 + self.w2_4 * h4+self.w2_5 * h5 + self.w2_6 * h6+self.w2_7 * h7 + self.w2_8 * h8+self.b9)
      return o1

  def train(self, data, all_y_trues):
      '''
      - data is a (n x 2) numpy array, n = # of samples in the dataset.
      - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
      '''
      learn_rate = 0.1
      epochs = 1100  # number of times to loop through the entire dataset

      for epoch in range(epochs):
          for x, z_true in zip(data, all_y_trues):
              # --- Do a feedforward (we'll need these values later)
              sum_h1 = self.w1_1_1 * x[0] + self.w1_1_2 * x[1]+ self.w1_1_3 * x[2]+ self.w1_1_4 * x[3]+ self.w1_1_5 * x[4]+ self.w1_1_6 * x[5]+ self.w1_1_7 * x[6]+ self.w1_1_8 * x[7] + self.b1
              h1 = sigmoid(sum_h1)

              sum_h2 = self.w1_2_1 * x[0] + self.w1_2_2 * x[1]+ self.w1_2_3 * x[2]+ self.w1_2_4 * x[3]+ self.w1_2_5 * x[4]+ self.w1_2_6 * x[5]+ self.w1_2_7 * x[6]+ self.w1_2_8 * x[7] + self.b2
              h2 = sigmoid(sum_h2)

              sum_h3 = self.w1_3_1 * x[0] + self.w1_3_2 * x[1] + self.w1_3_3 * x[2] + self.w1_3_4 * x[3] + self.w1_3_5 * x[4] + self.w1_3_6 * x[5] + self.w1_3_7 * x[6] + self.w1_3_8 * x[7] + self.b3
              h3 = sigmoid(sum_h3)

              sum_h4 =self.w1_4_1 * x[0] + self.w1_4_2 * x[1] + self.w1_4_3 * x[2] + self.w1_4_4 * x[3] + self.w1_4_5 * x[4] + self.w1_4_6 * x[5] + self.w1_4_7 * x[6] + self.w1_4_8 * x[7] + self.b4
              h4 = sigmoid(sum_h4)

              sum_h5 =self.w1_5_1 * x[0] + self.w1_5_2 * x[1] + self.w1_5_3 * x[2] + self.w1_5_4 * x[3] + self.w1_5_5 * x[4] + self.w1_5_6 * x[5] + self.w1_5_7 * x[6] + self.w1_5_8 * x[7] + self.b5
              h5 = sigmoid(sum_h5)

              sum_h6 = self.w1_6_1 * x[0] + self.w1_6_2 * x[1] + self.w1_6_3 * x[2] + self.w1_6_4 * x[3] + self.w1_6_5 * x[4] + self.w1_6_6 * x[5] + self.w1_6_7 * x[6] + self.w1_6_8 * x[7] + self.b6
              h6 = sigmoid(sum_h6)

              sum_h7 =self.w1_7_1 * x[0] + self.w1_7_2 * x[1] + self.w1_7_3 * x[2] + self.w1_7_4 * x[3] + self.w1_7_5 * x[4] + self.w1_7_6 * x[5] + self.w1_7_7 * x[6] + self.w1_7_8 * x[7] + self.b7
              h7 = sigmoid(sum_h7)

              sum_h8 =self.w1_8_1 * x[0] + self.w1_8_2 * x[1] + self.w1_8_3 * x[2] + self.w1_8_4 * x[3] + self.w1_8_5 * x[4] + self.w1_8_6 * x[5] + self.w1_8_7 * x[6] + self.w1_8_8 * x[7] + self.b8
              h8 = sigmoid(sum_h8)

              sum_o1 = self.w2_1 * h1 + self.w2_2 * h2+self.w2_3 * h3 + self.w2_4 * h4+self.w2_5 * h5 + self.w2_6 * h6+self.w2_7 * h7 + self.w2_8 * h8+self.b9
              o1 = sigmoid(sum_o1)
              z_pred = o1
              # --- Calculate partial derivatives.
              # --- Naming: d_L_d_w1 represents "partial L / partial w1"
              d_L_d_zpred = -2 * (z_true - z_pred)

              # Neuron o1
              d_zpred_d_w2_1 = h1 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_2 = h2 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_3 = h3 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_4 = h4 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_5 = h5 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_6 = h6 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_7 = h7 * deriv_sigmoid(sum_o1)
              d_zpred_d_w2_8 = h8 * deriv_sigmoid(sum_o1)
              d_zpred_d_b9 = deriv_sigmoid(sum_o1)

              d_zpred_d_h1 = self.w2_1 * deriv_sigmoid(sum_o1)
              d_zpred_d_h2 = self.w2_2 * deriv_sigmoid(sum_o1)
              d_zpred_d_h3 = self.w2_3 * deriv_sigmoid(sum_o1)
              d_zpred_d_h4 = self.w2_4 * deriv_sigmoid(sum_o1)
              d_zpred_d_h5 = self.w2_5 * deriv_sigmoid(sum_o1)
              d_zpred_d_h6 = self.w2_6 * deriv_sigmoid(sum_o1)
              d_zpred_d_h7 = self.w2_7 * deriv_sigmoid(sum_o1)
              d_zpred_d_h8 = self.w2_8 * deriv_sigmoid(sum_o1)

              # Neuron h1
              d_h1_d_w1_1_1 = x[0] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_2 = x[1] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_3 = x[2] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_4 = x[3] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_5 = x[4] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_6 = x[5] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_7 = x[6] * deriv_sigmoid(sum_h1)
              d_h1_d_w1_1_8 = x[7] * deriv_sigmoid(sum_h1)
              d_h1_d_b1 = deriv_sigmoid(sum_h1)

              # Neuron h2
              d_h2_d_w1_2_1 = x[0] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_2 = x[1] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_3 = x[2] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_4 = x[3] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_5 = x[4] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_6 = x[5] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_7 = x[6] * deriv_sigmoid(sum_h2)
              d_h2_d_w1_2_8 = x[7] * deriv_sigmoid(sum_h2)
              d_h2_d_b2 = deriv_sigmoid(sum_h2)

              # Neuron h3
              d_h3_d_w1_3_1 = x[0] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_2 = x[1] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_3 = x[2] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_4 = x[3] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_5 = x[4] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_6 = x[5] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_7 = x[6] * deriv_sigmoid(sum_h3)
              d_h3_d_w1_3_8 = x[7] * deriv_sigmoid(sum_h3)
              d_h3_d_b3 = deriv_sigmoid(sum_h3)

              # Neuron h4
              d_h4_d_w1_4_1 = x[0] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_2 = x[1] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_3 = x[2] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_4 = x[3] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_5 = x[4] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_6 = x[5] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_7 = x[6] * deriv_sigmoid(sum_h4)
              d_h4_d_w1_4_8 = x[7] * deriv_sigmoid(sum_h4)
              d_h4_d_b4 = deriv_sigmoid(sum_h4)

              # Neuron h5
              d_h5_d_w1_5_1 = x[0] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_2 = x[1] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_3 = x[2] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_4 = x[3] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_5 = x[4] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_6 = x[5] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_7 = x[6] * deriv_sigmoid(sum_h5)
              d_h5_d_w1_5_8 = x[7] * deriv_sigmoid(sum_h5)
              d_h5_d_b5 = deriv_sigmoid(sum_h5)

              # Neuron h6
              d_h6_d_w1_6_1 = x[0] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_2 = x[1] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_3 = x[2] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_4 = x[3] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_5 = x[4] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_6 = x[5] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_7 = x[6] * deriv_sigmoid(sum_h6)
              d_h6_d_w1_6_8 = x[7] * deriv_sigmoid(sum_h6)
              d_h6_d_b6 = deriv_sigmoid(sum_h6)

              # Neuron h7
              d_h7_d_w1_7_1 = x[0] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_2 = x[1] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_3 = x[2] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_4 = x[3] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_5 = x[4] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_6 = x[5] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_7 = x[6] * deriv_sigmoid(sum_h7)
              d_h7_d_w1_7_8 = x[7] * deriv_sigmoid(sum_h7)
              d_h7_d_b7 = deriv_sigmoid(sum_h7)

              # Neuron h8
              d_h8_d_w1_8_1 = x[0] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_2 = x[1] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_3 = x[2] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_4 = x[3] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_5 = x[4] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_6 = x[5] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_7 = x[6] * deriv_sigmoid(sum_h8)
              d_h8_d_w1_8_8 = x[7] * deriv_sigmoid(sum_h8)
              d_h8_d_b8 = deriv_sigmoid(sum_h8)
              # --- Update weights and biases
              # Neuron h1
              self.w1_1_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_1
              self.w1_1_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_2
              self.w1_1_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_3
              self.w1_1_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_4
              self.w1_1_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_5
              self.w1_1_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_6
              self.w1_1_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_7
              self.w1_1_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_w1_1_8
              self.b1 -= learn_rate * d_L_d_zpred * d_zpred_d_h1 * d_h1_d_b1

              # Neuron h2
              self.w1_2_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_1
              self.w1_2_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_2
              self.w1_2_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_3
              self.w1_2_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_4
              self.w1_2_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_5
              self.w1_2_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_6
              self.w1_2_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_7
              self.w1_2_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_w1_2_8
              self.b2 -= learn_rate * d_L_d_zpred * d_zpred_d_h2 * d_h2_d_b2

              # Neuron h3
              self.w1_3_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_1
              self.w1_3_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_2
              self.w1_3_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_3
              self.w1_3_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_4
              self.w1_3_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_5
              self.w1_3_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_6
              self.w1_3_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_7
              self.w1_3_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_w1_3_8
              self.b3 -= learn_rate * d_L_d_zpred * d_zpred_d_h3 * d_h3_d_b3

              # Neuron h4
              self.w1_4_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_1
              self.w1_4_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_2
              self.w1_4_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_3
              self.w1_4_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_4
              self.w1_4_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_5
              self.w1_4_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_6
              self.w1_4_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_7
              self.w1_4_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_w1_4_8
              self.b4 -= learn_rate * d_L_d_zpred * d_zpred_d_h4 * d_h4_d_b4

              # Neuron h5
              self.w1_5_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_1
              self.w1_5_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_2
              self.w1_5_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_3
              self.w1_5_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_4
              self.w1_5_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_5
              self.w1_5_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_6
              self.w1_5_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_7
              self.w1_5_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_w1_5_8
              self.b5 -= learn_rate * d_L_d_zpred * d_zpred_d_h5 * d_h5_d_b5

              # Neuron h6
              self.w1_6_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_1
              self.w1_6_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_2
              self.w1_6_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_3
              self.w1_6_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_4
              self.w1_6_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_5
              self.w1_6_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_6
              self.w1_6_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_7
              self.w1_6_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_w1_6_8
              self.b6 -= learn_rate * d_L_d_zpred * d_zpred_d_h6 * d_h6_d_b6

              # Neuron h7
              self.w1_7_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_1
              self.w1_7_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_2
              self.w1_7_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_3
              self.w1_7_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_4
              self.w1_7_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_5
              self.w1_7_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_6
              self.w1_7_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_7
              self.w1_7_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_w1_7_8
              self.b7 -= learn_rate * d_L_d_zpred * d_zpred_d_h7 * d_h7_d_b7

              # Neuron h8
              self.w1_8_1 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_1
              self.w1_8_2 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_2
              self.w1_8_3 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_3
              self.w1_8_4 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_4
              self.w1_8_5 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_5
              self.w1_8_6 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_6
              self.w1_8_7 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_7
              self.w1_8_8 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_w1_8_8
              self.b8 -= learn_rate * d_L_d_zpred * d_zpred_d_h8 * d_h8_d_b8

              # Neuron o1
              self.w2_1 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_1
              self.w2_2 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_2
              self.w2_3 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_3
              self.w2_4 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_4
              self.w2_5 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_5
              self.w2_6 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_6
              self.w2_7 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_7
              self.w2_8 -= learn_rate * d_L_d_zpred * d_zpred_d_w2_8
              self.b9 -= learn_rate * d_L_d_zpred * d_zpred_d_b9

              # --- Calculate total loss at the end of each epoch
          if epoch % 10 == 0:
              z_preds = np.apply_along_axis(self.feedforward, 1, data)
              loss = mse_loss(all_y_trues, z_preds)
              print("Epoch: %d loss: %.3f" % (epoch, loss))
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
emily=np.array([-7,-3])
frank=np.array([20,3])
print("Emily: %.3f" %network.feedforward(emily))
print("Frank: %.3f" %network.feedforward(frank))