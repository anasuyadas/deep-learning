import copy, numpy as np
np.random.seed(0)
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

#generate training dataset
int2binary = {}
binary_dim =8
largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1) #binarize the array in to int8 dtype

for i in range(largest_number):
    int2binary[i] = binary[i]

#input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

#initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim))-1
synapse_1 = 2*np.random.random((hidden_dim,output_dim))
synapse_h = 2*np.random.random((hidden_dim,hidden_dim))

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

#training logic
for j in range(10000):
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2)
    a= int2binary[a_int]

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    c_int = a_int + b_int
    c = int2binary[c_int]


    #store our best guess in d
    d =  np.zeros_like(c)

    overallerror=0

    layer_2_deltas =list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    for position in range(binary_dim):
        x= np.array([[a[binary_dim-position-1],b[binary_dim-position-1]]])

        y = np.array([[c[binary_dim-position-1]]]).T

        #hidden layer (input~+prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        #output layer
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        #calculate error
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error)*sigmoid_output_to_derivative(layer_2)
        overallerror +=np.abs(layer_2_error[0])

        #decode estimate
        d[binary_dim-position-1] = np.round(layer_2[0][0])

        #store the output of the hidden layer

        layer_1_values.append(np.copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):

        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        #error at output layer
        layer_2_deltas = layer_2_deltas[-position-1]
        #error at hidden layer
        layer_1_delta = future_layer_1_delta.dot(synapse_h.T)+
        layer_2_deltas.dot(synapse_1.T) * sigmoid_output_to_derivative(future_layer_1_delta)

        #lets update weights
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_deltas)

