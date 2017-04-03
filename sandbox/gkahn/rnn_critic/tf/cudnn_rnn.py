from tensorflow.contrib.cudnn_rnn import CudnnLSTM

def cudnn_lstm(num_layers, num_units, input_size):
    model = CudnnLSTM(num_layers, num_units, input_size)
    params_size = model.params_size()