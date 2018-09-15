# coding: utf-8
import random
import math
import sympy
import numpy
'''
    Partial_derivative ---- pd
'''

class Neuron():
    def __init__(self, input, weight, bias):
        self.__inputdata = input
        self.__weight = weight
        self.__bias = bias
        # self.__netdata = self.calculate_net()
        # self.__outdata = self.excitation(self.__netdata)

    def excitation(self, netdata):
        outdata = 1 / ( 1 + math.exp(-netdata) )
        return outdata

    def calculate_net(self):
        netdata = numpy.dot(self.__inputdata, self.__weight)
        netdata = netdata + self.__bias
        return netdata

    def calculate_out(self):
        return self.excitation(netdata=self.calculate_net())

    def update_bias(self, new_bias):
        self.__bias = new_bias

    def get_bias(self):
        return self.__bias

    def update_weight(self, data):
        self.__weight = data

    def get_weight(self):
        return self.__weight

    def update_inputs(self, new_input):
        self.__inputdata = new_input

    def get_inputs(self):
        return self.__inputdata

    def network_result_error(self, target_result):
        error = 0.5 * (( target_result - self.__outdata ) ** 2)
        return error

    def calculate_pd_error_out(self, outdata, target_data):
        x = sympy.symbols("x")
        y = 0.5 * (( target_data - x ) ** 2)
        error_out_dify = sympy.diff(y, x).subs({x: outdata})
        return error_out_dify

    def calculate_pd_out_net(self, netdata):
        x = sympy.symbols("x")
        y = 1 / ( 1 + sympy.exp(-x))
        out_net_dify = sympy.diff(y, x).subs({x: netdata})
        return out_net_dify

    def calculate_pd_net_input(self, input_index):
        return self.__weight[input_index]

    def calculate_pd_net_weight(self, weight_index):
        return self.__inputdata[weight_index]

#python继承时 私有不可继承覆盖
class Layer():
    def __init__(self, neurons_num, input_num, bias):
        self.neuronsnum = neurons_num
        self.inputnum = input_num
        self.input, self.neurons = self.create_neurons(input_num, bias)
        self.bias = bias

    def create_neurons(self, input_num, bias):
        neurons = []
        inputs = []
        for i in range(input_num):
            inputs.append(0)
        for i in range(self.neuronsnum):
            weight = []
            for j in range(input_num):
                weight.append(random.random())
            neuron = Neuron(inputs, weight, bias)
            neurons.append(neuron)
        return inputs, neurons

    def calculate_layer_outputs(self):
        outputs = []
        for i in range(self.neuronsnum):
            output = self.neurons[i].calculate_out()
            outputs.append(output)
        return outputs

    def update_neurons_weights(self, new_weights):
        for i in range(self.neuronsnum):
            self.neurons[i].update_weight(data=new_weights[i])

    def get_neurons_weights(self):
        weights = []
        for i in range(self.neuronsnum):
            weights.append(self.neurons[i].get_weight())
        return weights

    def update_neurons_bias(self, new_bias):
        self.bias = new_bias
        for i in range(self.neuronsnum):
            self.neurons[i].update_bias(new_bias=new_bias)

    def get_neurons_bias(self):
        return self.bias

    def update_neurons_inputs(self, new_inputs):
        for i in range(self.neuronsnum):
            self.neurons[i].update_inputs(new_input=new_inputs)

    def get_neurons_inputs(self):
        return self.neurons[0].get_inputs()


class OutputLayer(Layer):

    def __init__(self, neurons_num, input_num, bias, targets):
        Layer.__init__(self, neurons_num=neurons_num, input_num=input_num, bias=bias)
        self.__targets = targets

    def calculate_pd_bp_outputlayer(self):
        error_pd_net_outputlayer = []
        for i in range(self.neuronsnum):
            thisoutdata = self.neurons[i].calculate_out()
            error_pd_out = self.neurons[i].calculate_pd_error_out(outdata=thisoutdata, target_data=self.__targets[i])
            thisnetdata = self.neurons[i].calculate_net()
            out_pd_net = self.neurons[i].calculate_pd_out_net(netdata=thisnetdata)
            error_pd_net = error_pd_out * out_pd_net
            error_pd_net_outputlayer.append(error_pd_net)
        return error_pd_net_outputlayer

    def calculate_pd_bp_outputweights(self, error_pd_net_outputlayer):
        error_pd_weight_outputlayer = []
        for i in range(self.neuronsnum):
            error_pd_weight_neurons = []
            error_pd_net = error_pd_net_outputlayer[i]
            for j in range(self.inputnum):
                net_pd_weight = self.neurons[i].calculate_pd_net_weight(j)
                error_pd_weight = error_pd_net * net_pd_weight
                error_pd_weight_neurons.append(error_pd_weight)
            error_pd_weight_outputlayer.append(error_pd_weight_neurons)
        #print(error_pd_weight_outputlayer)
        return error_pd_weight_outputlayer

    def gradient_descent_outputweights(self, alpha):
        error_pd_net_outputlayer = self.calculate_pd_bp_outputlayer()
        error_pd_weight_outputlayer = self.calculate_pd_bp_outputweights(error_pd_net_outputlayer=error_pd_net_outputlayer)
        new_weights = []
        for i in range(self.neuronsnum):
            thisneurons = self.neurons[i]
            thisweights = []
            for j in range(self.inputnum):
                new_weight = (thisneurons.get_weight())[j] - alpha * ((error_pd_weight_outputlayer[i])[j])
                thisweights.append(new_weight)
            new_weights.append(thisweights)
        self.update_neurons_weights(new_weights)

    def gradient_descent_outputbias(self, alpha):
        error_pd_net_outputlayer = self.calculate_pd_bp_outputlayer()
        error_pd_bias = 0
        for i in range(self.neuronsnum):
           error_pd_bias += error_pd_net_outputlayer[i]
        new_bias = self.bias - alpha * error_pd_bias
        self.update_neurons_bias(new_bias)

    def calculate_error_pd_hiddenouts(self):
        error_pd_net_outputlayer = self.calculate_pd_bp_outputlayer()
        error_pd_hiddenouts = []
        for j in range(self.inputnum):
            error_pd_hiddenout = 0
            for i in range(self.neuronsnum):
                error_pd_hiddenout += error_pd_net_outputlayer[i] * (self.neurons[i].calculate_pd_net_input(j))
            error_pd_hiddenouts.append(error_pd_hiddenout)
        return error_pd_hiddenouts

class HiddenLayer(Layer):
    def __init__(self, neurons_num, input_num, bias):
        Layer.__init__(self, neurons_num=neurons_num, input_num=input_num, bias=bias)

    def calculate_error_pd_net(self, error_pd_outs):
        error_pd_nets = []
        for i in range(self.neuronsnum):
            netdata = self.neurons[i].calculate_net()
            error_pd_net_neurons = error_pd_outs[i] * (self.neurons[i].calculate_pd_out_net(netdata=netdata))
            error_pd_nets.append(error_pd_net_neurons)
        return error_pd_nets

    def calculate_error_pd_weights(self, error_pd_nets):
        error_pd_weights = []
        for i in range(self.neuronsnum):
            error_pd_net = error_pd_nets[i]
            error_pd_weights_neurons = []
            for j in range(self.inputnum):
                net_pd_weight = self.neurons[i].calculate_pd_net_weight(j)
                error_pd_weight = error_pd_net * net_pd_weight
                error_pd_weights_neurons.append(error_pd_weight)
            error_pd_weights.append(error_pd_weights_neurons)
        return error_pd_weights

    def gradient_descent_hiddenweights(self, error_pd_outs, alpha):
        error_pd_nets = self.calculate_error_pd_net(error_pd_outs=error_pd_outs)
        error_pd_weights = self.calculate_error_pd_weights(error_pd_nets=error_pd_nets)
        new_weights = []
        for i in range(self.neuronsnum):
            new_weight = []
            thisneurons = self.neurons[i]
            thisweights = thisneurons.get_weight()
            for j in range(self.inputnum):
                weight = thisweights[j] - alpha * ((error_pd_weights[i])[j])
                new_weight.append(weight)
            new_weights.append(new_weight)
        self.update_neurons_weights(new_weights)

    def gradient_descent_hiddenbias(self, error_pd_outs, alpha):
        error_pd_nets = self.calculate_error_pd_net(error_pd_outs=error_pd_outs)
        error_pd_bias = 0
        for i in range(self.neuronsnum):
            error_pd_bias += error_pd_nets[i]
        new_bias = self.bias - alpha * error_pd_bias
        self.update_neurons_bias(new_bias=new_bias)


    def calculate_error_pd_lasthiddenouts(self, error_pd_outs):
        error_pd_nets = self.calculate_error_pd_net(error_pd_outs=error_pd_outs)
        error_pd_lasthiddenouts = []
        for j in range(self.inputnum):
            error_pd_lastouts = 0
            for i in range(self.neuronsnum):
                error_pd_lastouts += (error_pd_nets[i]) * ((self.neurons[i]).calculate_pd_net_input(j))
            error_pd_lasthiddenouts.append(error_pd_lastouts)
        return error_pd_lasthiddenouts

class NeuronNetwork():
    def __init__(self):
        self.hiddenLayer1 = HiddenLayer(3, 3, random.random())
        self.hiddenLayer2 = HiddenLayer(3, 3, random.random())
        self.hiddenLayer3 = HiddenLayer(3, 3, random.random())
        self.outputLayer = OutputLayer(2, 3, random.random(), [0.01, 0.99])
        self.inputs = []

    def inputnum(self, inputs):
        for i in inputs:
            self.inputs.append(i)

    def train(self, alpha):
        self.hiddenLayer1.update_neurons_inputs(new_inputs=self.inputs)

        for i in range(1500):
            hiddenlayer1_outs = self.hiddenLayer1.calculate_layer_outputs()
            self.hiddenLayer2.update_neurons_inputs(new_inputs=hiddenlayer1_outs)
            hiddenlayer2_outs = self.hiddenLayer2.calculate_layer_outputs()
            self.hiddenLayer3.update_neurons_inputs(new_inputs=hiddenlayer2_outs)
            hiddenlayer3_outs = self.hiddenLayer3.calculate_layer_outputs()
            self.outputLayer.update_neurons_inputs(new_inputs=hiddenlayer3_outs)
            print(self.outputLayer.calculate_layer_outputs())

            self.outputLayer.gradient_descent_outputbias(alpha=alpha)
            self.outputLayer.gradient_descent_outputweights(alpha=alpha)
            error_pd_outs_hidden3 = self.outputLayer.calculate_error_pd_hiddenouts()

            self.hiddenLayer3.gradient_descent_hiddenweights(error_pd_outs=error_pd_outs_hidden3, alpha=alpha)
            self.hiddenLayer3.gradient_descent_hiddenbias(error_pd_outs=error_pd_outs_hidden3, alpha=alpha)
            error_pd_outs_hidden2 = self.hiddenLayer3.calculate_error_pd_lasthiddenouts(error_pd_outs_hidden3)

            self.hiddenLayer2.gradient_descent_hiddenweights(error_pd_outs=error_pd_outs_hidden2, alpha=alpha)
            self.hiddenLayer2.gradient_descent_hiddenbias(error_pd_outs=error_pd_outs_hidden2, alpha=alpha)
            error_pd_outs_hidden1 = self.hiddenLayer2.calculate_error_pd_lasthiddenouts(error_pd_outs=error_pd_outs_hidden2)

            self.hiddenLayer1.gradient_descent_hiddenweights(error_pd_outs=error_pd_outs_hidden1, alpha=alpha)
            self.hiddenLayer1.gradient_descent_hiddenbias(error_pd_outs=error_pd_outs_hidden1, alpha=alpha)

if __name__ == "__main__":
    nn = NeuronNetwork()
    nn.inputnum(inputs=[0.1, 0.1, 0.1])
    nn.train(alpha=0.3)














