# coding: utf-8
import random
import math
import sympy
import numpy
'''
    Partial_derivative ---- pd
'''

class Neuron():
    def __init__(self, input, weight):
        self.__inputdata = input
        self.__weight = weight
        # self.__netdata = self.calculate_net()
        # self.__outdata = self.excitation(self.__netdata)

    def excitation(self, netdata):
        outdata = 1 / ( 1 + math.exp(-netdata) )
        return outdata

    def calculate_net(self):
        netdata = numpy.dot(self.__inputdata, self.__weight)
        return netdata

    def calculate_out(self):
        return self.excitation(netdata=self.calculate_net())

    def update_weight(self, data):
        self.__weight = data

    def get_weight(self):
        return self.__weight

    def update_inputs(self, new_input):
        self.__inputdata = new_input

    def get_inputs(self):
        return self.__inputdata

    def network_result_error(self, target_result):
        error = 0.5 * [( target_result - self.__outdata ) ** 2]
        return error

    def calculate_pd_error_out(self, outdata, target_data):
        x = sympy.symbols("x")
        y = 0.5 * [( target_data - x ) ** 2]
        error_out_dify = sympy.diff(y, x).subs({x: outdata})
        return error_out_dify

    def calculate_pd_out_net(self, netdata):
        x = sympy.symbols("x")
        y = 1 / ( 1 + math.exp(-x))
        out_net_dify = sympy.diff(y, x).subs({x: netdata})
        return out_net_dify

    def calculate_pd_net_input(self, input_index):
        return self.__weight[input_index]

    def calculate_pd_net_weight(self, weight_index):
        return self.__inputdata[weight_index]

#python继承时 私有不可继承覆盖
class Layer():
    def __init__(self, neurons_num, input_num):
        self.neuronsnum = neurons_num
        self.inputnum = input_num
        self.input, self.neurons = self.create_neurons(input_num)

    def create_neurons(self, input_num):
        neurons = []
        inputs = []
        for i in range(input_num):
            inputs.append(0)
        for i in range(self.neuronsnum):
            weight = []
            for j in range(input_num):
                weight.append(random.random())
            neuron = Neuron(inputs, weight)
            neurons.append(neuron)
        return inputs, neurons

    def calculate_layer_outputs(self):
        outputs = []
        for i in (self.neuronsnum):
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

    def update_neurons_inputs(self, new_inputs):
        for i in range(self.neuronsnum):
            self.neurons[i].update_inputs(new_input=new_inputs)

    def get_neurons_inputs(self):
        return self.neurons[0].get_inputs()





class OutputLayer(Layer):

    def __init__(self, neurons_num, targets):
        Layer.__init__(neurons_num=neurons_num)
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
        print(error_pd_weight_outputlayer)
        return error_pd_weight_outputlayer

    def gradient_descent_outputweights(self, alpha):
        error_pd_net_outputlayer = self.calculate_pd_bp_outputlayer(targets=self.__targets)
        error_pd_weight_outputlayer = self.calculate_pd_bp_outputweights(error_pd_net_outputlayer=error_pd_net_outputlayer)
        new_weights = []
        for i in range(self.neuronsnum):
            thisneurons = self.neurons[i]
            thisweights = []
            for j in range(self.inputnum):
                new_weight = thisneurons.get_weight[j] - alpha * ((error_pd_weight_outputlayer[i])[j])
                thisweights.append(new_weight)
            new_weights.append(thisweights)
        self.update_neurons_weights(new_weights)

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
    pass









