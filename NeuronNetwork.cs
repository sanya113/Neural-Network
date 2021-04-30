using System;
using System.Collections.Generic;
using Neural.Core;

namespace Neural {
    public class NeuronNetwork {
        public List<NeuronLayer> layers;

        private NeuronLayer head;
        private NeuronLayer output;

        private Func<float,float> activate;
        private Func<float,float> deriv;

        private int inputCount;

        /// <summary>
        /// Neuron network with learn system
        /// </summary>
        /// <param name="inputCount">Count of input neurons</param>
        /// <param name="activateFunc">Activation function for neurons</param>
        /// <param name="derivFunc">Derivative function for neuron based on activate function </param>
        public NeuronNetwork (int inputCount, Func<float,float> activateFunc, Func<float,float> derivFunc) {
            activate = activateFunc;
            deriv = derivFunc;

            layers = new List<NeuronLayer>(2);

            this.inputCount = inputCount;
        }

        /// <summary>
        /// Create and add to list new layer with constant count of neurons
        /// </summary>
        /// <param name="neuronsCount">constant count of neurons in new layer</param>
        public void AddLayer(int neuronsCount) {
            // Get weights of previous layer;
            int prevWeights = 0;
            if (output == null) {
                prevWeights = inputCount;
            } else {
                prevWeights = output.Lenght;
            }

            AddLayerWithWeights(neuronsCount, prevWeights);
        }

        /// <summary>
        /// Add specfig layer with self calculated wights
        /// WARNING!: Do do it, if you dont understand what are you doing
        /// </summary>
        /// <param name="neuronsCount">Count of input neurons</param>
        /// <param name="weights">Count of wights for all neurons</param>
        public void AddLayerWithWeights (int neuronsCount, int weights) {
            NeuronLayer layer = new NeuronLayer(neuronsCount, weights, activate, deriv);

            if (output != null) {
                output.next = layer;
                layer.prev = output;
            }

            if (head == null) head = layer;
            output = layer;
        }

        /// <summary>
        /// Feed all neuron layers and return result of output layers
        /// </summary>
        /// <param name="input">Input data for input neurons</param>
        /// <returns>result of output neurons</returns>
        public float[] Feed (float[] input) {
            float[] data = input;
            
            NeuronLayer layer = head; 
            while (layer != null) {
                data = layer.Feed(data);
                layer = layer.next;
            }

            return data;
        }

        /// <summary>
        /// Function for training neurons.
        /// </summary>
        /// <param name="input">Input data for input neurons</param>
        /// <param name="waitResult">Right result of output neurons (in last layer)</param>
        /// <returns>result of output neurons</returns>
        public float[] TrainNeurons(float[] input, float[] waitResult) {

            float[] result = Feed(input);

            // Calculate loss for output neuron
            NeuronLayer lastLayer = output;
            float[] loss = lastLayer.CalculateOutputError(result, waitResult);

            // Calculate loss for hidden layer and input layer neurons
            NeuronLayer layer = output.prev;
            while (layer != null) {
                loss = layer.CalculateError(loss, lastLayer);

                lastLayer = layer;
                layer = layer.prev;
            }

            // Train neurons
            float[] data = input;
            
            layer = head;
            while (layer != null) {
                data = layer.TrainNeurons(data, 0.4f);
                layer = layer.next;
            }

            return Feed(input);
        }
    }
}