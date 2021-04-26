using System;
using System.Collections.Generic;

namespace Neural {
    public class NeuronNetwork {
        public List<NeuronLayer> layers;

        private Func<float,float> activate;
        private Func<float,float> deriv;

        private int inputCount;

        public NeuronNetwork (int inputCount, Func<float,float> activateFunc, Func<float,float> derivFunc) {
            activate = activateFunc;
            deriv = derivFunc;

            layers = new List<NeuronLayer>(2);

            this.inputCount = inputCount;
        }

        public void AddLayer(int neuronsCount) {
            // Get weights of previous layer;
            int lastLayer = layers.Count - 1;
            int prevWeights = lastLayer == -1 ? inputCount : layers[lastLayer].Lenght;

            AddLayerWithWeights(neuronsCount, prevWeights);
        }

        public void AddLayerWithWeights (int neuronsCount, int weights) {
            NeuronLayer layer = new NeuronLayer(neuronsCount, weights, activate, deriv);

            layers.Add(layer);
        }

        public float[] Feed (float[] input) {
            float[] data = input;
            
            for (int i = 0; i < layers.Count; i++) {
                data = layers[i].Feed(data);
            }

            return data;
        }

        public float[] TrainNeurons(float[] input, float[] waitResult) {

            float[] result = Feed(input);

            // Calculate loss for output neuron
            NeuronLayer lastLayer = layers[layers.Count - 1];
            float[] loss = lastLayer.CalculateOutputError(result, waitResult);

            // Calculate loss for every neuron
            for (int y = layers.Count - 2; y >= 0; y--) {
                loss = layers[y].CalculateError(loss, lastLayer);

                lastLayer = layers[y];
            }

            float[] data = input; 
            // Train neurons
            for (int y = 0; y < layers.Count; y++) {
                data = layers[y].TrainNeurons(data, 0.4f);
            }

            return Feed(input);
        }
    }
}