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

        public NeuronNetwork (int inputCount, Func<float,float> activateFunc, Func<float,float> derivFunc) {
            activate = activateFunc;
            deriv = derivFunc;

            layers = new List<NeuronLayer>(2);

            this.inputCount = inputCount;
        }

        public void OptimizeSystem() {
            GC.Collect(0, GCCollectionMode.Optimized);
            GC.WaitForPendingFinalizers();
        }

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

        public void AddLayerWithWeights (int neuronsCount, int weights) {
            NeuronLayer layer = new NeuronLayer(neuronsCount, weights, activate, deriv);

            if (output != null) {
                output.next = layer;
                layer.prev = output;
            }

            if (head == null) head = layer;
            output = layer;
        }

        public float[] Feed (float[] input) {
            float[] data = input;
            
            NeuronLayer layer = head; 
            while (layer != null) {
                data = layer.Feed(data);
                layer = layer.next;
            }

            return data;
        }

        public float[] TrainNeurons(float[] input, float[] waitResult) {

            float[] result = Feed(input);

            // Calculate loss for output neuron
            NeuronLayer lastLayer = output;
            float[] loss = lastLayer.CalculateOutputError(result, waitResult);

            NeuronLayer layer = output.prev;
            while (layer != null) {
                loss = layer.CalculateError(loss, lastLayer);

                lastLayer = layer;
                layer = layer.prev;
            }

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