using System;

namespace Neural {
    public class NeuronLayer {
        public Neuron this[int index]
        {
            get => neurons[index];
        }
        // return length with displacement neuron
        public int Lenght { get => neurons.Length + 1; }
        public bool isOutput = false;
        protected Neuron[] neurons;

        /// <summary>
        /// Neuron layer with array of neurons
        /// </summary>
        /// <param name="neuronsCount"></param>
        public NeuronLayer(int neuronsCount, int previousLayerNeuronCount, Func<float,float> activateFunc, Func<float,float> derivFunc) {
            neurons = new Neuron[neuronsCount];

            for (int i = 0; i < neurons.Length; i++) {
                neurons[i] = new Neuron(previousLayerNeuronCount, activateFunc, derivFunc);
            }
        }

        /// <summary>
        /// Feed all neurons and return output
        /// </summary>
        /// <param name="input">input data for neurons</param>
        /// <returns>output of feeded neurons</returns>
        public float[] Feed (float[] input) {
            // array with result of neurons output
            int count = isOutput ? neurons.Length : neurons.Length + 1;
            float[] result = new float[count];

            for (int i = 0; i < neurons.Length; i++) {
                result[i] = neurons[i].Feed(input);
            }
            
            if (!isOutput)
                result[neurons.Length] = 1;

            return result;
        }
        private float Square(float x) {
            return (float)Math.Pow(x, 2);
        }

        public float[] CalculateOutputError(float[] result, float[] waitResult) {
            float[] data = new float[neurons.Length];

            for (int i = 0; i < neurons.Length; i++) {
                data[i] = neurons[i].CalculateError( Square(waitResult[i]) - Square(result[i]) );
            }

            return data;
        }
        public float[] GetWeightsFor (int index) {
            float[] data = new float[neurons.Length];

            for (int i = 0; i < neurons.Length; i++) {
                data[i] = neurons[i].GetWeightFor(index);
            }

            return data;
        }

        public float[] CalculateError(float[] loss, NeuronLayer nextLayer) {
            float[] result = new float[neurons.Length];

            for (int i = 0; i < neurons.Length; i++) {
                float[] weights = nextLayer.GetWeightsFor(i);

                result[i] = neurons[i].CalculateError(loss, weights);
            }

            return result;
        }

        public float[] TrainNeurons (float[] input, float learnRate) {

            for (int i = 0; i < neurons.Length; i++) {
                neurons[i].Train(input, learnRate);
            }

            return Feed(input);
        }
    }
}