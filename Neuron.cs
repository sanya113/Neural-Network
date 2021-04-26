using System;
namespace Neural.Core {
    public class Neuron {
        private static Random rnd = new Random();

        private Func<float,float> activate;
        private Func<float,float> deriv;

        private float[] weights;

        private float result;
        private float loss;

        /// <summary>
        /// Neuron constructor
        /// </summary>
        /// <param name="weightsCount"> a count of weights based on previous layer</param>
        /// <param name="activateFunc">a neuron activate function; in - float, output - float</param>
        /// <param name="derivFunc">a neuron deriv function; in - float, output - float</param>
        public Neuron (int weightsCount, Func<float,float> activateFunc, Func<float,float> derivFunc) {
            activate = activateFunc;
            deriv = derivFunc;
            
            weights = new float[weightsCount];
            for (int i = 0; i < weightsCount;i++) {
                weights[i] = (float)rnd.NextDouble() * 2 - 1;
            }
        }

        /// <summary>
        /// Feed a neuron; accept input and return activated result
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public float Feed (float[] input) {
            if (weights.Length != input.Length) throw new IndexOutOfRangeException("weights length and input length must be equal");
            result = 0;
            for (int i = 0; i < weights.Length; i++) {
                result += input[i] * weights[i];
            }
            return activate(result);
        }

        public float GetWeightFor (int index) {
            return weights[index];
        }

        public float CalculateError(float error) {
            loss = error * deriv(result);
            return loss;
        }
        public float CalculateError (float[] inputLoss, float[] weights) {
            if (inputLoss.Length != weights.Length) throw new Exception("Cant calculate loss with defferent lenght of input loss and next layer weights");
			loss = 0;
            for (int i = 0; i < inputLoss.Length; i++) {
                loss += inputLoss[i] * weights[i];
            }
            loss *= deriv(result);
            return loss;
        }

        public void Train (float[] inputData, float learnRate) {
            if (inputData.Length != weights.Length) throw new Exception("Cant train neuron with defferent length of input data and neuron weights");
            for (int i = 0; i < inputData.Length; i++) {
				weights[i] += learnRate * loss * inputData[i];
            }
        }
    }
}