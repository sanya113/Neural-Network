using System;
using DefaultActivateFuncs;

namespace Neural
{
    class Program
    {
        static float[,] nrInputData = {
            {
                1,1,1,
                0,0,0,
                0,0,0
            },
            {
                0,1,0,
                0,1,0,
                0,1,0
            },
            {
                1,1,1,
                0,0,1,
                0,0,1
            },
            {
                1,0,0,
                1,0,0,
                1,1,1
            },
            {
                0,0,0,
                1,1,1,
                0,0,0
            },
            {
                0,0,1,
                0,0,1,
                0,0,1
            },
            {
                1,0,0,
                1,1,1,
                1,0,0
            },
            {
                0,1,0,
                0,1,0,
                1,1,1
            },
            {
                0,0,0,
                0,0,0,
                1,1,1
            },
            {
                1,1,1,
                1,0,0,
                1,0,0
            },
            {
                0,1,0,
                1,1,1,
                0,1,0
            },
            {
                0,0,1,
                0,0,1,
                1,1,1
            },
            {
                1,0,0,
                1,0,0,
                1,0,0
            },
            {
                1,1,1,
                0,1,0,
                0,1,0
            },
            {
                0,0,1,
                1,1,1,
                0,0,1
            },
            {
                0,0,0,
                0,0,0,
                0,0,0
            },
        };
        static float[,] nrRightResult = {
            {0.8f,0.2f},
            {0.2f,0.8f},
            {0.8f,0.8f},
            {0.8f,0.8f},
            {0.8f,0.2f},
            {0.2f,0.8f},
            {0.8f,0.8f},
            {0.8f,0.8f},
            {0.8f,0.2f},
            {0.8f,0.8f},
            {0.8f,0.8f},
            {0.8f,0.8f},
            {0.2f,0.8f},
            {0.8f,0.8f},
            {0.8f,0.8f},
            {0.2f,0.2f}
        };
        static float[][] inputData;
        static float[][] rightResult;
        private static void ArrayToArrays(float[,] input, out float[][] output) {
            int rows = input.GetUpperBound(0) + 1;
            int columns = input.Length / rows;

            output = new float[rows][];

            for (int y = 0; y < rows; y++) {
                output[y] = new float[columns];
                for (int x = 0; x < columns; x++ ) {
                    output[y][x] = input[y, x];
                }
            }
        }
        static float Square (float x) {
            return (float)Math.Pow(x, 2);
        }
        static float Loss (float[][] y_true, float[][] y_res) {
            float result = 0;
            for (int i = 0; i < y_res.Length; i++) {
                for (int x = 0; x < y_res[i].Length; x++) {
                    result += Square(y_true[i][x] - y_res[i][x]);
                }
            }
            return result / y_true.Length;
        }

        static void Main(string[] args)
        {
            ArrayToArrays(nrInputData, out inputData);
            ArrayToArrays(nrRightResult, out rightResult);

            IActivate funcs = new Sigmoid();

            NeuronNetwork network = new NeuronNetwork(inputData[0].Length, funcs.activate, funcs.deriv);

            network.AddLayer(6);

            network.AddLayer(2);

            network.OptimizeSystem();

            Console.Clear();

            float loss = 1;
            int loop = 0;
            while (loop < 100000) {
                float[][] result = new float[inputData.Length][];
                for (int y = 0; y < inputData.Length; y++) {
                    result[y] = network.TrainNeurons(inputData[y], rightResult[y]);
                }

                if (loop % 100 == 0) {
                    loss = Loss(result, rightResult);
                    Console.SetCursorPosition(0,1);
                    Console.WriteLine(String.Format("Loop {0} loss {1:f8}", loop, loss));
                }

                loop += 1;
            }
            Console.WriteLine(loss);
            for (int i = 0; i < inputData.Length; i++){
                float[] need = inputData[i];
                
                string writeData = String.Format("Пример - {0}; Ожидаемый результать - {1}; Результат - {2:f2};", ArrayToString(need), ArrayToString( rightResult[i] ), OutputArrayString(network.Feed(need)));
                Console.WriteLine(writeData);
                
            }
        }
        static string OutputArrayString(float[] need) {
            
            string needStr = "";
            string addChar = " ";
            for (int j = 0;j < need.Length; j++) {
                if (j == need.Length - 1) {
                    addChar = "";
                }
                needStr += Math.Round(need[j],1) + addChar;
            }
            return needStr;
        } 
        static string ArrayToString(float[] need) {
            string needStr = "";
            for (int j = 0;j < need.Length; j++) {
                needStr += need[j] + " ";
            }
            return needStr;
        }
    }
}
