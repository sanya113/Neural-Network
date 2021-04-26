using System;

namespace DefaultActivateFuncs
{
    public class Sigmoid : IActivate 
    {
        public float activate(float x) {
            if (x < -45.0) return 0.0f;
            else if (x > 45.0) return 1.0f;
            else return (1.0f / (1.0f + (float)Math.Exp(-x)));
        } 
        public float deriv(float x) {
            float f = activate(x);
            return f * (1 - f);
        } 
    } 
}
