namespace Microsoft.ML.OnnxRuntime
{
    public class SessionOptions : System.IDisposable
    {
        public void Dispose() {}
    }

    public class RunOptions : System.IDisposable
    {
        public void Dispose() {}
    }

    public class OrtValue : System.IDisposable
    {
        public void Dispose() {}
    }

    public class NamedOnnxValue
    {
        public static NamedOnnxValue CreateFromTensor<T>(string name, T value)
        {
            return new NamedOnnxValue();
        }
    }

    public class InferenceSession : System.IDisposable
    {
        public InferenceSession(string modelPath) {}
        public InferenceSession(string modelPath, SessionOptions options) {}
        
        public System.IDisposable Run(System.Collections.Generic.IReadOnlyCollection<NamedOnnxValue> inputs)
        {
            return new OrtValue();
        }

        public void Dispose() {}
    }
}
