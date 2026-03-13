package ai.onnxruntime;

import java.util.Map;
import java.util.Optional;

public class OrtEnvironment implements AutoCloseable {
    public static OrtEnvironment getEnvironment() {
        return new OrtEnvironment();
    }
    @Override
    public void close() {}
}

public class OrtSession implements AutoCloseable {
    public static class SessionOptions implements AutoCloseable {
        @Override
        public void close() {}
    }

    public static class Result implements AutoCloseable {
        @Override
        public void close() {}
    }

    @Override
    public void close() {}
}

public class OnnxTensor implements AutoCloseable {
    @Override
    public void close() {}
}

public class OnnxSequence implements AutoCloseable {
    @Override
    public void close() {}
}

public class OnnxMap implements AutoCloseable {
    @Override
    public void close() {}
}

class JNIBridges {}
