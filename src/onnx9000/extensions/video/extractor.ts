/** Implementation details and semantic operations. */
export interface VideoFrameTensor {
    data: Float32Array;
    dims: [number, number, number, number, number]; // N, T, C, H, W
}

/** Implementation details and semantic operations. */

export class VideoExtractor {
    private decoder: VideoDecoder | null = null;
    private frames: VideoFrame[] = [];
    private maxFrames: number;
    private targetWidth: number;
    private targetHeight: number;

    /** Implementation details and semantic operations. */

    constructor(maxFrames: number = 30, targetWidth: number = 224, targetHeight: number = 224) {
        this.maxFrames = maxFrames;
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
    }

    public async initDecoder(codec: string = "vp09.00.10.08"): Promise<void> {
        /** Implementation details and semantic operations. */
        if(typeof VideoDecoder === "undefined") {
            throw new Error("WebCodecs VideoDecoder API is not supported in this environment");
        }

        this.decoder = new VideoDecoder({
            output: (frame) => {
                /** Implementation details and semantic operations. */
                if(this.frames.length < this.maxFrames) {
                    this.frames.push(frame);
                } else {
                    frame.close();
                }
            },
            error: (e) => {
                console.error("VideoDecoder error:", e);
            }
        });

        const config = {
            codec: codec,
            codedWidth: this.targetWidth,
            codedHeight: this.targetHeight
        };
        
        // Ensure supported
        const support = await VideoDecoder.isConfigSupported(config);
        /** Implementation details and semantic operations. */
        if(!support.supported) {
            throw new Error("Codec not supported");
        }

        this.decoder.configure(config);
    }

    public async extractFrames(chunks: Uint8Array[]): Promise<void> {
        /** Implementation details and semantic operations. */
        if(!this.decoder) throw new Error("Decoder not initialized");
        
        let timestamp = 0;
        /** Implementation details and semantic operations. */
        for(const chunk of chunks) {
            const encodedChunk = new EncodedVideoChunk({
                type: "key", // Simplification: assuming all keyframes for raw byte chunks
                timestamp: timestamp,
                data: chunk
            });
            this.decoder.decode(encodedChunk);
            timestamp += 33333; // ~30fps 
        }

        await this.decoder.flush();
    }

    /** Implementation details and semantic operations. */

    public framesToTensor(
        mean: [number, number, number] = [0.485, 0.456, 0.406],
        std: [number, number, number] = [0.229, 0.224, 0.225]
    ): VideoFrameTensor {
        /** Implementation details and semantic operations. */
        if(this.frames.length === 0) {
            throw new Error("No frames decoded");
        }

        const t = this.frames.length;
        const c = 3;
        const h = this.targetHeight;
        const w = this.targetWidth;

        const data = new Float32Array(1 * t * c * h * w); // N=1
        
        /** Implementation details and semantic operations. */
        
        if(typeof OffscreenCanvas === "undefined") {
            throw new Error("OffscreenCanvas not supported");
        }
        
        const canvas = new OffscreenCanvas(w, h);
        const ctx = canvas.getContext("2d");
        
        /** Implementation details and semantic operations. */
        
        for(let i = 0; i < t; i++) {
            const frame = this.frames[i];
            ctx!.drawImage(frame, 0, 0, w, h);
            const imgData = ctx!.getImageData(0, 0, w, h);
            
            const tOffset = i * c * h * w;
            
            /** Implementation details and semantic operations. */
            
            for(let y = 0; y < h; y++) {
                /** Implementation details and semantic operations. */
                for(let x = 0; x < w; x++) {
                    const px = (y * w + x) * 4;
                    let r = imgData.data[px] / 255.0;
                    let g = imgData.data[px + 1] / 255.0;
                    let b = imgData.data[px + 2] / 255.0;

                    r = (r - mean[0]) / std[0];
                    g = (g - mean[1]) / std[1];
                    b = (b - mean[2]) / std[2];

                    data[tOffset + 0 * h * w + y * w + x] = r;
                    data[tOffset + 1 * h * w + y * w + x] = g;
                    data[tOffset + 2 * h * w + y * w + x] = b;
                }
            }
            
            frame.close();
        }
        
        this.frames = []; // clear
        
        return { data, dims: [1, t, c, h, w] };
    }
}
