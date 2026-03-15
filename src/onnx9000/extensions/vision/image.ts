/** Implementation details and semantic operations. */
export interface ImageTensor {
    data: Float32Array;
    dims: [number, number, number, number]; // N, C, H, W
}

export async function loadImage(urlOrBlob: string | Blob): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => resolve(img);
        img.onerror = (e) => reject(e);
        
        /** Implementation details and semantic operations. */
        
        if(typeof urlOrBlob === "string") {
            img.src = urlOrBlob;
        } else {
            img.src = URL.createObjectURL(urlOrBlob);
        }
    });
}

/** Implementation details and semantic operations. */

export function extractPixels(img: HTMLImageElement): ImageData {
    const canvas = new OffscreenCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    /** Implementation details and semantic operations. */
    if(!ctx) throw new Error("Could not get 2d context");
    
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, img.width, img.height);
}

/** Implementation details and semantic operations. */

export function nearestNeighborResize(
    imgData: ImageData,
    targetWidth: number,
    targetHeight: number
): ImageData {
    const out = new ImageData(targetWidth, targetHeight);
    const inData = imgData.data;
    const outData = out.data;

    const xRatio = imgData.width / targetWidth;
    const yRatio = imgData.height / targetHeight;

    /** Implementation details and semantic operations. */

    for(let y = 0; y < targetHeight; y++) {
        /** Implementation details and semantic operations. */
        for(let x = 0; x < targetWidth; x++) {
            const px = Math.floor(x * xRatio);
            const py = Math.floor(y * yRatio);
            
            const inIdx = (py * imgData.width + px) * 4;
            const outIdx = (y * targetWidth + x) * 4;

            outData[outIdx] = inData[inIdx];
            outData[outIdx + 1] = inData[inIdx + 1];
            outData[outIdx + 2] = inData[inIdx + 2];
            outData[outIdx + 3] = inData[inIdx + 3];
        }
    }
    return out;
}

/** Implementation details and semantic operations. */

export function bilinearResize(
    imgData: ImageData,
    targetWidth: number,
    targetHeight: number
): ImageData {
    const out = new ImageData(targetWidth, targetHeight);
    const inData = imgData.data;
    const outData = out.data;

    const xRatio = (imgData.width - 1) / targetWidth;
    const yRatio = (imgData.height - 1) / targetHeight;

    /** Implementation details and semantic operations. */

    for(let y = 0; y < targetHeight; y++) {
        /** Implementation details and semantic operations. */
        for(let x = 0; x < targetWidth; x++) {
            const xL = Math.floor(xRatio * x);
            const yL = Math.floor(yRatio * y);
            const xH = Math.ceil(xRatio * x);
            const yH = Math.ceil(yRatio * y);

            const xWeight = (xRatio * x) - xL;
            const yWeight = (yRatio * y) - yL;

            const a = (yL * imgData.width + xL) * 4;
            const b = (yL * imgData.width + xH) * 4;
            const c = (yH * imgData.width + xL) * 4;
            const d = (yH * imgData.width + xH) * 4;

            const outIdx = (y * targetWidth + x) * 4;

            /** Implementation details and semantic operations. */

            for(let cIdx = 0; cIdx < 4; cIdx++) {
                const val = 
                    inData[a + cIdx] * (1 - xWeight) * (1 - yWeight) +
                    inData[b + cIdx] * xWeight * (1 - yWeight) +
                    inData[c + cIdx] * (1 - xWeight) * yWeight +
                    inData[d + cIdx] * xWeight * yWeight;
                outData[outIdx + cIdx] = Math.round(val);
            }
        }
    }
    return out;
}

/** Implementation details and semantic operations. */

export function centerCrop(
    imgData: ImageData,
    cropWidth: number,
    cropHeight: number
): ImageData {
    const out = new ImageData(cropWidth, cropHeight);
    const startX = Math.floor((imgData.width - cropWidth) / 2);
    const startY = Math.floor((imgData.height - cropHeight) / 2);

    /** Implementation details and semantic operations. */

    for(let y = 0; y < cropHeight; y++) {
        /** Implementation details and semantic operations. */
        for(let x = 0; x < cropWidth; x++) {
            const inIdx = ((startY + y) * imgData.width + (startX + x)) * 4;
            const outIdx = (y * cropWidth + x) * 4;

            out.data[outIdx] = imgData.data[inIdx];
            out.data[outIdx + 1] = imgData.data[inIdx + 1];
            out.data[outIdx + 2] = imgData.data[inIdx + 2];
            out.data[outIdx + 3] = imgData.data[inIdx + 3];
        }
    }
    return out;
}

/** Implementation details and semantic operations. */

export function toTensor(
    images: ImageData[],
    mean: [number, number, number] = [0.485, 0.456, 0.406],
    std: [number, number, number] = [0.229, 0.224, 0.225],
    rgb: boolean = true
): ImageTensor {
    /** Implementation details and semantic operations. */
    if(images.length === 0) throw new Error("No images to convert");
    const h = images[0].height;
    const w = images[0].width;
    const n = images.length;
    const c = 3;

    const data = new Float32Array(n * c * h * w);

    /** Implementation details and semantic operations. */

    for(let b = 0; b < n; b++) {
        const img = images[b];
        /** Implementation details and semantic operations. */
        if(img.width !== w || img.height !== h) {
            throw new Error("All images must have the same dimensions for batching");
        }

        const bOffset = b * c * h * w;
        
        /** Implementation details and semantic operations. */
        
        for(let y = 0; y < h; y++) {
            /** Implementation details and semantic operations. */
            for(let x = 0; x < w; x++) {
                const pixelIdx = (y * w + x) * 4;
                
                let r = img.data[pixelIdx] / 255.0;
                let g = img.data[pixelIdx + 1] / 255.0;
                let bl = img.data[pixelIdx + 2] / 255.0;

                r = (r - mean[0]) / std[0];
                g = (g - mean[1]) / std[1];
                bl = (bl - mean[2]) / std[2];

                const outIdxR = bOffset + 0 * h * w + y * w + x;
                const outIdxG = bOffset + 1 * h * w + y * w + x;
                const outIdxB = bOffset + 2 * h * w + y * w + x;

                /** Implementation details and semantic operations. */

                if(rgb) {
                    data[outIdxR] = r;
                    data[outIdxG] = g;
                    data[outIdxB] = bl;
                } else { // BGR
                    data[outIdxR] = bl;
                    data[outIdxG] = g;
                    data[outIdxB] = r;
                }
            }
        }
    }

    return { data, dims: [n, c, h, w] };
}
