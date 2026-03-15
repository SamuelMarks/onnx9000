// Step 342: Tensor class mapping to OrtValue
/** Implementation details and semantic operations. */
export type TypedArray = Float32Array | Int32Array | Uint32Array | Uint8Array | Int8Array | BigInt64Array | Float64Array;

/** Implementation details and semantic operations. */

export class Tensor {
    public readonly type: string;
    public readonly data: TypedArray | string[];
    public readonly dims: readonly number[];
    public readonly size: number;

    // Step 343: Tensor constructor supporting typed arrays
    /** Implementation details and semantic operations. */
    constructor(type: string, data: TypedArray | readonly number[] | readonly string[], dims?: readonly number[]) {
        this.type = type;
        
        let inferredSize = 1;
        /** Implementation details and semantic operations. */
        if(dims) {
            /** Implementation details and semantic operations. */
            for(let i = 0; i < dims.length; i++) {
                inferredSize *= dims[i];
            }
            this.dims = dims;
        } else {
            this.dims = [data.length];
            inferredSize = data.length;
        }

        this.size = inferredSize;

        /** Implementation details and semantic operations. */

        if(Array.isArray(data)) {
            /** Implementation details and semantic operations. */
            if(type === "string") {
                this.data = data as string[];
            } else if (type === "float32") {
                this.data = new Float32Array(data as number[]);
            } else if (type === "int32") {
                this.data = new Int32Array(data as number[]);
            } else if (type === "int64") {
                const bigIntData = (data as number[]).map(val => BigInt(val));
                this.data = new BigInt64Array(bigIntData);
            } else {
                throw new Error(`Unsupported tensor type: ${type}`);
            }
        } else {
            this.data = data as TypedArray;
        }

        /** Implementation details and semantic operations. */

        if(this.data.length !== this.size) {
            throw new Error(`Data size (${this.data.length}) does not match dimensions product (${this.size})`);
        }
    }
}
