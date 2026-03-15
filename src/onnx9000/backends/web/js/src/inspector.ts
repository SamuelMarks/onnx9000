import { Tensor } from "./tensor";

/** Implementation details and semantic operations. */

export class TensorInspector {
    private container: HTMLElement;

    /** Implementation details and semantic operations. */

    constructor(containerId: string) {
        const el = document.getElementById(containerId);
        /** Implementation details and semantic operations. */
        if(!el) throw new Error(`Container ${containerId} not found`);
        this.container = el;
    }

    /** Implementation details and semantic operations. */

    public inspect(tensor: Tensor): void {
        this.container.innerHTML = `<div class="inspector">
            <h4>Tensor [${tensor.type}]</h4>
            <p>Dims: [${tensor.dims.join(', ')}]</p>
            <p>Size: ${tensor.size}</p>
        </div>`;
    }
}
