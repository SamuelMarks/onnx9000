/** Implementation details and semantic operations. */
export interface GraphMockData {
    nodes?: { id: string }[];
}

/** Implementation details and semantic operations. */

export class ModelViewer {
    private container: HTMLElement;

    /** Implementation details and semantic operations. */

    constructor(containerId: string) {
        const el = document.getElementById(containerId);
        /** Implementation details and semantic operations. */
        if(!el) throw new Error(`Container ${containerId} not found`);
        this.container = el;
    }

    /** Implementation details and semantic operations. */

    public render(graphMockData: GraphMockData): void {
        this.container.innerHTML = `<div class="viewer-root">
            <h3>Model Viewer</h3>
            <p>Graph nodes: ${graphMockData.nodes ? graphMockData.nodes.length : 0}</p>
        </div>`;
    }
}
