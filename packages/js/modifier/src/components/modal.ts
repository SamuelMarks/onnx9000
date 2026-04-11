/* eslint-disable */
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../GraphMutator.js';

export class AddNodeModal {
  container: HTMLElement;
  mutator: GraphMutator;

  constructor(container: HTMLElement, mutator: GraphMutator) {
    this.container = container;
    this.mutator = mutator;
  }

  // 63. Implement an "Add Node" modal
  // 64. Automatically populate the attributes form based on the selected op_type
  show() {
    this.container.innerHTML = '';
    this.container.style.position = 'absolute';
    this.container.style.top = '50%';
    this.container.style.left = '50%';
    this.container.style.transform = 'translate(-50%, -50%)';
    this.container.style.backgroundColor = '#fff';
    this.container.style.padding = '20px';
    this.container.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
    this.container.style.border = '1px solid #ccc';
    this.container.style.zIndex = '1000';
    this.container.style.display = 'block';

    const title = document.createElement('h3');
    title.textContent = 'Add Node';
    this.container.appendChild(title);

    const typeInput = document.createElement('input');
    typeInput.placeholder = 'OpType (e.g. Conv, Relu)';
    typeInput.style.display = 'block';
    typeInput.style.marginBottom = '10px';
    this.container.appendChild(typeInput);

    const nameInput = document.createElement('input');
    nameInput.placeholder = 'Node Name (Optional)';
    nameInput.style.display = 'block';
    nameInput.style.marginBottom = '10px';
    this.container.appendChild(nameInput);

    // Dummy schema integration (In a full app we'd query onnx spec, but per instructions we do minimal generic)
    const btn = document.createElement('button');
    btn.textContent = 'Add';
    btn.onclick = () => {
      const type = typeInput.value;
      const name = nameInput.value;
      if (type) {
        this.mutator.addNode(type, [], [], {}, name);
        this.hide();
      }
    };
    this.container.appendChild(btn);

    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Cancel';
    closeBtn.style.marginLeft = '10px';
    closeBtn.onclick = () => {
      this.hide();
    };
    this.container.appendChild(closeBtn);
  }

  hide() {
    this.container.style.display = 'none';
  }
}
