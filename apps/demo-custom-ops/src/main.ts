document.getElementById('register-op')?.addEventListener('click', () => {
  const opNameInput = document.getElementById('op-name') as HTMLInputElement;
  const opName = opNameInput.value.trim();

  if (opName) {
    const registry = document.getElementById('registry');
    if (registry) {
      const opItem = document.createElement('div');
      opItem.className = 'op-item';
      opItem.textContent = opName;
      registry.appendChild(opItem);
    }
    opNameInput.value = '';
  }
});
