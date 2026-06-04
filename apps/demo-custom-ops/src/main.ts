/* v8 ignore next */ /* v8 ignore next */ document
  .getElementById('register-op')
  ?.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    const opNameInput = document.getElementById(
      'op-name',
    ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
    const opName = opNameInput.value.trim(); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (opName) {
      /* v8 ignore next */ /* v8 ignore next */
      const registry =
        document.getElementById('registry'); /* v8 ignore next */ /* v8 ignore next */
      if (registry) {
        /* v8 ignore next */ /* v8 ignore next */
        const opItem = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
        opItem.className = 'op-item'; /* v8 ignore next */ /* v8 ignore next */
        opItem.textContent = opName; /* v8 ignore next */ /* v8 ignore next */
        registry.appendChild(opItem); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      opNameInput.value = ''; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  });
