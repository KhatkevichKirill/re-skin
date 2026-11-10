/* Coordinate every writer of the Run status fragment, including legacy polls. */
(() => {
  'use strict';
  if (window.reskinRunStatus) return;
  let generation = 0, sequence = 0, appliedSequence = 0;
  const requests = new WeakMap();
  const isTarget = element => element && element.id === 'run-status-content';
  document.addEventListener('htmx:beforeRequest', event => {
    if (isTarget(event.detail.target)) requests.set(event.detail.xhr, {generation, sequence:++sequence});
  });
  document.addEventListener('htmx:beforeSwap', event => {
    if (!isTarget(event.detail.target)) return;
    const request = requests.get(event.detail.xhr);
    if (!request || request.generation !== generation || request.sequence < appliedSequence) {
      event.detail.shouldSwap = false;
      event.preventDefault();
    }
  });
  document.addEventListener('htmx:afterSwap', event => {
    if (!isTarget(event.detail.target)) return;
    const request = requests.get(event.detail.xhr);
    if (request && request.generation === generation && event.detail.xhr.status >= 200 && event.detail.xhr.status < 300) {
      appliedSequence = Math.max(appliedSequence, request.sequence);
    }
  });
  window.reskinRunStatus = {
    invalidate() { generation += 1; },
    refresh() {
      const target = document.getElementById('run-status-content');
      if (!target || !window.htmx) return Promise.reject(new Error('Run status refresh is unavailable'));
      return window.htmx.ajax('GET', '/v2/runs/' + encodeURIComponent(target.dataset.runId) + '/status-fragment',
        {source:target, target, swap:'innerHTML'});
    },
  };
})();
