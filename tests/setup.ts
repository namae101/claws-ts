import '@tensorflow/tfjs-backend-cpu';

if (!('self' in globalThis)) {
  Object.defineProperty(globalThis, 'self', {
    value: globalThis,
    writable: true,
    configurable: true
  });
}

if (!('location' in globalThis)) {
  Object.defineProperty(globalThis, 'location', {
    value: {
      origin: 'http://localhost',
      href: 'http://localhost/'
    },
    writable: true,
    configurable: true
  });
}
