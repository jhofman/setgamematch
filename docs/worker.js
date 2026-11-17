/* Runs the card detector off the main thread so the page stays responsive. */
importScripts("setvision.js");

self.onmessage = function (e) {
  const { id, data, width, height } = e.data;
  try {
    const cards = SetVision.detectCards({ data, width, height });
    self.postMessage({ id, cards });
  } catch (err) {
    self.postMessage({ id, error: String(err) });
  }
};
