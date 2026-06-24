export function appendEventRowContent(documentRef, row, event, label) {
  const sequence = documentRef.createElement("span");
  sequence.className = "event-sequence";
  sequence.textContent = String(event.sequence);

  const title = documentRef.createElement("strong");
  title.textContent = label;

  const step = documentRef.createElement("em");
  step.textContent = `step ${event.step}`;

  row.append(sequence, title, step);
}
