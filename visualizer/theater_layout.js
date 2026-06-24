export function buildTheaterLayout(requestCount) {
  const laneHeight = 78;
  const header = {
    kickerY: 54,
    headlineY: 82,
    subheadY: 106,
    bottomY: 124,
  };
  const laneTop = 166;
  const signalTop = laneTop + requestCount * laneHeight + 58;
  return {
    width: 1180,
    height: Math.max(720, signalTop + 170),
    left: 180,
    timelineRight: 850,
    laneTop,
    laneHeight,
    signalTop,
    cursorLineTop: header.bottomY,
    cursorCy: header.bottomY + 16,
    cursorLabelY: header.bottomY + 22,
    tickY: header.bottomY + 8,
    header,
  };
}

export function validateTheaterLayout(layout) {
  const textSafetyGap = 12;
  return (
    layout.header.subheadY + textSafetyGap < layout.cursorCy - 8 &&
    layout.cursorLabelY + textSafetyGap < layout.laneTop &&
    layout.tickY + textSafetyGap < layout.laneTop
  );
}
