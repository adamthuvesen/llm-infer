export function theaterGeometry(laneCount) {
  const width = 1000;
  const left = 196;
  const right = width - 40;
  const top = 70;
  const laneHeight = 88;
  const laneTop = top;
  const height = laneTop + laneCount * laneHeight + 24;
  return {
    width,
    height,
    left,
    right,
    frameLeft: 8,
    frameRight: width - 8,
    laneTop,
    laneHeight,
    gridTop: top - 8,
    gridBottom: laneTop + laneCount * laneHeight + 4,
  };
}
