/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as d3 from "d3";

/**
 * A multi-dimensional example with arbitrary features and a label.
 */
export type ExampleND = {
  features: number[];
  label: number;
};

/**
 * A two dimensional example: x and y coordinates with the label.
 * Legacy type for backward compatibility.
 */
export type Example2D = {
  x: number;
  y: number;
  label: number;
};

type Point = {
  x: number;
  y: number;
};

/**
 * Label encoding mappings for text columns.
 */
export interface LabelEncodings {
  [columnName: string]: {
    [textValue: string]: number;
  };
}

/**
 * Dataset metadata for multi-dimensional datasets.
 */
export interface DatasetInfo {
  numFeatures: number;
  featureNames: string[];
  isClassification: boolean;
  numClasses: number;
  classLabels: number[];
  labelEncodings?: LabelEncodings;
  encodedColumns?: string[];
}

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  var counter = array.length;
  var temp = 0;
  var index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

/**
 * Convert Example2D to ExampleND.
 */
export function example2DToND(ex: Example2D): ExampleND {
  return {
    features: [ex.x, ex.y],
    label: ex.label,
  };
}

/**
 * Convert ExampleND to Example2D (only works for 2D data).
 */
export function exampleNDTo2D(ex: ExampleND): Example2D {
  if (ex.features.length < 2) {
    throw new Error(
      "ExampleND must have at least 2 features to convert to Example2D"
    );
  }
  return {
    x: ex.features[0],
    y: ex.features[1],
    label: ex.label,
  };
}

/**
 * Convert array of Example2D to ExampleND.
 */
export function convertToND(data: Example2D[]): ExampleND[] {
  return data.map(example2DToND);
}

/**
 * Convert array of ExampleND to Example2D.
 */
export function convertTo2D(data: ExampleND[]): Example2D[] {
  return data.map(exampleNDTo2D);
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];
export type DataGeneratorND = (
  numSamples: number,
  noise: number
) => ExampleND[];

/**
 * Check if a value is numeric.
 */
function isNumeric(value: string): boolean {
  if (value === null || value === undefined || value.trim() === "") {
    return false;
  }
  return !isNaN(parseFloat(value)) && isFinite(Number(value));
}

/**
 * Parse a CSV line handling quoted values.
 */
function parseCSVLine(line: string): string[] {
  var result: string[] = [];
  var current = "";
  var inQuotes = false;
  var i, char;

  for (i = 0; i < line.length; i++) {
    char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current.trim());

  return result;
}

/**
 * Detect which columns contain text (non-numeric) data.
 */
function detectTextColumns(lines: string[], numColumns: number): boolean[] {
  var isTextColumn: boolean[] = new Array(numColumns);
  var i, j, line, values, val, rowsToCheck;

  for (i = 0; i < numColumns; i++) {
    isTextColumn[i] = false;
  }

  // Check first 100 data rows (or all if fewer)
  rowsToCheck = Math.min(lines.length, 100);

  for (i = 1; i < rowsToCheck; i++) {
    line = lines[i].trim();
    if (line === "") continue;

    values = parseCSVLine(line);
    for (j = 0; j < numColumns && j < values.length; j++) {
      val = values[j].trim();
      if (val !== "" && !isNumeric(val)) {
        isTextColumn[j] = true;
      }
    }
  }

  return isTextColumn;
}

/**
 * Build label encoding mappings for text columns.
 */
function buildLabelEncodings(
  lines: string[],
  headers: string[],
  isTextColumn: boolean[]
): LabelEncodings {
  var encodings: LabelEncodings = {};
  var uniqueValues: { [col: string]: { [val: string]: boolean } } = {};
  var i, j, line, values, val, col, sortedValues, k, keys;

  // Initialize encodings for text columns
  for (j = 0; j < headers.length; j++) {
    if (isTextColumn[j]) {
      encodings[headers[j]] = {};
      uniqueValues[headers[j]] = {};
    }
  }

  // Collect unique values for each text column
  for (i = 1; i < lines.length; i++) {
    line = lines[i].trim();
    if (line === "") continue;

    values = parseCSVLine(line);
    for (j = 0; j < headers.length && j < values.length; j++) {
      if (isTextColumn[j]) {
        val = values[j].trim();
        if (val !== "") {
          uniqueValues[headers[j]][val] = true;
        }
      }
    }
  }

  // Assign numeric labels (sorted for consistency)
  for (col in uniqueValues) {
    if (uniqueValues.hasOwnProperty(col)) {
      keys = Object.keys(uniqueValues[col]);
      sortedValues = keys.sort();
      for (k = 0; k < sortedValues.length; k++) {
        encodings[col][sortedValues[k]] = k;
      }
    }
  }

  return encodings;
}

/**
 * Parse CSV data into ExampleND array with label encoding for text columns.
 * Expects last column to be the label.
 *
 * @param csvText The raw CSV text content.
 * @return Object containing the parsed data and dataset info.
 */
export function parseCSV(csvText: string): {
  data: ExampleND[];
  info: DatasetInfo;
} {
  var lines = csvText.trim().split("\n");
  var headers, numFeatures, featureNames, labelColumnName;
  var isTextColumn, labelEncodings, encodedColumns;
  var data: ExampleND[] = [];
  var labels: { [key: number]: boolean } = {};
  var i, j, line, values, features, skipRow, val, encoding, numVal;
  var labelVal, label, isClassification, classLabels, labelKeys;

  if (lines.length < 2) {
    throw new Error("CSV must have at least a header row and one data row");
  }

  headers = parseCSVLine(lines[0]);
  numFeatures = headers.length - 1;
  featureNames = headers.slice(0, numFeatures);
  labelColumnName = headers[numFeatures];

  // Detect text columns
  isTextColumn = detectTextColumns(lines, headers.length);

  // Build label encodings for text columns
  labelEncodings = buildLabelEncodings(lines, headers, isTextColumn);

  // Track which columns were encoded
  encodedColumns = [];
  for (j = 0; j < headers.length; j++) {
    if (isTextColumn[j]) {
      encodedColumns.push(headers[j]);
    }
  }

  for (i = 1; i < lines.length; i++) {
    line = lines[i].trim();
    if (line === "") continue;

    values = parseCSVLine(line);
    if (values.length !== headers.length) {
      console.warn(
        "Skipping row " +
          i +
          ": expected " +
          headers.length +
          " columns, got " +
          values.length
      );
      continue;
    }

    features = [];
    skipRow = false;

    for (j = 0; j < numFeatures; j++) {
      val = values[j].trim();

      if (isTextColumn[j]) {
        // Use label encoding
        encoding = labelEncodings[headers[j]];
        if (val in encoding) {
          features.push(encoding[val]);
        } else {
          console.warn(
            'Unknown value "' +
              val +
              '" in column "' +
              headers[j] +
              '" at row ' +
              i
          );
          skipRow = true;
          break;
        }
      } else {
        // Parse as number
        numVal = parseFloat(val);
        if (isNaN(numVal)) {
          console.warn(
            'Invalid number "' +
              val +
              '" in column "' +
              headers[j] +
              '" at row ' +
              i
          );
          skipRow = true;
          break;
        }
        features.push(numVal);
      }
    }

    if (skipRow) continue;

    // Handle label column
    labelVal = values[numFeatures].trim();

    if (isTextColumn[numFeatures]) {
      // Label is text - use encoding
      encoding = labelEncodings[labelColumnName];
      if (labelVal in encoding) {
        label = encoding[labelVal];
      } else {
        console.warn('Unknown label "' + labelVal + '" at row ' + i);
        continue;
      }
    } else {
      label = parseFloat(labelVal);
      if (isNaN(label)) {
        console.warn('Invalid label "' + labelVal + '" at row ' + i);
        continue;
      }
    }

    labels[label] = true;
    data.push({ features: features, label: label });
  }

  // Determine if classification or regression
  labelKeys = Object.keys(labels);
  isClassification = labelKeys.length <= 20;
  classLabels = [];
  for (i = 0; i < labelKeys.length; i++) {
    classLabels.push(parseFloat(labelKeys[i]));
  }
  classLabels.sort(function (a, b) {
    return a - b;
  });

  return {
    data: data,
    info: {
      numFeatures: numFeatures,
      featureNames: featureNames,
      isClassification: isClassification,
      numClasses: isClassification ? classLabels.length : 0,
      classLabels: classLabels,
      labelEncodings: encodedColumns.length > 0 ? labelEncodings : undefined,
      encodedColumns: encodedColumns.length > 0 ? encodedColumns : undefined,
    },
  };
}

/**
 * Normalize features to [-1, 1] range.
 */
export function normalizeData(data: ExampleND[]): ExampleND[] {
  if (data.length === 0) return data;

  var numFeatures = data[0].features.length;
  var mins: number[] = [];
  var maxs: number[] = [];
  var i, j, val, range, features;
  var normalized: ExampleND[] = [];

  // Initialize mins and maxs
  for (j = 0; j < numFeatures; j++) {
    mins.push(Infinity);
    maxs.push(-Infinity);
  }

  // Find min and max for each feature
  for (i = 0; i < data.length; i++) {
    for (j = 0; j < numFeatures; j++) {
      val = data[i].features[j];
      if (val < mins[j]) mins[j] = val;
      if (val > maxs[j]) maxs[j] = val;
    }
  }

  // Normalize to [-1, 1]
  for (i = 0; i < data.length; i++) {
    features = [];
    for (j = 0; j < numFeatures; j++) {
      range = maxs[j] - mins[j];
      if (range === 0) {
        features.push(0);
      } else {
        features.push((2 * (data[i].features[j] - mins[j])) / range - 1);
      }
    }
    normalized.push({ features: features, label: data[i].label });
  }

  return normalized;
}

/**
 * Normalize labels for classification.
 * Binary: maps to -1 and 1
 * Multi-class: maps to 0, 1, 2, ..., n-1
 */
export function normalizeLabelsForClassification(
  data: ExampleND[],
  info: DatasetInfo
): ExampleND[] {
  var labelMap: { [key: number]: number } = {};
  var i;

  if (info.numClasses === 2) {
    // Binary classification: map to -1 and 1
    labelMap[info.classLabels[0]] = -1;
    labelMap[info.classLabels[1]] = 1;
  } else {
    // Multi-class: map to 0, 1, 2, ..., n-1
    for (i = 0; i < info.classLabels.length; i++) {
      labelMap[info.classLabels[i]] = i;
    }
  }

  return data.map(function (ex) {
    return {
      features: ex.features.slice(),
      label: labelMap[ex.label],
    };
  });
}

/**
 * Normalize labels for regression to [-1, 1] range.
 */
export function normalizeLabelsForRegression(data: ExampleND[]): ExampleND[] {
  var minLabel = Infinity;
  var maxLabel = -Infinity;
  var i, range;

  for (i = 0; i < data.length; i++) {
    if (data[i].label < minLabel) minLabel = data[i].label;
    if (data[i].label > maxLabel) maxLabel = data[i].label;
  }

  range = maxLabel - minLabel;
  if (range === 0) {
    return data.map(function (ex) {
      return { features: ex.features.slice(), label: 0 };
    });
  }

  return data.map(function (ex) {
    return {
      features: ex.features.slice(),
      label: (2 * (ex.label - minLabel)) / range - 1,
    };
  });
}

// ============ Original 2D Dataset Generators ============

export function classifyTwoGaussData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let varianceScale = d3.scale.linear().domain([0, 0.5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({ x, y, label });
    }
  }

  genGauss(2, 2, 1); // Gaussian with positive examples.
  genGauss(-2, -2, -1); // Gaussian with negative examples.
  return points;
}

export function regressPlane(numSamples: number, noise: number): Example2D[] {
  let radius = 6;
  let labelScale = d3.scale.linear().domain([-10, 10]).range([-1, 1]);
  let getLabel = (x, y) => labelScale(x + y);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({ x, y, label });
  }
  return points;
}

export function regressGaussian(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let labelScale = d3.scale.linear().domain([0, 2]).range([1, 0]).clamp(true);

  let gaussians = [
    [-4, 2.5, 1],
    [0, 2.5, -1],
    [4, 2.5, 1],
    [-4, -2.5, -1],
    [0, -2.5, 1],
    [4, -2.5, -1],
  ];

  function getLabel(x, y) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      let newLabel = sign * labelScale(dist({ x, y }, { x: cx, y: cy }));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }

  let radius = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = randUniform(-radius, radius);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getLabel(x + noiseX, y + noiseY);
    points.push({ x, y, label });
  }
  return points;
}

export function classifySpiralData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = (i / n) * 5;
      let t = ((1.75 * i) / n) * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({ x, y, label });
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;

  function getCircleLabel(p: Point, center: Point) {
    return dist(p, center) < radius * 0.5 ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel(
      { x: x + noiseX, y: y + noiseY },
      { x: 0, y: 0 }
    );
    points.push({ x, y, label });
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel(
      { x: x + noiseX, y: y + noiseY },
      { x: 0, y: 0 }
    );
    points.push({ x, y, label });
  }
  return points;
}

export function classifyXORData(
  numSamples: number,
  noise: number
): Example2D[] {
  function getXORLabel(p: Point) {
    return p.x * p.y >= 0 ? 1 : -1;
  }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let padding = 0.3;
    x += x > 0 ? padding : -padding; // Padding.
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-5, 5) * noise;
    let noiseY = randUniform(-5, 5) * noise;
    let label = getXORLabel({ x: x + noiseX, y: y + noiseY });
    points.push({ x, y, label });
  }
  return points;
}

/**
 * Moons dataset - two interleaving half circles.
 * Based on scikit-learn's make_moons.
 */
export function classifyMoonsData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples / 2; i++) {
    let selection = randUniform(0, Math.PI);
    let x = 2.5 * (Math.cos(selection) - 0.5) + randUniform(-1, 1) * noise;
    let y = 2.5 * (Math.sin(selection) - 0.25) + randUniform(-1, 1) * noise;
    let label = 1;
    points.push({ x, y, label });
  }
  for (let i = 0; i < numSamples / 2; i++) {
    let selection = randUniform(0, Math.PI);
    let x = 2.5 * (-Math.cos(selection) + 0.5) + randUniform(-1, 1) * noise;
    let y = 2.5 * (-Math.sin(selection) + 0.25) + randUniform(-1, 1) * noise;
    let label = -1;
    points.push({ x, y, label });
  }
  return points;
}

/**
 * Polar heart function for heart-shaped dataset.
 * From https://pavpanchekha.com/blog/heart-polar-coordinates.html
 */
function polarHeart(t: number): number {
  t += Math.PI / 2;
  let r =
    (Math.sin(t) * Math.sqrt(Math.abs(Math.cos(t)))) / (Math.sin(t) + 7 / 5) -
    2 * Math.sin(t) +
    2;
  return r;
}

/**
 * Heart-shaped dataset.
 */
export function classifyHeartData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let step = (2 * Math.PI) / (numSamples / 2);
  for (let i = 0; i < numSamples / 2; i++) {
    let t = i * step;
    let r = polarHeart(t);
    let x = 2.25 * r * Math.sin(t) + randUniform(-1, 1) * noise;
    let y = 3.75 + 2.25 * r * Math.cos(t) + randUniform(-1, 1) * noise;
    let label = -1;
    points.push({ x, y, label });
    x = 2.25 * (r - 0.5) * Math.sin(t) + randUniform(-1, 1) * noise;
    y = 3.5 + 2.25 * (r - 0.6) * Math.cos(t) + randUniform(-1, 1) * noise;
    label = 1;
    points.push({ x, y, label });
  }
  return points;
}

/**
 * Polar snowflake function.
 */
function polarSnowflake(t: number): number {
  return 3.8 + 1.8 * Math.cos(6 * (t + Math.PI / 6));
}

/**
 * Snowflake-shaped dataset.
 */
export function classifySnowflakeData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let step = (2 * Math.PI) / (numSamples / 2);
  for (let i = 0; i < numSamples / 2; i++) {
    let t = i * step;
    let r = polarSnowflake(t);
    let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
    let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
    let label = -1;
    points.push({ x, y, label });
    x = (r - 1.2) * Math.sin(t) + randUniform(-1, 1) * noise;
    y = (r - 1.2) * Math.cos(t) + randUniform(-1, 1) * noise;
    label = 1;
    points.push({ x, y, label });
  }
  return points;
}

/**
 * Polar infinity function (lemniscate).
 */
function polarInfinity(t: number): number {
  let c = Math.cos(2 * (t + Math.PI / 2));
  if (c > 0) {
    return Math.sqrt(25 * c);
  } else {
    return null;
  }
}

/**
 * Infinity (lemniscate) shaped dataset.
 */
export function classifyInfinityData(
  numSamples: number,
  noise: number
): Example2D[] {
  let points: Example2D[] = [];
  let step = (2 * Math.PI) / numSamples;
  // This is a hack since we can't take the square root of a negative number.
  // See if statement in the polarInfinity function.
  numSamples *= 2;
  for (let i = 0; i < numSamples / 2; i++) {
    let t = i * step;
    let r = polarInfinity(t);
    if (r !== null) {
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = 1.2 * r * Math.cos(t) + randUniform(-1, 1) * noise;
      let label = i < numSamples / 4 ? 1 : -1;
      points.push({ x, y, label });
      x = (r - 1) * Math.sin(t) + randUniform(-1, 1) * noise;
      y = 1.2 * (r - 1) * Math.cos(t) + randUniform(-1, 1) * noise;
      label = i < numSamples / 4 ? -1 : 1;
      points.push({ x, y, label });
    }
  }
  return points;
}

// ============ Multi-dimensional Dataset Generators ============

/**
 * Generate N-dimensional Gaussian classification data.
 */
export function classifyNDGaussian(
  numSamples: number,
  numFeatures: number,
  noise: number
): ExampleND[] {
  var points: ExampleND[] = [];
  var variance = 0.5 + noise * 3.5;

  function genGauss(center: number[], label: number) {
    var i, j, features;
    for (i = 0; i < numSamples / 2; i++) {
      features = [];
      for (j = 0; j < numFeatures; j++) {
        features.push(normalRandom(center[j], variance));
      }
      points.push({ features: features, label: label });
    }
  }

  // Create two clusters at opposite corners
  var center1: number[] = [];
  var center2: number[] = [];
  var j;
  for (j = 0; j < numFeatures; j++) {
    center1.push(2);
    center2.push(-2);
  }

  genGauss(center1, 1);
  genGauss(center2, -1);
  return points;
}

/**
 * Generate N-dimensional XOR-like classification data.
 */
export function classifyNDXOR(
  numSamples: number,
  numFeatures: number,
  noise: number
): ExampleND[] {
  var points: ExampleND[] = [];
  var i, j, features, product, val, padding;

  for (i = 0; i < numSamples; i++) {
    features = [];
    product = 1;

    for (j = 0; j < numFeatures; j++) {
      val = randUniform(-5, 5);
      padding = 0.3;
      val += val > 0 ? padding : -padding;
      val += randUniform(-5, 5) * noise;
      features.push(val);
      product *= val > 0 ? 1 : -1;
    }

    points.push({ features: features, label: product });
  }
  return points;
}

/**
 * Generate N-dimensional regression data (hyperplane).
 */
export function regressNDPlane(
  numSamples: number,
  numFeatures: number,
  noise: number
): ExampleND[] {
  var points: ExampleND[] = [];
  var radius = 6;
  var i, j, features, sum, val, label;

  for (i = 0; i < numSamples; i++) {
    features = [];
    sum = 0;

    for (j = 0; j < numFeatures; j++) {
      val = randUniform(-radius, radius);
      features.push(val);
      sum += val + randUniform(-radius, radius) * noise;
    }

    // Normalize label to [-1, 1]
    label = Math.max(-1, Math.min(1, sum / (numFeatures * radius)));
    points.push({ features: features, label: label });
  }
  return points;
}

// ============ Utility Functions ============

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean?: number, variance?: number): number {
  if (mean === undefined) mean = 0;
  if (variance === undefined) variance = 1;

  let v1: number, v2: number, s: number, result: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  result = Math.sqrt((-2 * Math.log(s)) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the euclidean distance between two points in space. */
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
