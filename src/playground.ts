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

import * as nn from "./nn";
import { HeatMap, reduceMatrix } from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem,
} from "./state";
import {
  Example2D,
  ExampleND,
  shuffle,
  parseCSV,
  normalizeData,
  normalizeLabelsForClassification,
  normalizeLabelsForRegression,
  DatasetInfo,
  convertToND,
  convertTo2D,
} from "./dataset";
import { AppendingLineChart } from "./linechart";
import * as d3 from "d3";

var mainWidth;

// More scrolling
d3.select(".more button").on("click", function () {
  var position = 800;
  d3.transition().duration(1000).tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function () {
    var i = d3.interpolateNumber(
      window.pageYOffset || document.documentElement.scrollTop,
      offset
    );
    return function (t) {
      scrollTo(0, i(t));
    };
  };
}

var RECT_SIZE = 30;
var BIAS_SIZE = 5;
var NUM_SAMPLES_CLASSIFY = 500;
var NUM_SAMPLES_REGRESS = 1200;
var DENSITY = 100;

enum HoverType {
  BIAS,
  WEIGHT,
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

var INPUTS: { [name: string]: InputFeature } = {
  x: {
    f: function (x, y) {
      return x;
    },
    label: "X_1",
  },
  y: {
    f: function (x, y) {
      return y;
    },
    label: "X_2",
  },
  xSquared: {
    f: function (x, y) {
      return x * x;
    },
    label: "X_1^2",
  },
  ySquared: {
    f: function (x, y) {
      return y * y;
    },
    label: "X_2^2",
  },
  xTimesY: {
    f: function (x, y) {
      return x * y;
    },
    label: "X_1X_2",
  },
  sinX: {
    f: function (x, y) {
      return Math.sin(x);
    },
    label: "sin(X_1)",
  },
  sinY: {
    f: function (x, y) {
      return Math.sin(y);
    },
    label: "sin(X_2)",
  },
};

var HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    var self = this;
    d3.timer(function () {
      if (localTimerIndex < self.timerIndex) {
        return true; // Done.
      }
      oneStep();
      return false; // Not done.
    }, 0);
  }
}

var state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(function (prop) {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

var boundary: { [id: string]: number[][] } = {};
var selectedNodeId: string = null;
// Plot the heatmap.
var xDomain: [number, number] = [-6, 6];
var heatMap = new HeatMap(
  300,
  DENSITY,
  xDomain,
  xDomain,
  d3.select("#heatmap"),
  { showAxes: true }
);
var linkWidthScale = d3.scale
  .linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
var colorScale = d3.scale
  .linear<string, number>()
  .domain([-1, 0, 1])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);

var iter = 0;
var trainData: Example2D[] = [];
var testData: Example2D[] = [];
var trainDataND: ExampleND[] = [];
var testDataND: ExampleND[] = [];
var network: nn.Node[][] = null;
var lossTrain = 0;
var lossTest = 0;
var player = new Player();
var lineChart = new AppendingLineChart(d3.select("#linechart"), [
  "#777",
  "black",
]);

// Multi-dimensional dataset state
var isMultiDim = false;
var customDatasetInfo: DatasetInfo = null;
var numInputFeatures = 2;
var activeFeatures: boolean[] = [];
var neuronActivations: { [nodeId: string]: number } = {};
var isMultiClass = false;
var numClasses = 2;

/**
 * Sets multi-dimensional mode on or off and updates UI accordingly.
 */
function setMultiDimMode(enabled: boolean, info?: DatasetInfo) {
  var i, colormapParent;

  isMultiDim = enabled;
  customDatasetInfo = info || null;

  d3.select("body").classed("multi-dim-mode", enabled);

  if (enabled && info) {
    numInputFeatures = info.numFeatures;

    activeFeatures = [];
    for (i = 0; i < info.numFeatures; i++) {
      activeFeatures.push(true);
    }

    d3.select("#heatmap").style("visibility", "hidden");
    d3.select(".callout.thumbnail").style("display", "none");
    d3.select(".callout.weights").style("display", "none");
    d3.select("#colormap").style("visibility", "hidden");
    d3.select(".ui-showTestData").style("display", "none");
    d3.select(".ui-discretize").style("display", "none");

    // Hide the color scale label text
    colormapParent = d3.select("#colormap").node().parentNode;
    d3.select(colormapParent).select(".label").style("display", "none");

    showMultiDimInfo(info);

    d3.select(".column.features h4").html(
      "Features <span class='feature-count'>(" +
        info.numFeatures +
        " inputs)</span>"
    );

    d3.select(".column.features p")
      .text("Click on a feature to toggle it on/off.")
      .style("display", null);

    d3.select(".column.features").classed("many-inputs", info.numFeatures > 8);
  } else {
    numInputFeatures = 2;
    activeFeatures = [];
    isMultiClass = false;
    numClasses = 2;
    d3.select("#heatmap").style("visibility", "visible");
    d3.select("#colormap").style("visibility", "visible");
    d3.select(".ui-showTestData").style("display", null);
    d3.select(".ui-discretize").style("display", null);
    d3.select("#multi-dim-info").remove();

    // Show the color scale label text
    colormapParent = d3.select("#colormap").node().parentNode;
    d3.select(colormapParent).select(".label").style("display", null);

    d3.select(".column.features p")
      .text("Which properties do you want to feed in?")
      .style("display", null);
    d3.select(".column.features h4").text("Features");
    d3.select(".column.features").classed("many-inputs", false);
  }
}

/**
 * Shows information panel for multi-dimensional datasets.
 */
function showMultiDimInfo(info: DatasetInfo) {
  var infoDiv, typeText, encodingDiv, encodingList;
  var i, col, mapping, mappingStr, keys, entries;

  d3.select("#multi-dim-info").remove();

  // Insert after metrics but before heatmap
  infoDiv = d3
    .select(".column.output")
    .insert("div", "#heatmap")
    .attr("id", "multi-dim-info");

  infoDiv
    .append("div")
    .style("font-weight", "bold")
    .text("Custom Dataset Loaded");

  infoDiv
    .append("div")
    .html("<strong>Total Features:</strong> " + info.numFeatures);

  infoDiv
    .append("div")
    .attr("id", "active-features-count")
    .html("<strong>Active Features:</strong> " + info.numFeatures);

  typeText = info.isClassification
    ? info.numClasses > 2
      ? "Multi-class (" + info.numClasses + " classes)"
      : "Binary Classification"
    : "Regression";
  infoDiv.append("div").html("<strong>Type:</strong> " + typeText);

  if (info.isClassification && info.numClasses > 2 && info.classLabels) {
    infoDiv
      .append("div")
      .style("margin-top", "5px")
      .html("<strong>Classes:</strong> " + info.classLabels.join(", "));
  }

  if (info.isClassification) {
    infoDiv
      .append("div")
      .attr("id", "accuracy-display")
      .style({
        "margin-top": "8px",
        padding: "5px",
        background: "#e8f4f8",
        "border-radius": "3px",
      })
      .html(
        "<strong>Train Accuracy:</strong> <span id='train-acc'>--</span> | <strong>Test Accuracy:</strong> <span id='test-acc'>--</span>"
      );
  }

  if (info.featureNames.length > 0 && info.featureNames.length <= 10) {
    infoDiv
      .append("div")
      .style("margin-top", "8px")
      .attr("id", "feature-list")
      .html(
        "<strong>Columns:</strong><br/>" +
          info.featureNames
            .map(function (n, i) {
              var encoded =
                info.encodedColumns && info.encodedColumns.indexOf(n) !== -1;
              var suffix = encoded
                ? " <em style='color:#666;'>(encoded)</em>"
                : "";
              return (
                "<span style='color:#183D4E;'>" +
                (i + 1) +
                ".</span> " +
                n +
                suffix
              );
            })
            .join("<br/>")
      );
  } else if (info.featureNames.length > 10) {
    infoDiv
      .append("div")
      .style("margin-top", "8px")
      .html(
        "<strong>Columns:</strong> " +
          info.featureNames.slice(0, 5).join(", ") +
          " ... <em>(+" +
          (info.featureNames.length - 5) +
          " more)</em>"
      );
  }

  // Show label encoding info if any columns were encoded
  if (info.encodedColumns && info.encodedColumns.length > 0) {
    encodingDiv = infoDiv.append("div").style({
      "margin-top": "10px",
      padding: "8px",
      background: "#fff8e1",
      "border-radius": "4px",
      "font-size": "11px",
    });

    encodingDiv
      .append("div")
      .style({ "font-weight": "bold", "margin-bottom": "5px" })
      .text("Label Encoding Applied");

    encodingList = encodingDiv.append("div");

    for (i = 0; i < info.encodedColumns.length; i++) {
      col = info.encodedColumns[i];
      mapping = info.labelEncodings[col];
      keys = Object.keys(mapping);

      // Build mapping string
      entries = [];
      for (var k = 0; k < keys.length; k++) {
        entries.push([keys[k], mapping[keys[k]]]);
      }

      mappingStr = entries
        .map(function (entry) {
          return '"' + entry[0] + '"→' + entry[1];
        })
        .join(", ");

      // Truncate if too long
      if (mappingStr.length > 50) {
        mappingStr =
          entries
            .slice(0, 3)
            .map(function (entry) {
              return '"' + entry[0] + '"→' + entry[1];
            })
            .join(", ") +
          " ... (+" +
          (entries.length - 3) +
          " more)";
      }

      encodingList
        .append("div")
        .style("margin", "2px 0")
        .html("<strong>" + col + ":</strong> " + mappingStr);
    }
  }
}

/**
 * Updates the display of active feature count.
 */
function updateActiveFeatureCount() {
  var activeCount, i;

  if (!isMultiDim || !customDatasetInfo) return;

  activeCount = 0;
  for (i = 0; i < activeFeatures.length; i++) {
    if (activeFeatures[i]) activeCount++;
  }

  d3.select("#active-features-count").html(
    "<strong>Active Features:</strong> " +
      activeCount +
      " / " +
      customDatasetInfo.numFeatures
  );
}

/**
 * Computes average activations for all neurons using a sample of training data.
 */
function computeNeuronActivations() {
  var dataToUse: ExampleND[] | Example2D[];
  var isND = isMultiDim;
  var sampleSize, step, activationSums, activationCounts;
  var inputSums, inputCounts, activeIndices;
  var i, j, input, point, val, featureIdx;

  neuronActivations = {};

  if (isND) {
    dataToUse = trainDataND;
    if (dataToUse.length === 0) return;
  } else {
    dataToUse = trainData;
    if (dataToUse.length === 0) return;
  }

  sampleSize = Math.min(dataToUse.length, 100);
  step = Math.max(1, Math.floor(dataToUse.length / sampleSize));

  activationSums = {};
  activationCounts = {};

  nn.forEachNode(network, true, function (node) {
    activationSums[node.id] = 0;
    activationCounts[node.id] = 0;
  });

  inputSums = [];
  inputCounts = 0;
  if (isND) {
    activeIndices = getActiveFeatureIndices();
    for (i = 0; i < activeIndices.length; i++) {
      inputSums.push(0);
    }
  }

  activeIndices = isND ? getActiveFeatureIndices() : null;

  for (i = 0; i < dataToUse.length; i += step) {
    if (isND) {
      point = dataToUse[i] as ExampleND;
      input = [];
      for (j = 0; j < activeIndices.length; j++) {
        val = point.features[activeIndices[j]];
        input.push(val);
        inputSums[j] += val;
      }
      inputCounts++;
    } else {
      point = dataToUse[i] as Example2D;
      input = constructInput(point.x, point.y);
    }

    if (isMultiClass) {
      nn.forwardPropMultiClass(network, input);
    } else {
      nn.forwardProp(network, input);
    }

    nn.forEachNode(network, true, function (node) {
      activationSums[node.id] += node.output;
      activationCounts[node.id]++;
    });
  }

  nn.forEachNode(network, true, function (node) {
    if (activationCounts[node.id] > 0) {
      neuronActivations[node.id] =
        activationSums[node.id] / activationCounts[node.id];
    } else {
      neuronActivations[node.id] = 0;
    }
  });

  if (isND && inputCounts > 0) {
    activeIndices = getActiveFeatureIndices();
    for (j = 0; j < activeIndices.length; j++) {
      featureIdx = activeIndices[j];
      neuronActivations["input_" + featureIdx] = inputSums[j] / inputCounts;
    }
  }
}

/**
 * Updates the visual colors of neurons based on their activations.
 */
function updateNeuronColors() {
  var i, nodeId, isActive, activation, color, canvasDiv, canvas, ctx;
  var layerIdx, currentLayer, node;

  if (isMultiDim && customDatasetInfo) {
    for (i = 0; i < customDatasetInfo.numFeatures; i++) {
      nodeId = "input_" + i;
      isActive = activeFeatures[i];

      if (isActive && neuronActivations[nodeId] !== undefined) {
        activation = neuronActivations[nodeId];
        color = colorScale(activation).toString();

        d3.select("#node" + nodeId + " rect").style("fill", color);

        canvasDiv = d3.select("#canvas-" + nodeId);
        canvas = canvasDiv.select("canvas");
        if (canvas.node()) {
          ctx = (canvas.node() as HTMLCanvasElement).getContext("2d");
          ctx.fillStyle = color;
          ctx.fillRect(0, 0, 10, 10);
        }
      }
    }
  }

  for (layerIdx = 1; layerIdx < network.length; layerIdx++) {
    currentLayer = network[layerIdx];
    for (i = 0; i < currentLayer.length; i++) {
      node = currentLayer[i];
      activation = neuronActivations[node.id] || 0;
      color = colorScale(activation).toString();

      d3.select("#node" + node.id + " rect").style("fill", color);

      canvasDiv = d3.select("#canvas-" + node.id);
      canvas = canvasDiv.select("canvas");
      if (canvas.node()) {
        ctx = (canvas.node() as HTMLCanvasElement).getContext("2d");
        ctx.fillStyle = color;
        ctx.fillRect(0, 0, 10, 10);
      }
    }
  }
}

function makeGUI() {
  var showTestData, discretize, percTrain, noise, batchSize;
  var activationDropdown, learningRate, regularDropdown, regularRate, problem;
  var currentMax, x, xAxis;
  var dataThumbnails, datasetKey, regDataThumbnails, regDatasetKey;

  d3.select("#reset-button").on("click", function () {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(function (isPlaying) {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", function () {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", function () {
    if (!isMultiDim) {
      generateData();
      parametersChanged = true;
    }
  });

  d3.select("#upload-button").on("click", function () {
    (document.getElementById("file-upload") as HTMLInputElement).click();
  });

  d3.select("#file-upload").on("change", function () {
    var input = this as HTMLInputElement;
    var file, reader;

    if (input.files && input.files[0]) {
      file = input.files[0];
      reader = new FileReader();

      reader.onload = function (e) {
        var csvText, result, normalizedData, splitIndex, classInfo, encodedInfo;

        try {
          csvText = e.target.result as string;
          result = parseCSV(csvText);

          normalizedData = normalizeData(result.data);

          if (result.info.isClassification) {
            normalizedData = normalizeLabelsForClassification(
              normalizedData,
              result.info
            );
            state.problem = Problem.CLASSIFICATION;

            isMultiClass = result.info.numClasses > 2;
            numClasses = result.info.numClasses;
          } else {
            normalizedData = normalizeLabelsForRegression(normalizedData);
            state.problem = Problem.REGRESSION;
            isMultiClass = false;
            numClasses = 1;
          }

          d3.select("#problem").property(
            "value",
            getKeyFromValue(problems, state.problem)
          );

          shuffle(normalizedData);
          splitIndex = Math.floor(
            (normalizedData.length * state.percTrainData) / 100
          );
          trainDataND = normalizedData.slice(0, splitIndex);
          testDataND = normalizedData.slice(splitIndex);

          setMultiDimMode(true, result.info);

          classInfo = result.info.isClassification
            ? isMultiClass
              ? ", " + numClasses + " classes"
              : ", binary"
            : "";
          encodedInfo =
            result.info.encodedColumns && result.info.encodedColumns.length > 0
              ? ", " + result.info.encodedColumns.length + " encoded"
              : "";
          d3.select("#upload-status").text(
            "Loaded: " +
              file.name +
              " (" +
              normalizedData.length +
              " samples, " +
              result.info.numFeatures +
              " features" +
              classInfo +
              encodedInfo +
              ")"
          );

          d3.selectAll(".data-thumbnail").classed("selected", false);

          parametersChanged = true;
          reset();
        } catch (err) {
          d3.select("#upload-status")
            .style("color", "red")
            .text("Error: " + err.message);
        }
      };

      reader.readAsText(file);
    }
  });

  dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function () {
    var newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset && !isMultiDim) {
      return; // No-op.
    }
    state.dataset = newDataset;
    dataThumbnails.classed("selected", false);
    d3.selectAll("canvas[data-regDataset]").classed("selected", false);
    d3.select(this).classed("selected", true);

    setMultiDimMode(false);
    d3.select("#upload-status").text("");

    generateData();
    parametersChanged = true;
    reset();
  });

  datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select("canvas[data-dataset=" + datasetKey + "]").classed(
    "selected",
    true
  );

  regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function () {
    var newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset && !isMultiDim) {
      return; // No-op.
    }
    state.regDataset = newDataset;
    regDataThumbnails.classed("selected", false);
    d3.selectAll("canvas[data-dataset]").classed("selected", false);
    d3.select(this).classed("selected", true);

    setMultiDimMode(false);
    d3.select("#upload-status").text("");

    generateData();
    parametersChanged = true;
    reset();
  });

  regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select("canvas[data-regDataset=" + regDatasetKey + "]").classed(
    "selected",
    true
  );

  d3.select("#add-layers").on("click", function () {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", function () {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  showTestData = d3.select("#show-test-data").on("change", function () {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    if (!isMultiDim) {
      heatMap.updateTestPoints(state.showTestData ? testData : []);
    }
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  discretize = d3.select("#discretize").on("change", function () {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checkbox according to the current state.
  discretize.property("checked", state.discretize);

  percTrain = d3.select("#percTrainData").on("input", function () {
    var allData, splitIndex;
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    if (!isMultiDim) {
      generateData();
    } else {
      allData = trainDataND.concat(testDataND);
      shuffle(allData);
      splitIndex = Math.floor((allData.length * state.percTrainData) / 100);
      trainDataND = allData.slice(0, splitIndex);
      testDataND = allData.slice(splitIndex);
    }
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  noise = d3.select("#noise").on("input", function () {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    if (!isMultiDim) {
      generateData();
    }
    parametersChanged = true;
    reset();
  });
  currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  batchSize = d3.select("#batchSize").on("input", function () {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  activationDropdown = d3.select("#activations").on("change", function () {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
  });
  activationDropdown.property(
    "value",
    getKeyFromValue(activations, state.activation)
  );

  learningRate = d3.select("#learningRate").on("change", function () {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  regularDropdown = d3.select("#regularizations").on("change", function () {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property(
    "value",
    getKeyFromValue(regularizations, state.regularization)
  );

  regularRate = d3.select("#regularRate").on("change", function () {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  problem = d3.select("#problem").on("change", function () {
    state.problem = problems[this.value];
    if (!isMultiDim) {
      generateData();
      drawDatasetThumbnails();
    }
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Add scale to the gradient color map.
  x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  xAxis = d3.svg
    .axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core")
    .append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.
  window.addEventListener("resize", function () {
    var newWidth = document
      .querySelector("#main-part")
      .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, function (node) {
    d3.select("rect#bias-" + node.id).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  var layerIdx, currentLayer, i, node, j, link;

  for (layerIdx = 1; layerIdx < network.length; layerIdx++) {
    currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (i = 0; i < currentLayer.length; i++) {
      node = currentLayer[i];
      for (j = 0; j < node.inputLinks.length; j++) {
        link = node.inputLinks[j];
        container
          .select("#link" + link.source.id + "-" + link.dest.id)
          .style({
            "stroke-dashoffset": -iter / 3,
            "stroke-width": linkWidthScale(Math.abs(link.weight)),
            stroke: colorScale(link.weight),
          })
          .datum(link);
      }
    }
  }
}

function drawNode(
  cx: number,
  cy: number,
  nodeId: string,
  isInput: boolean,
  container,
  node?: nn.Node
) {
  var x = cx - RECT_SIZE / 2;
  var y = cy - RECT_SIZE / 2;
  var nodeGroup, activeOrNotClass, idx, label, fullLabel, isMultiDimLabel;
  var text, myRe, myArray, lastIndex, prefix, sep, suffix;
  var div, shouldCreateHeatmap, nodeHeatMap, canvas, ctx, isActive;

  nodeGroup = container.append("g").attr({
    class: "node",
    id: "node" + nodeId,
    transform: "translate(" + x + "," + y + ")",
  });

  // Draw the main rectangle.
  nodeGroup
    .append("rect")
    .attr({ x: 0, y: 0, width: RECT_SIZE, height: RECT_SIZE });

  if (isMultiDim && activeFeatures.length > 0) {
    idx = parseInt(nodeId.replace("input_", ""));
    if (!isNaN(idx) && idx >= 0 && idx < activeFeatures.length) {
      activeOrNotClass = activeFeatures[idx] ? "active" : "inactive";
    } else {
      activeOrNotClass = "active";
    }
  } else if (!isMultiDim) {
    activeOrNotClass =
      nodeId in INPUTS ? (state[nodeId] ? "active" : "inactive") : "active";
  } else {
    activeOrNotClass = "active";
  }

  if (isInput) {
    isMultiDimLabel = false;

    if (isMultiDim && customDatasetInfo) {
      idx = parseInt(nodeId.replace("input_", ""));
      fullLabel = customDatasetInfo.featureNames[idx] || "X" + (idx + 1);
      label = fullLabel;
      isMultiDimLabel = true;
      if (label.length > 12) {
        label = label.substring(0, 10) + "..";
      }
    } else if (INPUTS[nodeId]) {
      label = INPUTS[nodeId].label != null ? INPUTS[nodeId].label : nodeId;
      fullLabel = label;
    } else {
      label = nodeId;
      fullLabel = label;
    }

    // Draw the input label.
    text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2,
      "text-anchor": "end",
    });

    if (fullLabel !== label) {
      text.append("title").text(fullLabel);
    }

    if (isMultiDimLabel) {
      text.style({
        "font-size": "11px",
        "font-weight": "400",
        fill: "#333",
      });
      text.append("tspan").text(label);
    } else if (/[_^]/.test(label)) {
      myRe = /(.*?)([_^])(.)/g;
      lastIndex = 0;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        prefix = myArray[1];
        sep = myArray[2];
        suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text
          .append("tspan")
          .attr("baseline-shift", sep === "_" ? "sub" : "super")
          .style("font-size", "9px")
          .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }

  if (!isInput) {
    // Draw the node's bias.
    nodeGroup
      .append("rect")
      .attr({
        id: "bias-" + nodeId,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      })
      .on("mouseenter", function () {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      })
      .on("mouseleave", function () {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  div = d3
    .select("#network")
    .insert("div", ":first-child")
    .attr({
      id: "canvas-" + nodeId,
      class: "canvas",
    })
    .style({
      position: "absolute",
      left: x + 3 + "px",
      top: y + 3 + "px",
    });

  if (!isMultiDim) {
    div
      .on("mouseenter", function () {
        selectedNodeId = nodeId;
        div.classed("hovered", true);
        nodeGroup.classed("hovered", true);
        updateDecisionBoundary(network, false);
        heatMap.updateBackground(boundary[nodeId], state.discretize);
      })
      .on("mouseleave", function () {
        selectedNodeId = null;
        div.classed("hovered", false);
        nodeGroup.classed("hovered", false);
        updateDecisionBoundary(network, false);
        heatMap.updateBackground(
          boundary[nn.getOutputNode(network).id],
          state.discretize
        );
      });
  }

  if (isInput && !isMultiDim && nodeId in INPUTS) {
    div.on("click", function () {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }

  if (isInput) {
    div.classed(activeOrNotClass, true);
  }

  if (!isMultiDim) {
    shouldCreateHeatmap = !isInput || nodeId in INPUTS;
    if (shouldCreateHeatmap) {
      nodeHeatMap = new HeatMap(
        RECT_SIZE,
        DENSITY / 10,
        xDomain,
        xDomain,
        div,
        { noSvg: true }
      );
      div.datum({ heatmap: nodeHeatMap, id: nodeId });
    }
  } else {
    canvas = div
      .append("canvas")
      .attr("width", 10)
      .attr("height", 10)
      .style("width", RECT_SIZE + "px")
      .style("height", RECT_SIZE + "px");

    ctx = (canvas.node() as HTMLCanvasElement).getContext("2d");

    if (isInput) {
      isActive =
        activeFeatures.length > 0 &&
        activeFeatures[parseInt(nodeId.replace("input_", ""))] === true;
      ctx.fillStyle = isActive ? "#e8f4f8" : "#f5f5f5";
    } else {
      ctx.fillStyle = "#e8eaeb";
    }
    ctx.fillRect(0, 0, 10, 10);
  }
}

/** Draw the neural network. */
function drawNetwork(network: nn.Node[][]): void {
  var svg = d3.select("#svg");
  var padding, co, cf, width, node2coord, container;
  var numLayers, featureWidth, layerScale, nodeIndexScale;
  var calloutThumb, calloutWeights, idWithCallout, targetIdWithCallout;
  var cx, inputLayer, maxY, totalFeatures, i, nodeId, cy, nodeIds;
  var layerIdx,
    numNodes,
    node,
    j,
    link,
    path,
    prevLayer,
    lastNodePrevLayer,
    midPoint;
  var outputLayer, classLabel;

  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  padding = 3;
  co = d3.select(".column.output").node() as HTMLDivElement;
  cf = d3.select(".column.features").node() as HTMLDivElement;
  width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  node2coord = {};
  container = svg
    .append("g")
    .classed("core", true)
    .attr("transform", "translate(" + padding + "," + padding + ")");

  // Draw the network layer by layer.
  numLayers = network.length;
  featureWidth = 118;
  layerScale = d3.scale
    .ordinal<number, number>()
    .domain(d3.range(1, numLayers - 1))
    .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  nodeIndexScale = function (nodeIndex: number) {
    return nodeIndex * (RECT_SIZE + 25);
  };

  calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  calloutWeights = d3.select(".callout.weights").style("display", "none");
  idWithCallout = null;
  targetIdWithCallout = null;

  // Draw the input layer separately.
  cx = RECT_SIZE / 2 + 50;
  inputLayer = network[0];
  maxY = nodeIndexScale(inputLayer.length);

  if (isMultiDim && customDatasetInfo) {
    totalFeatures = customDatasetInfo.numFeatures;
    maxY = nodeIndexScale(totalFeatures);

    for (i = 0; i < totalFeatures; i++) {
      nodeId = "input_" + i;
      cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[nodeId] = { cx: cx, cy: cy };
      drawNode(cx, cy, nodeId, true, container);

      addFeatureClickHandler(i, nodeId);
    }
  } else if (!isMultiDim) {
    nodeIds = Object.keys(INPUTS);
    maxY = nodeIndexScale(nodeIds.length);

    for (i = 0; i < nodeIds.length; i++) {
      nodeId = nodeIds[i];
      cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[nodeId] = { cx: cx, cy: cy };
      drawNode(cx, cy, nodeId, true, container);
    }
  }

  // Draw the intermediate layers.
  for (layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    numNodes = network[layerIdx].length;
    cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);

    for (i = 0; i < numNodes; i++) {
      node = network[layerIdx][i];
      cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = { cx: cx, cy: cy };
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      if (!isMultiDim) {
        numNodes = network[layerIdx].length;
        var nextNumNodes = network[layerIdx + 1].length;
        if (
          idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes
        ) {
          calloutThumb.style({
            display: null,
            top: 20 + 3 + cy + "px",
            left: cx + "px",
          });
          idWithCallout = node.id;
        }
      }

      // Draw links.
      for (j = 0; j < node.inputLinks.length; j++) {
        link = node.inputLinks[j];
        if (!(link.source.id in node2coord)) {
          continue;
        }
        path = drawLink(
          link,
          node2coord,
          network,
          container,
          j === 0,
          j,
          node.inputLinks.length
        ).node() as any;

        // Show callout to weights.
        if (!isMultiDim) {
          prevLayer = network[layerIdx - 1];
          lastNodePrevLayer = prevLayer[prevLayer.length - 1];
          if (
            targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes
          ) {
            midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
            calloutWeights.style({
              display: null,
              top: midPoint.y + 5 + "px",
              left: midPoint.x + 3 + "px",
            });
            targetIdWithCallout = link.dest.id;
          }
        }
      }
    }
  }

  // Draw the output node(s) separately.
  cx = width + RECT_SIZE / 2;
  outputLayer = network[numLayers - 1];

  for (i = 0; i < outputLayer.length; i++) {
    node = outputLayer[i];
    cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[node.id] = { cx: cx, cy: cy };

    // Draw links.
    for (j = 0; j < node.inputLinks.length; j++) {
      link = node.inputLinks[j];
      if (!(link.source.id in node2coord)) {
        continue;
      }
      drawLink(
        link,
        node2coord,
        network,
        container,
        j === 0,
        j,
        node.inputLinks.length
      );
    }

    // Draw the output node only for multi-dimensional datasets
    if (isMultiDim) {
      drawNode(cx, cy, node.id, false, container, node);

      // For multi-class, add class label next to output node
      if (isMultiClass && customDatasetInfo && customDatasetInfo.classLabels) {
        classLabel = customDatasetInfo.classLabels[i];
        container
          .append("text")
          .attr({
            class: "output-label",
            x: cx + RECT_SIZE / 2 + 8,
            y: cy + 4,
            "text-anchor": "start",
          })
          .style({
            "font-size": "11px",
            fill: "#666",
          })
          .text("Class " + classLabel);
      }
    }
  }

  // Update maxY for output layer (only if multi-dim)
  if (isMultiDim) {
    maxY = Math.max(maxY, nodeIndexScale(outputLayer.length));
  }

  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  var height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  var node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

/**
 * Adds click handler for toggling features on/off.
 */
function addFeatureClickHandler(featureIdx: number, nodeId: string) {
  var canvasDiv, nodeGroup;

  if (!isMultiDim || activeFeatures.length === 0) return;

  var toggleFeature = function () {
    var activeCount = 0;
    var i;
    for (i = 0; i < activeFeatures.length; i++) {
      if (activeFeatures[i]) activeCount++;
    }

    // Don't allow disabling the last active feature
    if (activeFeatures[featureIdx] && activeCount <= 1) {
      return;
    }

    activeFeatures[featureIdx] = !activeFeatures[featureIdx];
    parametersChanged = true;
    reset();
  };

  canvasDiv = document.getElementById("canvas-" + nodeId);
  if (canvasDiv) {
    canvasDiv.style.cursor = "pointer";
    canvasDiv.onclick = function (e) {
      e.preventDefault();
      e.stopPropagation();
      toggleFeature();
    };
  }

  nodeGroup = d3.select("#node" + nodeId);
  if (nodeGroup && nodeGroup.node()) {
    (nodeGroup.node() as SVGElement).style.cursor = "pointer";
    nodeGroup.on("click", function () {
      toggleFeature();
    });
  }
}

function addPlusMinusControl(x: number, layerIdx: number) {
  var div, i, firstRow, suffix;

  div = d3
    .select("#network")
    .append("div")
    .classed("plus-minus-neurons", true)
    .style("left", x - 10 + "px");

  i = layerIdx - 1;
  firstRow = div.append("div").attr("class", "ui-numNodes" + layerIdx);

  firstRow
    .append("button")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", function () {
      var numNeurons = state.networkShape[i];
      if (numNeurons >= 8) return;
      state.networkShape[i]++;
      parametersChanged = true;
      reset();
    })
    .append("i")
    .attr("class", "material-icons")
    .text("add");

  firstRow
    .append("button")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", function () {
      var numNeurons = state.networkShape[i];
      if (numNeurons <= 1) return;
      state.networkShape[i]--;
      parametersChanged = true;
      reset();
    })
    .append("i")
    .attr("class", "material-icons")
    .text("remove");

  suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(state.networkShape[i] + " neuron" + suffix);
}

function updateHoverCard(
  type: HoverType,
  nodeOrLink?: nn.Node | nn.Link,
  coordinates?: [number, number]
) {
  var hovercard = d3.select("#hovercard");
  var value, name, input;

  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", function () {
    hovercard.select(".value").style("display", "none");
    input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function () {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", function () {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  value =
    type === HoverType.WEIGHT
      ? (nodeOrLink as nn.Link).weight
      : (nodeOrLink as nn.Node).bias;
  name = type === HoverType.WEIGHT ? "Weight" : "Bias";
  hovercard.style({
    left: coordinates[0] + 20 + "px",
    top: coordinates[1] + "px",
    display: "block",
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value").style("display", null).text(value.toPrecision(2));
  hovercard
    .select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
  input: nn.Link,
  node2coord: { [id: string]: { cx: number; cy: number } },
  network: nn.Node[][],
  container,
  isFirst: boolean,
  index: number,
  length: number
) {
  var line, source, dest, datum, diagonal;

  line = container.insert("path", ":first-child");
  source = node2coord[input.source.id];
  dest = node2coord[input.dest.id];
  datum = {
    source: { y: source.cx + RECT_SIZE / 2 + 2, x: source.cy },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12,
    },
  };
  diagonal = d3.svg.diagonal().projection(function (d) {
    return [d.y, d.x];
  });
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0),
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container
    .append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function () {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    })
    .on("mouseleave", function () {
      updateHoverCard(null);
    });
  return line;
}

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  var xScale, yScale, i, j, x, y, input, nodeId;

  if (isMultiDim) return;

  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, function (node) {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }

  xScale = d3.scale
    .linear()
    .domain([0, DENSITY - 1])
    .range(xDomain);
  yScale = d3.scale
    .linear()
    .domain([DENSITY - 1, 0])
    .range(xDomain);

  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, function (node) {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      x = xScale(i);
      y = yScale(j);
      input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, function (node) {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  var loss = 0;
  var i, dataPoint, input, output;

  for (i = 0; i < dataPoints.length; i++) {
    dataPoint = dataPoints[i];
    input = constructInput(dataPoint.x, dataPoint.y);
    output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function getLossND(network: nn.Node[][], dataPoints: ExampleND[]): number {
  var loss = 0;
  var i, dataPoint, output;

  for (i = 0; i < dataPoints.length; i++) {
    dataPoint = dataPoints[i];
    output = nn.forwardProp(network, dataPoint.features);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function getLossNDActive(
  network: nn.Node[][],
  dataPoints: ExampleND[],
  activeIndices: number[]
): number {
  var loss = 0;
  var i, j, dataPoint, activeInput, output;

  for (i = 0; i < dataPoints.length; i++) {
    dataPoint = dataPoints[i];
    activeInput = [];
    for (j = 0; j < activeIndices.length; j++) {
      activeInput.push(dataPoint.features[activeIndices[j]]);
    }
    output = nn.forwardProp(network, activeInput);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function getLossNDMultiClass(
  network: nn.Node[][],
  dataPoints: ExampleND[],
  activeIndices: number[]
): number {
  var loss = 0;
  var i, j, dataPoint, activeInput, outputs, targetClass;

  for (i = 0; i < dataPoints.length; i++) {
    dataPoint = dataPoints[i];
    activeInput = [];
    for (j = 0; j < activeIndices.length; j++) {
      activeInput.push(dataPoint.features[activeIndices[j]]);
    }
    outputs = nn.forwardPropMultiClass(network, activeInput);
    targetClass = Math.round(dataPoint.label);
    loss += nn.MultiClassErrors.crossEntropy(outputs, targetClass);
  }
  return loss / dataPoints.length;
}

function getAccuracyNDMultiClass(
  network: nn.Node[][],
  dataPoints: ExampleND[],
  activeIndices: number[]
): number {
  var correct = 0;
  var i,
    j,
    c,
    dataPoint,
    activeInput,
    outputs,
    predictedClass,
    maxProb,
    targetClass;

  for (i = 0; i < dataPoints.length; i++) {
    dataPoint = dataPoints[i];
    activeInput = [];
    for (j = 0; j < activeIndices.length; j++) {
      activeInput.push(dataPoint.features[activeIndices[j]]);
    }
    outputs = nn.forwardPropMultiClass(network, activeInput);

    predictedClass = 0;
    maxProb = outputs[0];
    for (c = 1; c < outputs.length; c++) {
      if (outputs[c] > maxProb) {
        maxProb = outputs[c];
        predictedClass = c;
      }
    }

    targetClass = Math.round(dataPoint.label);
    if (predictedClass === targetClass) {
      correct++;
    }
  }
  return correct / dataPoints.length;
}

function getAccuracyNDBinary(
  network: nn.Node[][],
  dataPoints: ExampleND[],
  activeIndices: number[]
): number {
  var correct = 0;
  var i, j, dataPoint, activeInput, output, predicted, target;

  for (i = 0; i < dataPoints.length; i++) {
    dataPoint = dataPoints[i];
    activeInput = [];
    for (j = 0; j < activeIndices.length; j++) {
      activeInput.push(dataPoint.features[activeIndices[j]]);
    }
    output = nn.forwardProp(network, activeInput);

    predicted = output >= 0 ? 1 : -1;
    target = dataPoint.label >= 0 ? 1 : -1;

    if (predicted === target) {
      correct++;
    }
  }
  return correct / dataPoints.length;
}

function updateUI(firstStep?: boolean) {
  var selectedId, activeIndices, trainAcc, testAcc;

  if (firstStep === undefined) firstStep = false;

  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);

  if (isMultiDim) {
    computeNeuronActivations();
    updateNeuronColors();
  }

  if (!isMultiDim) {
    // Get the decision boundary of the network.
    updateDecisionBoundary(network, firstStep);
    selectedId =
      selectedNodeId != null ? selectedNodeId : nn.getOutputNode(network).id;
    heatMap.updateBackground(boundary[selectedId], state.discretize);

    // Update all decision boundaries.
    d3.select("#network")
      .selectAll("div.canvas")
      .each(function (data: { heatmap: HeatMap; id: string }) {
        if (data && data.heatmap && data.id in boundary) {
          data.heatmap.updateBackground(
            reduceMatrix(boundary[data.id], 10),
            state.discretize
          );
        }
      });
  }

  function zeroPad(n: number): string {
    var pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);

  // Update accuracy for classification tasks
  if (isMultiDim && state.problem === Problem.CLASSIFICATION) {
    activeIndices = getActiveFeatureIndices();

    if (isMultiClass) {
      trainAcc = getAccuracyNDMultiClass(network, trainDataND, activeIndices);
      testAcc = getAccuracyNDMultiClass(network, testDataND, activeIndices);
    } else {
      trainAcc = getAccuracyNDBinary(network, trainDataND, activeIndices);
      testAcc = getAccuracyNDBinary(network, testDataND, activeIndices);
    }
    d3.select("#train-acc").text((trainAcc * 100).toFixed(1) + "%");
    d3.select("#test-acc").text((testAcc * 100).toFixed(1) + "%");
  }
}

function constructInputIds(): string[] {
  var result = [];
  var i, inputName;

  if (isMultiDim) {
    for (i = 0; i < activeFeatures.length; i++) {
      if (activeFeatures[i]) {
        result.push("input_" + i);
      }
    }
    return result;
  }

  for (inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function getActiveFeatureIndices(): number[] {
  var indices = [];
  var i;

  for (i = 0; i < activeFeatures.length; i++) {
    if (activeFeatures[i]) {
      indices.push(i);
    }
  }
  return indices;
}

function constructInput(x: number, y: number): number[] {
  var input = [];
  var inputName;

  for (inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  var i, j, activeIndices, point, activeInput, targetClass;

  iter++;

  if (isMultiDim) {
    activeIndices = getActiveFeatureIndices();

    if (isMultiClass) {
      // Multi-class classification training
      for (i = 0; i < trainDataND.length; i++) {
        point = trainDataND[i];
        activeInput = [];
        for (j = 0; j < activeIndices.length; j++) {
          activeInput.push(point.features[activeIndices[j]]);
        }
        nn.forwardPropMultiClass(network, activeInput);
        targetClass = Math.round(point.label);
        nn.backPropMultiClass(network, targetClass);
        if ((i + 1) % state.batchSize === 0) {
          nn.updateWeights(
            network,
            state.learningRate,
            state.regularizationRate
          );
        }
      }
      lossTrain = getLossNDMultiClass(network, trainDataND, activeIndices);
      lossTest = getLossNDMultiClass(network, testDataND, activeIndices);
    } else {
      // Binary classification or regression training
      for (i = 0; i < trainDataND.length; i++) {
        point = trainDataND[i];
        activeInput = [];
        for (j = 0; j < activeIndices.length; j++) {
          activeInput.push(point.features[activeIndices[j]]);
        }
        nn.forwardProp(network, activeInput);
        nn.backProp(network, point.label, nn.Errors.SQUARE);
        if ((i + 1) % state.batchSize === 0) {
          nn.updateWeights(
            network,
            state.learningRate,
            state.regularizationRate
          );
        }
      }
      lossTrain = getLossNDActive(network, trainDataND, activeIndices);
      lossTest = getLossNDActive(network, testDataND, activeIndices);
    }
  } else {
    // Original 2D training
    for (i = 0; i < trainData.length; i++) {
      point = trainData[i];
      var input = constructInput(point.x, point.y);
      nn.forwardProp(network, input);
      nn.backProp(network, point.label, nn.Errors.SQUARE);
      if ((i + 1) % state.batchSize === 0) {
        nn.updateWeights(network, state.learningRate, state.regularizationRate);
      }
    }
    // Compute the loss.
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
  }

  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  var weights = [];
  var layerIdx, currentLayer, i, node, j, output;

  for (layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    currentLayer = network[layerIdx];
    for (i = 0; i < currentLayer.length; i++) {
      node = currentLayer[i];
      for (j = 0; j < node.outputs.length; j++) {
        output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup?: boolean) {
  var suffix, numInputs, inputIds, numOutputs, activeIndices;
  var shape, outputActivation;

  if (onStartup === undefined) onStartup = false;

  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;

  if (isMultiDim) {
    activeIndices = getActiveFeatureIndices();
    numInputs = activeIndices.length;
    inputIds = constructInputIds();

    if (customDatasetInfo) {
      d3.select(".column.features h4").html(
        "Features <span class='feature-count'>(" +
          numInputs +
          "/" +
          customDatasetInfo.numFeatures +
          " active)</span>"
      );
    }

    if (state.problem === Problem.CLASSIFICATION && isMultiClass) {
      numOutputs = numClasses;
    } else {
      numOutputs = 1;
    }
  } else {
    numInputs = constructInput(0, 0).length;
    inputIds = constructInputIds();
    numOutputs = 1;
    isMultiClass = false;
    numClasses = 2;
  }

  shape = [numInputs].concat(state.networkShape).concat([numOutputs]);
  outputActivation =
    state.problem === Problem.REGRESSION
      ? nn.Activations.LINEAR
      : nn.Activations.TANH;

  if (isMultiClass) {
    outputActivation = nn.Activations.LINEAR;
  }

  network = nn.buildNetwork(
    shape,
    state.activation,
    outputActivation,
    state.regularization,
    inputIds,
    state.initZero
  );

  // Compute initial loss.
  if (isMultiDim) {
    activeIndices = getActiveFeatureIndices();
    if (isMultiClass) {
      lossTrain = getLossNDMultiClass(network, trainDataND, activeIndices);
      lossTest = getLossNDMultiClass(network, testDataND, activeIndices);
    } else {
      lossTrain = getLossNDActive(network, trainDataND, activeIndices);
      lossTest = getLossNDActive(network, testDataND, activeIndices);
    }
  } else {
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
  }

  drawNetwork(network);
  updateUI(true);

  if (isMultiDim) {
    updateActiveFeatureCount();
  }
}

function initTutorial() {
  var tutorial;

  if (state.tutorial == null || state.tutorial === "" || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  tutorial = d3.select("article").append("div").attr("class", "l--body");
  // Insert tutorial text.
  d3.html(
    "tutorials/" + state.tutorial + ".html",
    function (err, htmlFragment) {
      var title;
      if (err) throw err;
      tutorial.node().appendChild(htmlFragment);
      // If the tutorial has a <title> tag, set the page title to that.
      title = tutorial.select("title");
      if (title.size()) {
        d3.select("header h1")
          .style({ "margin-top": "20px", "margin-bottom": "20px" })
          .text(title.text());
        document.title = title.text();
      }
    }
  );
}

function drawDatasetThumbnails() {
  var dataset, canvas, dataGenerator, regDataset;

  function renderThumbnail(canvas, dataGenerator) {
    var w = 100;
    var h = 100;
    var context, data;

    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    context = canvas.getContext("2d");
    data = dataGenerator(200, 0);
    data.forEach(function (d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect((w * (d.x + 6)) / 12, (h * (d.y + 6)) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (dataset in datasets) {
      canvas = document.querySelector("canvas[data-dataset=" + dataset + "]");
      dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (regDataset in regDatasets) {
      canvas = document.querySelector(
        "canvas[data-regDataset=" + regDataset + "]"
      );
      dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  var hiddenProps, hideControlsEl, i, text, id, label, input, prop, controls;

  // Set display:none to all the UI elements that are hidden.
  hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(function (prop) {
    controls = d3.selectAll(".ui-" + prop);
    if (controls.size() === 0) {
      console.warn("0 html elements found with class .ui-" + prop);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classroom"
  // section.
  hideControlsEl = d3.select(".hide-controls");
  for (i = 0; i < HIDABLE_CONTROLS.length; i++) {
    text = HIDABLE_CONTROLS[i][0];
    id = HIDABLE_CONTROLS[i][1];

    label = hideControlsEl
      .append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    input = label
      .append("input")
      .attr({ type: "checkbox", class: "mdl-checkbox__input" });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    (function (capturedId) {
      input.on("change", function () {
        state.setHideProperty(capturedId, !this.checked);
        state.serialize();
        userHasInteracted();
        d3.select(".hide-controls-link").attr("href", window.location.href);
      });
    })(id);
    label.append("span").attr("class", "mdl-checkbox__label label").text(text);
  }
  d3.select(".hide-controls-link").attr("href", window.location.href);
}

function generateData(firstTime?: boolean) {
  var numSamples, generator, data, splitIndex;

  if (firstTime === undefined) firstTime = false;

  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  numSamples =
    state.problem === Problem.REGRESSION
      ? NUM_SAMPLES_REGRESS
      : NUM_SAMPLES_CLASSIFY;
  generator =
    state.problem === Problem.CLASSIFICATION ? state.dataset : state.regDataset;
  data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  splitIndex = Math.floor((data.length * state.percTrainData) / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

var firstInteraction = true;
var parametersChanged = false;

function userHasInteracted() {
  var page;

  if (!firstInteraction) return;
  firstInteraction = false;
  page = "index";
  if (state.tutorial != null && state.tutorial !== "") {
    page = "/v/tutorials/" + state.tutorial;
  }
  ga("set", "page", page);
  ga("send", "pageview", { sessionControl: "start" });
}

function simulationStarted() {
  ga("send", {
    hitType: "event",
    eventCategory: "Starting Simulation",
    eventAction: parametersChanged ? "changed" : "unchanged",
    eventLabel: state.tutorial == null ? "" : state.tutorial,
  });
  parametersChanged = false;
}

// Initialize the application
drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
