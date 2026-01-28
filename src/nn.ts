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

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
export class Node {
  id: string;
  /** List of input links. */
  inputLinks: Link[] = [];
  bias = 0.1;
  /** List of output links. */
  outputs: Link[] = [];
  totalInput: number;
  output: number;
  /** Error derivative with respect to this node's output. */
  outputDer = 0;
  /** Error derivative with respect to this node's total input. */
  inputDer = 0;
  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  accInputDer = 0;
  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  numAccumulatedDers = 0;
  /** Activation function that takes total input and returns node's output */
  activation: ActivationFunction;

  /**
   * Creates a new node with the provided id and activation function.
   */
  constructor(id: string, activation: ActivationFunction, initZero?: boolean) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0;
    }
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }
}

/**
 * An error function and its derivative.
 */
export interface ErrorFunction {
  error: (output: number, target: number) => number;
  der: (output: number, target: number) => number;
}

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output: (input: number) => number;
  der: (input: number) => number;
}

/** Function that computes a penalty cost for a given weight in the network. */
export interface RegularizationFunction {
  output: (weight: number) => number;
  der: (weight: number) => number;
}

/** Built-in error functions */
export class Errors {
  public static SQUARE: ErrorFunction = {
    error: (output: number, target: number) =>
      0.5 * Math.pow(output - target, 2),
    der: (output: number, target: number) => output - target,
  };
}

/**
 * Multi-class error functions
 */
export class MultiClassErrors {
  /**
   * Cross-entropy loss for multi-class classification
   * @param outputs Array of softmax probabilities
   * @param targetClass The correct class index (0, 1, 2, ...)
   */
  public static crossEntropy(outputs: number[], targetClass: number): number {
    // Clamp to avoid log(0)
    let p = Math.max(outputs[targetClass], 1e-15);
    return -Math.log(p);
  }

  /**
   * Derivative of cross-entropy with softmax.
   * For softmax + cross-entropy, the derivative simplifies to: output - target
   * where target is 1 for the correct class and 0 otherwise.
   */
  public static crossEntropyDer(
    outputs: number[],
    targetClass: number,
    outputIndex: number
  ): number {
    let target = outputIndex === targetClass ? 1 : 0;
    return outputs[outputIndex] - target;
  }
}

/**
 * Compute softmax over an array of values.
 * Subtracts max for numerical stability.
 */
export function softmax(values: number[]): number[] {
  // Subtract max for numerical stability
  let max = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }

  let exps: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    let e = Math.exp(values[i] - max);
    exps.push(e);
    sum += e;
  }

  let result: number[] = [];
  for (let i = 0; i < exps.length; i++) {
    result.push(exps[i] / sum);
  }
  return result;
}

/** Polyfill for TANH */
(Math as any).tanh =
  (Math as any).tanh ||
  function (x) {
    if (x === Infinity) return 1;
    if (x === -Infinity) return -1;
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  };

/** Built-in activation functions */
export class Activations {
  public static TANH: ActivationFunction = {
    output: (x) => (Math as any).tanh(x),
    der: (x) => {
      let output = Activations.TANH.output(x);
      return 1 - output * output;
    },
  };
  public static RELU: ActivationFunction = {
    output: (x) => Math.max(0, x),
    der: (x) => (x <= 0 ? 0 : 1),
  };
  public static LEAKY_RELU: ActivationFunction = {
    output: (x) => (x <= 0 ? 0.01 * x : x),
    der: (x) => (x <= 0 ? 0.01 : 1),
  };
  public static ELU: ActivationFunction = {
    output: (x) => (x <= 0 ? Math.E ** x - 1 : x),
    der: (x) => {
      let output = Activations.ELU.output(x);
      return x <= 0 ? output + 1 : 1;
    },
  };
  public static SWISH: ActivationFunction = {
    output: (x) => x / (1 + Math.E ** -x),
    der: (x) => {
      let output = Activations.SWISH.output(x);
      return output + (1 / (1 + Math.E ** -x)) * (1 - output);
    },
  };
  public static SOFTPLUS: ActivationFunction = {
    output: (x) => Math.log(1 + Math.E ** x),
    der: (x) => 1 / (1 + Math.E ** -x),
  };
  public static SIGMOID: ActivationFunction = {
    output: (x) => 1 / (1 + Math.exp(-x)),
    der: (x) => {
      let output = Activations.SIGMOID.output(x);
      return output * (1 - output);
    },
  };
  public static LINEAR: ActivationFunction = {
    output: (x) => x,
    der: (x) => 1,
  };
}

/** Built-in regularization functions */
export class RegularizationFunction {
  public static L1: RegularizationFunction = {
    output: (w) => Math.abs(w),
    der: (w) => (w < 0 ? -1 : w > 0 ? 1 : 0),
  };
  public static L2: RegularizationFunction = {
    output: (w) => 0.5 * w * w,
    der: (w) => w,
  };
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight = Math.random() - 0.5;
  isDead = false;
  /** Error derivative with respect to this weight. */
  errorDer = 0;
  /** Accumulated error derivative since the last update. */
  accErrorDer = 0;
  /** Number of accumulated derivatives since the last update. */
  numAccumulatedDers = 0;
  regularization: RegularizationFunction;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  constructor(
    source: Node,
    dest: Node,
    regularization: RegularizationFunction,
    initZero?: boolean
  ) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0;
    }
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 * @param initZero If true, initialize all weights and biases to zero.
 */
export function buildNetwork(
  networkShape: number[],
  activation: ActivationFunction,
  outputActivation: ActivationFunction,
  regularization: RegularizationFunction,
  inputIds?: string[],
  initZero?: boolean
): Node[][] {
  let numLayers = networkShape.length;
  let id = 1;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];

  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];

    for (let i = 0; i < numNodes; i++) {
      let nodeId: string;
      if (isInputLayer) {
        nodeId = inputIds && inputIds[i] ? inputIds[i] : "input_" + i;
      } else {
        nodeId = id.toString();
        id++;
      }

      let node = new Node(
        nodeId,
        isOutputLayer ? outputActivation : activation,
        initZero
      );
      currentLayer.push(node);

      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node, regularization, initZero);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
      }
    }
  }
  return network;
}

/**
 * Build network for multi-dimensional input.
 *
 * @param numInputs Number of input features
 * @param hiddenLayers Array of hidden layer sizes
 * @param activation Activation function for hidden layers
 * @param outputActivation Activation function for output
 * @param regularization Regularization function
 * @param initZero If true, initialize all weights and biases to zero.
 */
export function buildNetworkND(
  numInputs: number,
  hiddenLayers: number[],
  activation: ActivationFunction,
  outputActivation: ActivationFunction,
  regularization: RegularizationFunction,
  initZero?: boolean
): Node[][] {
  // Create input IDs
  let inputIds: string[] = [];
  for (let i = 0; i < numInputs; i++) {
    inputIds.push("input_" + i);
  }

  // Build shape array
  let shape = [numInputs].concat(hiddenLayers).concat([1]);

  return buildNetwork(
    shape,
    activation,
    outputActivation,
    regularization,
    inputIds,
    initZero
  );
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network (single value for binary classification or regression).
 */
export function forwardProp(network: Node[][], inputs: number[]): number {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error(
      "The number of inputs (" +
        inputs.length +
        ") must match the number of nodes in the input layer (" +
        inputLayer.length +
        ")"
    );
  }

  // Update the input layer.
  for (let i = 0; i < inputLayer.length; i++) {
    let node = inputLayer[i];
    node.output = inputs[i];
  }

  // Update all the nodes in subsequent layers.
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * Forward propagation for multi-class classification.
 * Returns array of softmax probabilities (one per class).
 *
 * @param network The neural network.
 * @param inputs The input array.
 * @return Array of softmax probabilities for each class.
 */
export function forwardPropMultiClass(
  network: Node[][],
  inputs: number[]
): number[] {
  let inputLayer = network[0];
  if (inputs.length !== inputLayer.length) {
    throw new Error(
      "The number of inputs (" +
        inputs.length +
        ") must match the number of nodes in the input layer (" +
        inputLayer.length +
        ")"
    );
  }

  // Set input layer outputs
  for (let i = 0; i < inputLayer.length; i++) {
    inputLayer[i].output = inputs[i];
  }

  // Forward through hidden layers (not including output layer)
  for (let layerIdx = 1; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      currentLayer[i].updateOutput();
    }
  }

  // For output layer, compute raw values (before softmax)
  let outputLayer = network[network.length - 1];
  let rawOutputs: number[] = [];
  for (let i = 0; i < outputLayer.length; i++) {
    let node = outputLayer[i];
    // Compute total input but don't apply activation
    node.totalInput = node.bias;
    for (let j = 0; j < node.inputLinks.length; j++) {
      let link = node.inputLinks[j];
      node.totalInput += link.weight * link.source.output;
    }
    rawOutputs.push(node.totalInput);
  }

  // Apply softmax
  let softmaxOutputs = softmax(rawOutputs);

  // Store softmax outputs in nodes
  for (let i = 0; i < outputLayer.length; i++) {
    outputLayer[i].output = softmaxOutputs[i];
  }

  return softmaxOutputs;
}

/**
 * Back propagation for multi-class classification with cross-entropy loss.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node and each weight.
 *
 * @param network The neural network.
 * @param targetClass The correct class index (0, 1, 2, ...).
 */
export function backPropMultiClass(
  network: Node[][],
  targetClass: number
): void {
  let outputLayer = network[network.length - 1];

  // Get softmax outputs
  let outputs: number[] = [];
  for (let i = 0; i < outputLayer.length; i++) {
    outputs.push(outputLayer[i].output);
  }

  // Compute output layer derivatives (softmax + cross-entropy simplifies nicely)
  for (let i = 0; i < outputLayer.length; i++) {
    let node = outputLayer[i];
    // For softmax + cross-entropy: derivative = output - target
    let target = i === targetClass ? 1 : 0;
    node.outputDer = outputs[i] - target;
    node.inputDer = node.outputDer; // No activation derivative needed (absorbed into softmax)
    node.accInputDer += node.inputDer;
    node.numAccumulatedDers++;
  }

  // Compute weight derivatives for output layer
  for (let i = 0; i < outputLayer.length; i++) {
    let node = outputLayer[i];
    for (let j = 0; j < node.inputLinks.length; j++) {
      let link = node.inputLinks[j];
      if (link.isDead) continue;
      link.errorDer = node.inputDer * link.source.output;
      link.accErrorDer += link.errorDer;
      link.numAccumulatedDers++;
    }
  }

  // Propagate to previous layers (go through the layers backwards)
  for (let layerIdx = network.length - 2; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];
    let nextLayer = network[layerIdx + 1];

    // Compute output derivatives for current layer
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let link = node.outputs[j];
        node.outputDer += link.weight * link.dest.inputDer;
      }
    }

    // Compute input derivatives
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // Compute weight derivatives
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) continue;
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
  }
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 *
 * @param network The neural network.
 * @param target The target value.
 * @param errorFunc The error function to use.
 */
export function backProp(
  network: Node[][],
  target: number,
  errorFunc: ErrorFunction
): void {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  let outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Go through the layers backwards.
  for (let layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    let currentLayer = network[layerIdx];

    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // Error derivative with respect to each weight coming into the node.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) continue;
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }

    if (layerIdx === 1) continue;

    let prevLayer = network[layerIdx - 1];
    for (let i = 0; i < prevLayer.length; i++) {
      let node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0;
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 *
 * @param network The neural network.
 * @param learningRate The learning rate.
 * @param regularizationRate The regularization rate.
 */
export function updateWeights(
  network: Node[][],
  learningRate: number,
  regularizationRate: number
) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];

      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        node.bias -=
          (learningRate * node.accInputDer) / node.numAccumulatedDers;
        node.accInputDer = 0;
        node.numAccumulatedDers = 0;
      }

      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        if (link.isDead) continue;

        let regulDer = link.regularization
          ? link.regularization.der(link.weight)
          : 0;

        if (link.numAccumulatedDers > 0) {
          // Update the weight based on dE/dw.
          link.weight =
            link.weight -
            (learningRate / link.numAccumulatedDers) * link.accErrorDer;
          // Further update the weight based on regularization.
          let newLinkWeight =
            link.weight - learningRate * regularizationRate * regulDer;

          if (
            link.regularization === RegularizationFunction.L1 &&
            link.weight * newLinkWeight < 0
          ) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0;
          link.numAccumulatedDers = 0;
        }
      }
    }
  }
}

/** Iterates over every node in the network. */
export function forEachNode(
  network: Node[][],
  ignoreInputs: boolean,
  accessor: (node: Node) => any
) {
  for (
    let layerIdx = ignoreInputs ? 1 : 0;
    layerIdx < network.length;
    layerIdx++
  ) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}

/** Returns all output nodes (for multi-class). */
export function getOutputNodes(network: Node[][]): Node[] {
  return network[network.length - 1];
}

/** Returns the number of output nodes. */
export function getNumOutputs(network: Node[][]): number {
  return network[network.length - 1].length;
}

/**
 * Get total number of parameters in the network (weights + biases).
 */
export function getNumParameters(network: Node[][]): number {
  let count = 0;
  forEachNode(network, true, (node) => {
    count++; // bias
    count += node.inputLinks.length; // weights
  });
  return count;
}

/**
 * Get network info as string.
 */
export function getNetworkInfo(network: Node[][]): string {
  let layerSizes: number[] = [];
  for (let i = 0; i < network.length; i++) {
    layerSizes.push(network[i].length);
  }
  return (
    "Network shape: [" +
    layerSizes.join(", ") +
    "], Parameters: " +
    getNumParameters(network)
  );
}
