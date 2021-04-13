import * as tf from '@tensorflow/tfjs';
import imdbDictionary from './dict.txt';
import lstmModelConfig from './model.json';

async function createModel() {
  const model = await tf.loadModel(lstmModelConfig);
  return model;
}

function process(text) {
  let knownSymbols = text.replace(/[^a-zA-Z0-9\s]/, '');
  knownSymbols = knownSymbols.trim().split(/\s+/);

  for (let i = 0; i < knownSymbols.length; i++) {
    knownSymbols[i] = knownSymbols[i].toLowerCase();
  }

  return knownSymbols;
}

function createSequences({ dictionary, text }) {
  const maxTokens = 40;
  const words = process(text);
  const wordsAsNumbers = Array.from({ length: maxTokens }, () => 0);
  const start = maxTokens - words.length;
  const wordsFromDictionary = Object.keys(dictionary);

  for (let i = 0; i < words.length; i++) {
    if (wordsFromDictionary.includes(words[i])) {
      wordsAsNumbers[i + start] = dictionary[words[i]];
    }
  }

  return wordsAsNumbers;
}

function success(data) {
  const wordToVector = {};
  const lines = data.split(/\r?\n|\r/);

  for (let i = 0; i < lines.length; i++) {
    const key = lines[i].split(',')[0];
    const value = lines[i].split(',')[1];

    if (key == '') continue;

    wordToVector[key] = parseInt(value);
  }

  return wordToVector;
}

const defaultLogger = {
  info: console.info,
  error: console.error,
};

export class SentimentPredictor {
  constructor({ logger = defaultLogger } = {}) {
    this.dictionary = null;
    this.model = null;
    this.logger = logger;
    this.loadingPromise = this.init()
      .then(() => {
        this.logger.info('Sentiment predictor initialized');
      })
      .catch((error) => {
        this.logger.error('Sentiment predictor failed to initialize');
        this.logger.error(error);
      });
  }

  static create() {
    return new SentimentPredictor();
  }

  async init() {
    await this.loadDictionary().then(() => {
      this.logger.info('Dictionary loaded successfully');
    });
    await this.initModel();
  }

  async waitForLoaded() {
    return this.loadingPromise;
  }

  async loadDictionary() {
    this.dictionary = success(imdbDictionary);
  }

  async initModel() {
    this.model = await createModel();
  }

  performMeasurement(text) {
    const sequence = createSequences({ dictionary: this.dictionary, text });
    let input = tf.tensor(sequence);
    input = input.expandDims(0);
    return this.model.predict(input).dataSync();
  }
}

export const createSentimentPredictor = async (params) => {
  const sentimentPredictor = new SentimentPredictor(params);
  await sentimentPredictor.waitForLoaded();
  return sentimentPredictor;
};
