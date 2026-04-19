importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest");

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;

const INPUT_DIMENTIONS_W = 640;
const INPUT_DIMENTIONS_H = 640;
const THRESHOLD_SCORE = 0.41;

let _labels = [];
let _model = null;

async function loadModelAndLabels() {
  await tf.ready();

  _labels = await (await fetch(LABELS_PATH)).json();
  _model = await tf.loadGraphModel(MODEL_PATH);

  //aquecimento
  const dummyInput = tf.ones(_model.inputs[0].shape);
  await _model.executeAsync(dummyInput);
  tf.dispose(dummyInput);

  postMessage({ type: "model-loaded" });
}

function prepProcImg(input) {
  return tf.tidy(() => {
    const image = tf.browser.fromPixels(input);
    return tf.image
      .resizeBilinear(image, [INPUT_DIMENTIONS_W, INPUT_DIMENTIONS_H])
      .div(255)
      .expandDims(0);
  });
}

async function runInference(tensor) {
  const output = await _model.executeAsync(tensor);
  tf.dispose(tensor);

  const [boxes, scores, classes] = output.slice(0, 3);

  const [boxesData, scoresData, classesData] = await Promise.all([
    boxes.data(),
    scores.data(),
    classes.data(),
  ]);

  output.forEach((t) => t.dispose());

  return {
    boxes: boxesData,
    scores: scoresData,
    classes: classesData,
  };
}

function* processPrediction({ boxes, scores, classes }, width, height) {
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] < THRESHOLD_SCORE) continue;

    const label = _labels[classes[i]];
    if (label != "kite") continue; //MODELO RECONHECE O PATO COMO PIPA (KITE)

    let [x1, y1, x2, y2] = boxes.slice(i * 4, (i + 1) * 4);

    //AJUSTANDO AS DIMENSOES PARA O TAMANHO DA TELA
    x1 *= width;
    x2 *= width;
    y1 *= height;
    y2 *= height;

    //Meu calc: -> Em um cenario extremo pode dar overflow...
    // const centerWidth = (x1 + x2) / 2;
    // const centerHeight = (y1 + y2) / 2;

    //Calc Aula:
    const centerWidth = x1 + (x2 - x1) / 2;
    const centerHeight = y1 + (y2 - y1) / 2;

    debugger;
    yield {
      x: centerWidth,
      y: centerHeight,
      score: (scores[i] * 100).toFixed(2),
    };
  }
}

loadModelAndLabels();

self.onmessage = async ({ data }) => {
  if (!_model) return;
  if (data.type !== "predict") return;

  const { width, height } = data.image;
  const input = prepProcImg(data.image);

  const inferenceResults = await runInference(input);

  for (const prediction of processPrediction(inferenceResults, width, height)) {
    postMessage({
      type: "prediction",
      //   x: prediction.x,
      //   y: prediction.y,
      //   score: prediction.score,
      ...prediction,
    });
  }
};

console.log("🧠 YOLOv5n Web Worker initialized");
