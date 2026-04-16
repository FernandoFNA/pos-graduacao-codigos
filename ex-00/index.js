import tf, { softmax } from "@tensorflow/tfjs-node";

async function trainModel(input, output) {
  const model = tf.sequential();

  //Inputs [7] (normalized_age;color1;color2;color3;location1;loocation2;location3)

  //Utilizando 80 neuronios

  //ReLU corta dados sujos, dados negativos ou zerados ignora
  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  //Output: 3 neuronios (premium, medium, basic)
  //ACTIVATION: SOFTMAX -> probabilidade de ser cada output
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  //OPTIMIZAR: ADAM -> umas das N formas de treinar redes neurais (aprende com historico de erros e acertos)
  //Loss: categoricalCrossentropy -> valida o que ele "acha" com o "real"
  //metrics: accuracy -> quanto ele esta acertando
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  //TREINAMENTO
  await model.fit(inputXs, outputYs, {
    verbose: 0, //sem log interno
    epochs: 100, //passara 100x pelos dados
    shuffle: true, //vai randomizar a ordem dos dados (evita bias)
    callbacks: {
      //   onEpochEnd: (epochs, log) => //caso queira logs so descomentar
      //     console.log(`Epoch: ${epochs} | loss = ${log.loss}`),
    },
  });

  return model;
}

async function predict(model, pessoa) {
  //transformar array js para tensor (tfjs)
  const tfInput = tf.tensor2d(pessoa);

  //fazer o predict
  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  //console.log(predArray); //log da predicao
  return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// COMENTANDO OS PRINTS
// inputXs.print();
// outputYs.print();

const model = await trainModel(inputXs, outputYs);

//criando PESSOA que n foi utilizada no treinamento
const pessoaTeste1 = {
  nome: "Bilu Teteia",
  idade: 28,
  cor: "verde",
  localizacao: "Curitiba",
};

//mormalizar campo idade [ (atual - min) / (max - min) ] -> [ (28 - 25) / (40 - 25) ] = 0.2
const pessoaNormalizado = [
  [
    0.2, //idade normalizada
    1, //cor: azul
    0, //cor: vermelho
    0, //cor: verde
    1, //local: Sao Paulo
    0, //local: Rio
    0, //local: Curitiba
  ],
];

const predictions = await predict(model, pessoaNormalizado);
const result = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(3)}%)`)
  .join("\n");

console.log(result);
