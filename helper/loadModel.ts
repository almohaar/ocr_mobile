import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import * as FileSystem from "expo-file-system";
import { Asset } from "expo-asset";
import { loadTensorflowModel } from "react-native-fast-tflite";
import { Tensor } from "@tensorflow/tfjs";

let model: any;

const classNames: string[] = [
  "a",
  "A",
  "b",
  "B",
  "d",
  "D",
  "e",
  "E",
  "f",
  "F",
  "g",
  "G",
  "gb",
  "GB",
  "h",
  "H",
  "i",
  "I",
  "j",
  "J",
  "k",
  "K",
  "l",
  "L",
  "m",
  "M",
  "n",
  "N",
  "o",
  "O",
  "p",
  "P",
  "r",
  "R",
  "s",
  "S",
  "t",
  "T",
  "u",
  "U",
  "w",
  "W",
  "y",
  "Y",
  "á",
  "Á",
  "à",
  "À",
  "é",
  "É",
  "è",
  "È",
  "ẹ",
  "Ẹ",
  "í",
  "Í",
  "ì",
  "Ì",
  "ó",
  "Ó",
  "ò",
  "Ò",
  "ọ",
  "Ọ",
  "ṣ",
  "Ṣ",
  "ú",
  "Ú",
  "ù",
  "Ù",
];

const loadModelToFileSystem = async () => {
  const modelAsset = (Asset as any).fromModule(
    require("../assets/model/yoruba_ocr_model.tflite")
  );
  await modelAsset.downloadAsync();

  console.log("Model asset:", modelAsset); // Debugging

  if (!modelAsset.localUri) {
    throw new Error("Failed to resolve model file path");
  }

  const modelPath = FileSystem.documentDirectory + "yoruba_ocr_model.tflite";

  const fileInfo = await FileSystem.getInfoAsync(modelPath);
  console.log("File Info:", fileInfo); // Debugging

  if (!fileInfo.exists) {
    await FileSystem.copyAsync({
      from: modelAsset.localUri,
      to: modelPath,
    });
  }

  return modelPath;
};



export const loadModel = async () => {
  try {
    await tf.ready();
    console.log("TensorFlow.js is ready ");

    const modelPath = require("../assets/model/yoruba_ocr_model.tflite");
    

    console.log("Model loaded successfully ");
    return model;
  } catch (error) {
    console.error("Failed to load model :", error);
  }
};

export const predict = async (inputTensor: Tensor): Promise<string | null> => {
  if (!model) {
    await loadModel();
  }

  // Convert tensor to Float32 array
  const inputData = inputTensor.dataSync() as Float32Array;

  // Run model prediction
  const output = await model.run(inputData);

  console.log("Raw prediction:", output);

  // Get the index of the highest value in the output
  const predictedClassIndex = output[0].indexOf(Math.max(...output[0]));

  return classNames[predictedClassIndex] || null;
};

export const testModel = async () => {
  const model = await loadModel();

  if (model) {
    try {
      console.log("Testing model prediction...");

      // Create an input tensor with shape [1, 32, 32, 1] (example for grayscale)
      const inputTensor = new Float32Array(32 * 32).fill(0); // Example tensor data
      const output = await model.run(inputTensor);

      console.log("Prediction output:", output);
    } catch (error) {
      console.error("Prediction failed:", error);
    }
  }
};

