import { useEffect, useState } from "react";
import {
  View,
  Image,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { Camera } from "expo-camera";
import { SafeAreaView } from "react-native-safe-area-context";
import { loadModel } from "@/helper/loadModel";

export default function HomeScreen() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    const initialize = async () => {
      await loadModel();
    };

    initialize();
  }, []);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission Denied",
        "Permission to access media library is required!"
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"], // ✅ Fixed mediaTypes
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null);
      await handlePrediction(result.assets[0].uri);
    }
  };

  const takePhoto = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission Denied", "Camera permission is required!");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null);
      await handlePrediction(result.assets[0].uri);
    }
  };

  const handlePrediction = async (uri: string) => {
    try {
      console.log("Image picked:", uri);

      // ✅ Load model
      const model = await loadModel();
      if (!model) {
        console.error("Model not loaded ❌");
        Alert.alert("Error", "Model failed to load.");
        return;
      }

      // ✅ Read file as base64
      const base64 = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // ✅ Decode base64 into image tensor
      const imgBuffer = tf.util.encodeString(base64, "base64");
      const raw = new Uint8Array(imgBuffer);

      const imageTensor = tf.browser
        .fromPixels({ data: raw, width: 32, height: 32 })
        .toFloat()
        .div(tf.scalar(255))
        .expandDims(); // Add batch dimension

      console.log("Tensor shape:", imageTensor.shape);

      // ✅ Run model prediction
      const output = await model.run(imageTensor); // 🚨 Fix here!
      console.log("Prediction output:", output);

      // ✅ Handle output (modify based on model's output format)
      setResult(`Prediction: ${output[0]}`);
    } catch (error) {
      console.error("Prediction Error:", error);
      Alert.alert("Error", "Failed to process the image.");
    }
  };



  return (
    <SafeAreaView style={styles.container}>
      <TouchableOpacity style={styles.button} onPress={pickImage}>
        <Text style={styles.buttonText}>Select Image from Gallery</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={takePhoto}>
        <Text style={styles.buttonText}>Capture Photo</Text>
      </TouchableOpacity>

      {image && <Image source={{ uri: image }} style={styles.image} />}
      {result && <Text style={styles.resultText}>{result}</Text>}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    alignItems: "center",
    backgroundColor: "#f4f4f4",
  },
  button: {
    backgroundColor: "#4CAF50",
    padding: 12,
    marginVertical: 10,
    borderRadius: 8,
    alignItems: "center",
    width: "80%",
  },
  buttonText: {
    color: "#FFFFFF",
    fontSize: 16,
  },
  image: {
    width: 200,
    height: 200,
    marginTop: 16,
    borderRadius: 10,
  },
  resultText: {
    fontSize: 18,
    color: "#333",
    marginTop: 10,
  },
});
