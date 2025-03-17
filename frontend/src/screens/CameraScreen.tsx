import React, { useState } from 'react';
import { View, Button, Image, StyleSheet } from 'react-native';
import Camera from '../components/Camera';
import { uploadFootprintImage } from '../services/api';
import { processImage } from '../utils/imageProcessor';

const CameraScreen = () => {
  const [imageUri, setImageUri] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCapture = async (uri) => {
    setImageUri(uri);
    setLoading(true);
    try {
      const processedImage = await processImage(uri);
      const response = await uploadFootprintImage(processedImage);
      // Handle response (e.g., navigate to ResultScreen with response data)
    } catch (error) {
      console.error('Error uploading image:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      {imageUri ? (
        <Image source={{ uri: imageUri }} style={styles.image} />
      ) : (
        <Camera onCapture={handleCapture} />
      )}
      <Button title="Retake" onPress={() => setImageUri(null)} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: '100%',
    height: '50%',
    resizeMode: 'contain',
  },
});

export default CameraScreen;