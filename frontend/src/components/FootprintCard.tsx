import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface FootprintCardProps {
  species: string;
  confidence: number;
}

const FootprintCard: React.FC<FootprintCardProps> = ({ species, confidence }) => {
  return (
    <View style={styles.card}>
      <Text style={styles.speciesText}>Species: {species}</Text>
      <Text style={styles.confidenceText}>Confidence: {confidence.toFixed(2)}%</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    padding: 16,
    margin: 8,
    borderRadius: 8,
    backgroundColor: '#f9f9f9',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  speciesText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  confidenceText: {
    fontSize: 16,
    color: '#555',
  },
});

export default FootprintCard;