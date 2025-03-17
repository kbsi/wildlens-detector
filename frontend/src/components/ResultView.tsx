import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const ResultView = ({ species, confidence, additionalInfo }) => {
    return (
        <View style={styles.container}>
            <Text style={styles.title}>Footprint Identification Result</Text>
            <Text style={styles.species}>Species: {species}</Text>
            <Text style={styles.confidence}>Confidence: {confidence}%</Text>
            {additionalInfo && (
                <Text style={styles.additionalInfo}>Additional Information: {additionalInfo}</Text>
            )}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
        backgroundColor: '#fff',
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 10,
    },
    species: {
        fontSize: 18,
        marginBottom: 5,
    },
    confidence: {
        fontSize: 18,
        marginBottom: 5,
    },
    additionalInfo: {
        fontSize: 16,
        marginTop: 10,
        textAlign: 'center',
    },
});

export default ResultView;