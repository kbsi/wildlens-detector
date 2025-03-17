import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import ResultView from '../components/ResultView';

const ResultScreen = ({ route }) => {
    const { footprintData } = route.params;

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Footprint Identification Result</Text>
            <ResultView data={footprintData} />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
        backgroundColor: '#f5f5f5',
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 20,
    },
});

export default ResultScreen;