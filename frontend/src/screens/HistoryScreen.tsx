import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { getHistory } from '../services/storage';
import FootprintCard from '../components/FootprintCard';

const HistoryScreen = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      const data = await getHistory();
      setHistory(data);
    };

    fetchHistory();
  }, []);

  const renderItem = ({ item }) => (
    <FootprintCard footprint={item} />
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>History of Identified Footprints</Text>
      <FlatList
        data={history}
        renderItem={renderItem}
        keyExtractor={(item) => item.id.toString()}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
});

export default HistoryScreen;