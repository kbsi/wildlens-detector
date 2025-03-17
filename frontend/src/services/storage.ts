import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = '@wildlife_footprints_history';

export const saveFootprintHistory = async (data) => {
    try {
        const existingData = await getFootprintHistory();
        const updatedData = existingData ? [...existingData, data] : [data];
        await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updatedData));
    } catch (error) {
        console.error('Error saving footprint history:', error);
    }
};

export const getFootprintHistory = async () => {
    try {
        const jsonValue = await AsyncStorage.getItem(STORAGE_KEY);
        return jsonValue != null ? JSON.parse(jsonValue) : null;
    } catch (error) {
        console.error('Error retrieving footprint history:', error);
        return null;
    }
};

export const clearFootprintHistory = async () => {
    try {
        await AsyncStorage.removeItem(STORAGE_KEY);
    } catch (error) {
        console.error('Error clearing footprint history:', error);
    }
};