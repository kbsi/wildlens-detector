import React from 'react';
import { View, Text, Button } from 'react-native';
import { useNavigation } from '@react-navigation/native';

const HomeScreen = () => {
    const navigation = useNavigation();

    return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <Text style={{ fontSize: 24, marginBottom: 20 }}>Wildlife Footprint Identifier</Text>
            <Button
                title="Capture Footprint"
                onPress={() => navigation.navigate('Camera')}
            />
            <Button
                title="View Results"
                onPress={() => navigation.navigate('Result')}
            />
            <Button
                title="History"
                onPress={() => navigation.navigate('History')}
            />
        </View>
    );
};

export default HomeScreen;