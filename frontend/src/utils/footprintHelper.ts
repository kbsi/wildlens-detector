export const processFootprintData = (data: any) => {
    // Process the footprint data received from the backend
    // This function can include normalization, validation, etc.
    return {
        species: data.species,
        confidence: data.confidence,
        timestamp: new Date(data.timestamp),
    };
};

export const formatFootprintHistory = (history: any[]) => {
    // Format the footprint history for display
    return history.map(item => ({
        id: item.id,
        species: item.species,
        confidence: item.confidence,
        date: new Date(item.timestamp).toLocaleDateString(),
    }));
};

export const getFootprintIcon = (species: string) => {
    // Return the appropriate icon based on the species
    const icons: { [key: string]: string } = {
        deer: 'path/to/deer/icon',
        bear: 'path/to/bear/icon',
        // Add more species and their corresponding icons
    };
    return icons[species] || 'path/to/default/icon';
};