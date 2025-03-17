import axios from 'axios';

const API_URL = 'http://localhost:8000/api'; // Update with your backend URL

export const uploadFootprintImage = async (imageData) => {
    try {
        const response = await axios.post(`${API_URL}/footprints/upload`, imageData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error uploading footprint image:', error);
        throw error;
    }
};

export const getSpeciesInfo = async (speciesId) => {
    try {
        const response = await axios.get(`${API_URL}/footprints/${speciesId}`);
        return response.data;
    } catch (error) {
        console.error('Error fetching species information:', error);
        throw error;
    }
};