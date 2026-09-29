// API client service
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const apiClient = {
  async getHealth() {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) throw new Error('Health check failed');
    return response.json();
  },

  async predict(file: File, model: 'cnn' | 'lstm' | 'resnet') {
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('model', model);

    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Prediction failed');
    }

    return response.json();
  },

  async compare(file: File) {
    const formData = new FormData();
    formData.append('audio', file);

    const response = await fetch(`${API_URL}/compare`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Comparison failed');
    }

    return response.json();
  },
};
