// API client service
import { LibrarySong, LibrarySongDetail, LibraryUploadResult, Playlist, PlaylistDetail, RecommendationResponse } from '../types/api';

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

  async uploadToLibrary(files: File[]): Promise<{ results: LibraryUploadResult[]; successful: number; failed: number }> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const response = await fetch(`${API_URL}/library/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Library upload failed');
    }
    return response.json();
  },

  async getLibrary(search = '', genre = ''): Promise<{ songs: LibrarySong[] }> {
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (genre) params.set('genre', genre);
    const query = params.toString();
    const response = await fetch(`${API_URL}/library${query ? `?${query}` : ''}`);
    if (!response.ok) throw new Error('Could not load library');
    return response.json();
  },

  async getLibrarySong(songId: string): Promise<LibrarySongDetail> {
    const response = await fetch(`${API_URL}/library/${encodeURIComponent(songId)}`);
    if (!response.ok) throw new Error('Could not load song details');
    return response.json();
  },

  async deleteLibrarySong(songId: string) {
    const response = await fetch(`${API_URL}/library/${encodeURIComponent(songId)}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Could not delete song');
    return response.json();
  },

  async getRecommendations(songId: string, limit = 10): Promise<RecommendationResponse> {
    const response = await fetch(`${API_URL}/recommendations/${encodeURIComponent(songId)}?limit=${limit}`);
    if (!response.ok) throw new Error('Could not load recommendations');
    return response.json();
  },

  async rebuildFeatures() {
    const response = await fetch(`${API_URL}/library/rebuild-features`, { method: 'POST' });
    if (!response.ok) throw new Error('Could not rebuild audio features');
    return response.json();
  },

  async getPlaylists(): Promise<{ playlists: Playlist[] }> {
    const response = await fetch(`${API_URL}/playlists`);
    if (!response.ok) throw new Error('Could not load playlists');
    return response.json();
  },

  async createPlaylist(name: string, description: string): Promise<PlaylistDetail> {
    const response = await fetch(`${API_URL}/playlists`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: description || null }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Could not create playlist');
    }
    return response.json();
  },

  async getPlaylist(playlistId: string): Promise<PlaylistDetail> {
    const response = await fetch(`${API_URL}/playlists/${encodeURIComponent(playlistId)}`);
    if (!response.ok) throw new Error('Could not load playlist');
    return response.json();
  },

  async updatePlaylist(playlistId: string, name: string, description: string | null): Promise<PlaylistDetail> {
    const response = await fetch(`${API_URL}/playlists/${encodeURIComponent(playlistId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    if (!response.ok) throw new Error('Could not update playlist');
    return response.json();
  },

  async deletePlaylist(playlistId: string) {
    const response = await fetch(`${API_URL}/playlists/${encodeURIComponent(playlistId)}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Could not delete playlist');
    return response.json();
  },

  async addSongsToPlaylist(playlistId: string, songIds: string[], recommendations = false) {
    const suffix = recommendations ? '/recommendations' : '/songs';
    const response = await fetch(`${API_URL}/playlists/${encodeURIComponent(playlistId)}${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ song_ids: songIds }),
    });
    if (!response.ok) throw new Error('Could not add songs to playlist');
    return response.json();
  },

  async removeSongFromPlaylist(playlistId: string, songId: string) {
    const response = await fetch(`${API_URL}/playlists/${encodeURIComponent(playlistId)}/songs/${encodeURIComponent(songId)}`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Could not remove song from playlist');
    return response.json();
  },

  async reorderPlaylist(playlistId: string, songIds: string[]) {
    const response = await fetch(`${API_URL}/playlists/${encodeURIComponent(playlistId)}/songs/reorder`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ song_ids: songIds }),
    });
    if (!response.ok) throw new Error('Could not reorder playlist');
    return response.json();
  },
};
