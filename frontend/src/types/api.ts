// API types and interfaces
export interface PredictionResponse {
  model: string;
  genre: string;
  confidence: number;
  probabilities: Record<string, number>;
  windows?: WindowPrediction[];
}

export interface WindowPrediction {
  window_index?: number;
  start_seconds: number;
  end_seconds: number;
  genre: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface ComparisonResponse {
  predictions: PredictionResponse[];
  consensus_genre: string;
}

export interface HealthResponse {
  status: string;
  models_loaded: Record<string, boolean>;
}

export type ModelType = 'cnn' | 'lstm' | 'resnet' | 'compare';

export interface AnalysisState {
  loading: boolean;
  error: string | null;
  prediction: PredictionResponse | null;
  comparison: ComparisonResponse | null;
}

export interface LibrarySong {
  id: string;
  original_filename: string;
  stored_audio_path: string;
  uploaded_at: string;
  predicted_genre: string;
  prediction_confidence: number;
  probabilities: Record<string, number>;
  analysis_status: string;
}

export interface LibrarySongDetail extends LibrarySong {
  windows: WindowPrediction[];
}

export interface LibraryUploadResult {
  success: boolean;
  filename: string;
  song_id?: string;
  genre?: string;
  confidence?: number;
  error?: string;
}

export interface Recommendation {
  song_id: string;
  filename: string;
  predicted_genre: string;
  confidence: number;
  score: number;
  audio_similarity: number;
  genre_similarity: number;
  explanation: string;
}

export interface RecommendationResponse {
  source: {
    song_id: string;
    filename: string;
    predicted_genre: string;
    confidence: number;
  };
  recommendations: Recommendation[];
  formula: string;
  message: string;
  feature_errors: Array<{ song_id: string; filename: string; error: string }>;
}

export interface Playlist {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  song_count: number;
}

export interface PlaylistSong extends LibrarySong {
  position: number;
}

export interface PlaylistDetail extends Playlist {
  songs: PlaylistSong[];
}
