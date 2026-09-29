// API types and interfaces
export interface PredictionResponse {
  model: string;
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
