import { useState, useEffect } from 'react';
import Header from './components/Header';
import AudioUploader from './components/AudioUploader';
import ModelSelector from './components/ModelSelector';
import AnalysisButton from './components/AnalysisButton';
import PredictionResult from './components/PredictionResult';
import ModelComparison from './components/ModelComparison';
import ModelInfo from './components/ModelInfo';
import TechnicalDetails from './components/TechnicalDetails';
import LibraryPanel from './components/LibraryPanel';
import PlaylistPanel from './components/PlaylistPanel';
import MusicPlayer from './components/MusicPlayer';
import { apiClient } from './services/api';
import { LibraryUploadResult, PredictionResponse, ComparisonResponse } from './types/api';
import './index.css';

function App() {
  const [apiStatus, setApiStatus] = useState(false);
  const [audioFiles, setAudioFiles] = useState<File[]>([]);
  const [selectedModel, setSelectedModel] = useState('resnet');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryResults, setLibraryResults] = useState<LibraryUploadResult[]>([]);
  const [libraryRefreshToken, setLibraryRefreshToken] = useState(0);
  const [playlistRefreshToken, setPlaylistRefreshToken] = useState(0);

  const audioFile = audioFiles[0] || null;

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.getHealth();
        setApiStatus(true);
      } catch {
        setApiStatus(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = (files: File[]) => {
    setAudioFiles(files);
    setPrediction(null);
    setComparison(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (audioFiles.length !== 1) {
      setError('Select exactly one file for model analysis, or save the selection to your library.');
      return;
    }
    if (!audioFile) return;

    setLoading(true);
    setError(null);
    setPrediction(null);
    setComparison(null);

    try {
      if (selectedModel === 'compare') {
        const result = await apiClient.compare(audioFile);
        setComparison(result);
      } else {
        const result = await apiClient.predict(audioFile, selectedModel as 'cnn' | 'lstm' | 'resnet');
        setPrediction(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLibraryUpload = async () => {
    if (audioFiles.length === 0) return;

    setLibraryLoading(true);
    setLibraryResults(audioFiles.map((file) => ({
      success: false,
      filename: file.name,
      error: 'Processing...',
    })));
    try {
      const result = await apiClient.uploadToLibrary(audioFiles);
      setLibraryResults(result.results);
      setLibraryRefreshToken((value) => value + 1);
      setPlaylistRefreshToken((value) => value + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Library upload failed');
    } finally {
      setLibraryLoading(false);
    }
  };

  const showResults = !loading && (prediction || comparison);

  return (
    <div className="animated-bg min-h-screen">
      <Header apiStatus={apiStatus} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pb-40">
        {/* Hero Section */}
        <section className="mb-16 text-center fade-in">
          <h2 className="text-5xl md:text-6xl font-bold mb-4 text-slate-100">
            What does your
            <span className="gradient-text"> music sound like?</span>
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Upload a track for full-song analysis or save multiple tracks to your persistent library.
            Three architectures work together to classify your music.
          </p>
        </section>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          {/* Left Column - Upload & Controls */}
          <div className="lg:col-span-1 space-y-6">
            <AudioUploader
              onFileSelect={handleFileSelect}
              disabled={loading}
              fileName={audioFiles.map((file) => file.name).join(', ')}
            />

            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
              disabled={loading || !audioFile}
            />

            <AnalysisButton
              onClick={handleAnalyze}
              loading={loading}
              disabled={audioFiles.length !== 1 || !apiStatus}
            />

            <button
              onClick={handleLibraryUpload}
              disabled={audioFiles.length === 0 || libraryLoading || !apiStatus}
              className="w-full py-3 px-6 rounded-lg font-semibold border border-cyan-400/40 text-cyan-200 hover:bg-cyan-400/10 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {libraryLoading ? 'Saving to library...' : `Save ${audioFiles.length || ''} file${audioFiles.length === 1 ? '' : 's'} to library`}
            </button>

            {libraryResults.length > 0 && (
              <div className="space-y-2 text-sm">
                {libraryResults.map((result) => (
                  <div key={result.filename} className={result.success ? 'text-emerald-300' : 'text-orange-300'}>
                    {result.filename}: {result.success ? `${result.genre} saved` : result.error}
                  </div>
                ))}
              </div>
            )}

            {!apiStatus && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                <p className="text-sm text-red-300">API is currently unavailable</p>
              </div>
            )}

            {error && (
              <div className="p-4 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                <p className="text-sm text-orange-300">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2">
            {showResults ? (
              prediction ? (
                <PredictionResult prediction={prediction} />
              ) : comparison ? (
                <ModelComparison comparison={comparison} />
              ) : null
            ) : loading ? (
              <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center min-h-96 fade-in">
                <div className="flex gap-2 mb-6">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className="w-3 h-12 bg-gradient-to-t from-violet-500 to-cyan-500 rounded-full wave-bar"
                      style={{ animationDelay: `${i * 0.1}s` }}
                    ></div>
                  ))}
                </div>
                <p className="text-slate-300 font-semibold">Analyzing audio...</p>
                <p className="text-sm text-slate-500 mt-2">Extracting acoustic features and running neural networks</p>
              </div>
            ) : (
              <div className="glass-panel p-12 rounded-2xl flex items-center justify-center min-h-96 text-center">
                <div>
                  <p className="text-slate-400 text-lg">Upload an audio file and select a model to begin</p>
                  <p className="text-slate-500 text-sm mt-2">Supports MP3, WAV, FLAC, and other common audio formats</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Model Info */}
        <ModelInfo />

        {/* Technical Details */}
        <TechnicalDetails />

        <LibraryPanel refreshToken={libraryRefreshToken} onPlaylistChange={() => setPlaylistRefreshToken((value) => value + 1)} />

        <PlaylistPanel refreshToken={playlistRefreshToken} />

        {/* Footer */}
        <footer className="mt-20 pt-12 border-t border-slate-800 text-center text-slate-500 text-sm">
          <p>GenreLab © 2024 • Music Genre Classification Capstone</p>
          <p className="mt-2">CNN • LSTM • ResNet18</p>
        </footer>
      </main>
      <MusicPlayer />
    </div>
  );
}

export default App;
