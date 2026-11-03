import { useEffect, useState } from 'react';
import { Database, Play, Sparkles, Trash2, X } from 'lucide-react';
import { apiClient } from '../services/api';
import { LibrarySong, LibrarySongDetail, Playlist, RecommendationResponse } from '../types/api';
import { useMusicPlayer } from '../player/MusicPlayerContext';

const GENRES = [
  'Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic',
  'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
  'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken',
];

const formatPercent = (value: unknown) => {
  const number = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return `${(number * 100).toFixed(1)}%`;
};

interface LibraryPanelProps {
  refreshToken: number;
  onPlaylistChange?: () => void;
}

export default function LibraryPanel({ refreshToken, onPlaylistChange }: LibraryPanelProps) {
  const { currentSong, playSong, syncQueue } = useMusicPlayer();
  const [songs, setSongs] = useState<LibrarySong[]>([]);
  const [search, setSearch] = useState('');
  const [genre, setGenre] = useState('');
  const [selectedSong, setSelectedSong] = useState<LibrarySongDetail | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendationTarget, setRecommendationTarget] = useState<string | null>(null);
  const [playlistOptions, setPlaylistOptions] = useState<Playlist[]>([]);
  const [selectedRecommendationIds, setSelectedRecommendationIds] = useState<string[]>([]);
  const [recommendationPlaylistId, setRecommendationPlaylistId] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    apiClient.getLibrary(search, genre)
      .then((result) => {
        if (active) {
          setSongs(result.songs);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (active) setError(err instanceof Error ? err.message : 'Could not load library');
      });
    return () => {
      active = false;
    };
  }, [genre, refreshToken, search]);

  const openDetails = async (songId: string) => {
    try {
      setSelectedSong(await apiClient.getLibrarySong(songId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load song details');
    }
  };

  const deleteSong = async (song: LibrarySong) => {
    if (!window.confirm(`Delete ${song.original_filename}?`)) return;
    try {
      await apiClient.deleteLibrarySong(song.id);
      setSongs((current) => current.filter((item) => item.id !== song.id));
      if (selectedSong?.id === song.id) setSelectedSong(null);
      if (currentSong?.id === song.id) syncQueue([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete song');
    }
  };

  const loadRecommendations = async (song: LibrarySong) => {
    setRecommendationLoading(true);
    setRecommendationTarget(song.original_filename);
    setRecommendations(null);
    setSelectedRecommendationIds([]);
    try {
      const [result, playlistResult] = await Promise.all([
        apiClient.getRecommendations(song.id),
        apiClient.getPlaylists(),
      ]);
      setRecommendations(result);
      setPlaylistOptions(playlistResult.playlists);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load recommendations');
    } finally {
      setRecommendationLoading(false);
    }
  };

  const addSelectedRecommendations = async () => {
    if (!recommendationPlaylistId || selectedRecommendationIds.length === 0) return;
    try {
      await apiClient.addSongsToPlaylist(recommendationPlaylistId, selectedRecommendationIds, true);
      setSelectedRecommendationIds([]);
      onPlaylistChange?.();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not add recommendations to playlist');
    }
  };

  const recommendationItems = recommendations && Array.isArray(recommendations.recommendations)
    ? recommendations.recommendations.filter((item) => item && typeof item === 'object')
    : [];
  const recommendationSource = recommendations?.source?.filename || recommendationTarget || 'selected song';

  return (
    <section className="mt-16 fade-in">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between mb-6">
        <div>
          <p className="text-sm uppercase tracking-widest text-cyan-400">Persistent collection</p>
          <h2 className="text-3xl font-bold text-slate-100">Music library</h2>
        </div>
        <div className="flex flex-col sm:flex-row gap-3 w-full sm:w-auto">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search filenames"
            className="bg-slate-900/80 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400"
          />
          <select
            value={genre}
            onChange={(event) => setGenre(event.target.value)}
            className="bg-slate-900/80 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400"
          >
            <option value="">All genres</option>
            {GENRES.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>
      </div>

      {error && <p className="mb-4 text-sm text-orange-300">{error}</p>}
      {songs.length === 0 ? (
        <div className="glass-panel rounded-2xl p-8 text-center text-slate-400">
          <Database className="mx-auto mb-3 text-slate-500" />
          No saved songs match this filter.
        </div>
      ) : (
        <div className="grid gap-3">
          {songs.map((song) => (
            <div key={song.id} className="glass-panel rounded-xl p-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between card-hover">
              <button onClick={() => openDetails(song.id)} className="text-left min-w-0">
                <p className="font-semibold text-slate-100 truncate">{song.original_filename}</p>
                <p className="text-sm text-slate-400 mt-1">{song.predicted_genre} · {(song.prediction_confidence * 100).toFixed(1)}% confidence</p>
              </button>
              <div className="flex items-center gap-4">
                <span className="text-xs text-slate-500">{new Date(song.uploaded_at).toLocaleString()}</span>
                <button onClick={() => void playSong(song, [song], 'Music library')} title="Play song" aria-label={`Play ${song.original_filename}`} className="p-2 text-slate-400 hover:text-cyan-300">
                  <Play className="w-4 h-4" />
                </button>
                <button onClick={() => loadRecommendations(song)} title="Find similar songs" className="p-2 text-slate-400 hover:text-cyan-300" aria-label={`Find similar songs to ${song.original_filename}`}>
                  <Sparkles className="w-4 h-4" />
                </button>
                <button onClick={() => deleteSong(song)} title="Delete song" className="p-2 text-slate-400 hover:text-red-300" aria-label={`Delete ${song.original_filename}`}>
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {(recommendationLoading || recommendations) && (
        <div className="glass-panel rounded-2xl p-6 mt-6">
          <div className="flex items-start justify-between gap-4 mb-4">
            <div>
              <p className="text-sm text-cyan-400">Recommended songs</p>
              <h3 className="text-xl font-bold text-slate-100">Similar to {recommendationSource}</h3>
              <p className="text-sm text-slate-400 mt-1">{recommendations?.message || 'Calculating similarity from stored audio features...'}</p>
            </div>
            <button onClick={() => { setRecommendations(null); setRecommendationTarget(null); }} aria-label="Close recommendations" className="p-2 text-slate-400 hover:text-slate-100"><X /></button>
          </div>
          {recommendationLoading ? (
            <p className="text-slate-400">Calculating similarity...</p>
          ) : recommendationItems.length === 0 ? (
            <p className="text-slate-400">Add another successfully analyzed song to compare this track.</p>
          ) : (
            <div className="space-y-3">
              {recommendationItems.map((recommendation) => (
                <div key={recommendation.song_id} className="bg-slate-900/60 rounded-lg p-3">
                  <div className="flex justify-between gap-4">
                    <div>
                      <label className="flex items-center gap-2 font-semibold text-slate-100"><input type="checkbox" checked={selectedRecommendationIds.includes(recommendation.song_id)} onChange={(event) => setSelectedRecommendationIds((current) => event.target.checked ? [...current, recommendation.song_id] : current.filter((id) => id !== recommendation.song_id))} />{recommendation.filename}</label>
                      <p className="text-sm text-slate-400">{recommendation.predicted_genre || 'Unknown genre'} · {formatPercent(recommendation.confidence)} confidence</p>
                    </div>
                    <span className="text-cyan-300 font-semibold">{formatPercent(recommendation.score)}</span>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">{recommendation.explanation}</p>
                </div>
              ))}
            </div>
          )}
          {!recommendationLoading && recommendationItems.length > 0 && (
            <div className="mt-4 flex flex-col sm:flex-row gap-2">
              <select value={recommendationPlaylistId} onChange={(event) => setRecommendationPlaylistId(event.target.value)} className="flex-1 bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100">
                <option value="">Choose a playlist</option>
                {playlistOptions.map((playlist) => <option key={playlist.id} value={playlist.id}>{playlist.name}</option>)}
              </select>
              <button onClick={addSelectedRecommendations} disabled={!recommendationPlaylistId || !selectedRecommendationIds.length} className="rounded-lg px-4 py-2 bg-cyan-500/20 text-cyan-200 disabled:opacity-40">Add selected</button>
            </div>
          )}
        </div>
      )}

      {selectedSong && (
        <div className="fixed inset-0 z-20 bg-slate-950/80 p-4 flex items-center justify-center" onClick={() => setSelectedSong(null)}>
          <div className="glass-panel rounded-2xl p-6 w-full max-w-3xl max-h-[90vh] overflow-y-auto" onClick={(event) => event.stopPropagation()}>
            <div className="flex justify-between items-start gap-4 mb-6">
              <div>
                <p className="text-sm text-cyan-400">Song details</p>
                <h3 className="text-2xl font-bold text-slate-100 break-words">{selectedSong.original_filename}</h3>
                <p className="text-slate-400 mt-1">{selectedSong.predicted_genre} · {(selectedSong.prediction_confidence * 100).toFixed(2)}%</p>
              </div>
              <button onClick={() => setSelectedSong(null)} aria-label="Close details" className="p-2 text-slate-400 hover:text-slate-100"><X /></button>
            </div>
            <div className="grid gap-2 mb-6">
              {Object.entries(selectedSong.probabilities).sort(([, a], [, b]) => b - a).map(([name, probability]) => (
                <div key={name} className="flex justify-between text-sm border-b border-white/5 py-2">
                  <span className="text-slate-300">{name}</span>
                  <span className="text-slate-400">{(probability * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>
            <h4 className="font-semibold text-slate-100 mb-3">Window analysis</h4>
            <div className="space-y-2">
              {selectedSong.windows.map((window, index) => (
                <div key={`${window.start_seconds}-${index}`} className="bg-slate-900/60 rounded-lg px-3 py-2 flex justify-between text-sm">
                  <span className="text-slate-300">{window.start_seconds.toFixed(2)}s - {window.end_seconds.toFixed(2)}s</span>
                  <span className="text-cyan-300">{window.genre} · {(window.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
