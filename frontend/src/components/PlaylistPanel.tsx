import { useEffect, useState } from 'react';
import { ArrowDown, ArrowUp, ListMusic, Play, Plus, Save, Trash2, X } from 'lucide-react';
import { apiClient } from '../services/api';
import { LibrarySong, Playlist, PlaylistDetail } from '../types/api';
import { useMusicPlayer } from '../player/MusicPlayerContext';

interface PlaylistPanelProps {
  refreshToken: number;
}

export default function PlaylistPanel({ refreshToken }: PlaylistPanelProps) {
  const { currentSong, playQueue, playSong, syncQueue } = useMusicPlayer();
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [librarySongs, setLibrarySongs] = useState<LibrarySong[]>([]);
  const [activePlaylist, setActivePlaylist] = useState<PlaylistDetail | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedSongIds, setSelectedSongIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const loadPlaylists = async () => {
    const result = await apiClient.getPlaylists();
    setPlaylists(result.playlists);
  };

  useEffect(() => {
    const refresh = async () => {
      const [, library] = await Promise.all([loadPlaylists(), apiClient.getLibrary('', '')]);
      setLibrarySongs(library.songs);
      if (activePlaylist) {
        setActivePlaylist(await apiClient.getPlaylist(activePlaylist.id));
      }
    };
    refresh()
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Could not load playlists'));
  }, [refreshToken]);

  const openPlaylist = async (playlistId: string) => {
    try {
      const playlist = await apiClient.getPlaylist(playlistId);
      setActivePlaylist(playlist);
      setName(playlist.name);
      setDescription(playlist.description || '');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load playlist');
    }
  };

  const createPlaylist = async () => {
    if (!name.trim()) {
      setError('Playlist name cannot be blank.');
      return;
    }
    setLoading(true);
    try {
      const playlist = await apiClient.createPlaylist(name, description);
      await loadPlaylists();
      setActivePlaylist(playlist);
      setNotice('Playlist created.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create playlist');
    } finally {
      setLoading(false);
    }
  };

  const savePlaylist = async () => {
    if (!activePlaylist || !name.trim()) return;
    setLoading(true);
    try {
      const playlist = await apiClient.updatePlaylist(activePlaylist.id, name, description || null);
      setActivePlaylist(playlist);
      await loadPlaylists();
      setNotice('Playlist updated.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not update playlist');
    } finally {
      setLoading(false);
    }
  };

  const removePlaylist = async () => {
    if (!activePlaylist || !window.confirm(`Delete ${activePlaylist.name}? Songs will remain in the library.`)) return;
    try {
      await apiClient.deletePlaylist(activePlaylist.id);
      setActivePlaylist(null);
      await loadPlaylists();
      setNotice('Playlist deleted.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not delete playlist');
    }
  };

  const addSelectedSongs = async () => {
    if (!activePlaylist || selectedSongIds.length === 0) return;
    try {
      const result = await apiClient.addSongsToPlaylist(activePlaylist.id, selectedSongIds);
      setActivePlaylist(result.playlist);
      setSelectedSongIds([]);
      await loadPlaylists();
      setNotice('Songs added.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not add songs');
    }
  };

  const moveSong = async (index: number, direction: -1 | 1) => {
    if (!activePlaylist) return;
    const next = [...activePlaylist.songs];
    const target = index + direction;
    if (target < 0 || target >= next.length) return;
    [next[index], next[target]] = [next[target], next[index]];
    try {
      const reordered = await apiClient.reorderPlaylist(activePlaylist.id, next.map((song) => song.id));
      setActivePlaylist(reordered);
      syncQueue(reordered.songs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reorder playlist');
    }
  };

  const removeSong = async (songId: string) => {
    if (!activePlaylist) return;
    try {
      const updated = await apiClient.removeSongFromPlaylist(activePlaylist.id, songId);
      setActivePlaylist(updated);
      syncQueue(updated.songs);
      await loadPlaylists();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not remove song');
    }
  };

  return (
    <section className="mt-16 fade-in">
      <div className="flex items-center gap-3 mb-6">
        <ListMusic className="text-cyan-400" />
        <div>
          <p className="text-sm uppercase tracking-widest text-cyan-400">Persistent collection</p>
          <h2 className="text-3xl font-bold text-slate-100">Playlists</h2>
        </div>
      </div>
      {error && <p className="mb-3 text-sm text-orange-300">{error}</p>}
      {notice && <p className="mb-3 text-sm text-emerald-300">{notice}</p>}
      <div className="grid gap-6 lg:grid-cols-[minmax(13rem,0.8fr)_minmax(0,2fr)]">
        <div className="space-y-3">
          <div className="glass-panel rounded-xl p-4 space-y-3">
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Playlist name" maxLength={120} className="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100" />
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Description (optional)" maxLength={1000} rows={2} className="w-full bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100" />
            <button onClick={activePlaylist ? savePlaylist : createPlaylist} disabled={loading} className="w-full flex items-center justify-center gap-2 rounded-lg py-2 bg-cyan-500/20 text-cyan-200 hover:bg-cyan-500/30 disabled:opacity-50">
              {activePlaylist ? <Save className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
              {activePlaylist ? 'Save changes' : 'Create playlist'}
            </button>
            {activePlaylist && <button onClick={() => { setActivePlaylist(null); setName(''); setDescription(''); }} className="w-full text-xs text-slate-400 hover:text-slate-100">New playlist</button>}
          </div>
          {playlists.length === 0 ? <p className="text-sm text-slate-500 px-2">No playlists yet.</p> : playlists.map((playlist) => (
            <button key={playlist.id} onClick={() => openPlaylist(playlist.id)} className={`w-full text-left rounded-xl p-3 border ${activePlaylist?.id === playlist.id ? 'border-cyan-400/60 bg-cyan-400/10' : 'border-white/10 bg-white/5 hover:border-cyan-400/40'}`}>
              <p className="font-semibold text-slate-100 truncate">{playlist.name}</p>
              <p className="text-xs text-slate-400 mt-1">{playlist.song_count} song{playlist.song_count === 1 ? '' : 's'}</p>
            </button>
          ))}
        </div>
        <div className="glass-panel rounded-2xl p-6 min-h-64">
          {!activePlaylist ? <div className="h-full flex items-center justify-center text-center text-slate-500">Create or select a playlist to manage its songs.</div> : (
            <>
              <div className="flex justify-between gap-4 mb-5">
                <div><h3 className="text-xl font-bold text-slate-100">{activePlaylist.name}</h3><p className="text-sm text-slate-400">{activePlaylist.description || 'No description'}</p></div>
                <button onClick={() => void playQueue(activePlaylist.songs, activePlaylist.name)} disabled={!activePlaylist.songs.length} title="Play all songs" aria-label="Play all songs" className="p-2 text-cyan-300 disabled:opacity-30"><Play className="w-5 h-5" /></button>
                <button onClick={removePlaylist} title="Delete playlist" aria-label="Delete playlist" className="p-2 text-slate-400 hover:text-red-300"><Trash2 className="w-4 h-4" /></button>
              </div>
              <div className="mb-6 flex flex-col sm:flex-row gap-2">
                <select multiple value={selectedSongIds} onChange={(event) => setSelectedSongIds(Array.from(event.target.selectedOptions, (option) => option.value))} className="flex-1 bg-slate-900/80 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 min-h-20">
                  {librarySongs.filter((song) => !activePlaylist.songs.some((item) => item.id === song.id)).map((song) => <option key={song.id} value={song.id}>{song.original_filename}</option>)}
                </select>
                <button onClick={addSelectedSongs} disabled={!selectedSongIds.length} className="rounded-lg px-4 py-2 bg-violet-500/20 text-violet-200 disabled:opacity-40">Add songs</button>
              </div>
              {activePlaylist.songs.length === 0 ? <p className="text-slate-500">This playlist is empty.</p> : <div className="space-y-2">
                {activePlaylist.songs.map((song, index) => <div key={song.id} className={`flex items-center gap-3 rounded-lg p-3 ${currentSong?.id === song.id ? 'bg-cyan-400/10 ring-1 ring-cyan-400/40' : 'bg-slate-900/60'}`}><span className="text-xs text-slate-500 w-5">{index + 1}</span><button onClick={() => void playSong(song, activePlaylist.songs, activePlaylist.name)} aria-label={`Play ${song.original_filename}`} className="text-cyan-300"><Play className="w-4 h-4" /></button><span className="flex-1 text-sm text-slate-200 truncate">{song.original_filename}</span><button onClick={() => moveSong(index, -1)} disabled={index === 0} aria-label="Move song up" className="text-slate-400 disabled:opacity-30"><ArrowUp className="w-4 h-4" /></button><button onClick={() => moveSong(index, 1)} disabled={index === activePlaylist.songs.length - 1} aria-label="Move song down" className="text-slate-400 disabled:opacity-30"><ArrowDown className="w-4 h-4" /></button><button onClick={() => removeSong(song.id)} aria-label={`Remove ${song.original_filename}`} className="text-slate-400 hover:text-red-300"><X className="w-4 h-4" /></button></div>)}
              </div>}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
