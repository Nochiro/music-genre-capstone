import { Pause, Play, SkipBack, SkipForward, Volume2, VolumeX } from 'lucide-react';
import { useMusicPlayer } from '../player/MusicPlayerContext';

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
};

export default function MusicPlayer() {
  const {
    currentSong, currentSource, isPlaying, isLoading, currentTime, duration,
    volume, isMuted, error, togglePlay, next, previous, seek, setVolume, toggleMute,
  } = useMusicPlayer();

  if (!currentSong) return null;

  return (
    <aside className="fixed bottom-0 left-0 right-0 z-30 border-t border-white/10 bg-slate-950/95 backdrop-blur-xl shadow-2xl" aria-label="Music player">
      <div className="max-w-7xl mx-auto px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-2 md:grid md:grid-cols-[minmax(12rem,1fr)_minmax(18rem,2fr)_minmax(8rem,auto)] md:items-center md:gap-6">
          <div className="min-w-0">
            <p className="text-xs text-cyan-400 truncate">{currentSource || 'Music library'}</p>
            <p className="font-semibold text-slate-100 truncate">{currentSong.original_filename}</p>
            {error && <p className="text-xs text-orange-300 truncate" role="alert">{error}</p>}
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => void previous()} aria-label="Previous song" title="Previous song" className="p-2 text-slate-300 hover:text-white"><SkipBack className="w-5 h-5" /></button>
            <button onClick={() => void togglePlay()} aria-label={isPlaying ? 'Pause' : 'Play'} title={isPlaying ? 'Pause' : 'Play'} className="p-3 rounded-full bg-cyan-500 text-slate-950 hover:bg-cyan-300">
              {isLoading ? <span className="block w-5 h-5 rounded-full border-2 border-slate-950/30 border-t-slate-950 animate-spin" /> : isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
            </button>
            <button onClick={() => void next()} aria-label="Next song" title="Next song" className="p-2 text-slate-300 hover:text-white"><SkipForward className="w-5 h-5" /></button>
            <span className="text-xs text-slate-400 tabular-nums w-20">{formatTime(currentTime)} / {formatTime(duration)}</span>
            <input aria-label="Seek" type="range" min="0" max={duration || 0} step="0.1" value={Math.min(currentTime, duration || 0)} onChange={(event) => seek(Number(event.target.value))} disabled={!duration} className="flex-1 accent-cyan-400" />
          </div>
          <div className="flex items-center gap-2 justify-end">
            <button onClick={toggleMute} aria-label={isMuted ? 'Unmute' : 'Mute'} title={isMuted ? 'Unmute' : 'Mute'} className="p-2 text-slate-300 hover:text-white">{isMuted || volume === 0 ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}</button>
            <input aria-label="Volume" type="range" min="0" max="1" step="0.01" value={isMuted ? 0 : volume} onChange={(event) => setVolume(Number(event.target.value))} className="w-24 accent-cyan-400" />
          </div>
        </div>
      </div>
    </aside>
  );
}
