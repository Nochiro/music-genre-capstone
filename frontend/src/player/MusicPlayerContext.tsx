import { createContext, useContext, useEffect, useRef, useState } from 'react';

export interface PlayableSong {
  id: string;
  original_filename: string;
}

interface MusicPlayerContextValue {
  currentSong: PlayableSong | null;
  currentSource: string | null;
  isPlaying: boolean;
  isLoading: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  error: string | null;
  playSong: (song: PlayableSong, queue?: PlayableSong[], source?: string) => Promise<void>;
  playQueue: (songs: PlayableSong[], source?: string) => Promise<void>;
  togglePlay: () => Promise<void>;
  next: () => Promise<void>;
  previous: () => Promise<void>;
  seek: (time: number) => void;
  setVolume: (volume: number) => void;
  toggleMute: () => void;
  syncQueue: (songs: PlayableSong[]) => void;
}

const MusicPlayerContext = createContext<MusicPlayerContextValue | null>(null);
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export function MusicPlayerProvider({ children }: { children: React.ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const queueRef = useRef<PlayableSong[]>([]);
  const indexRef = useRef(-1);
  const [currentSong, setCurrentSong] = useState<PlayableSong | null>(null);
  const [currentSource, setCurrentSource] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.preload = 'metadata';

    const onLoadStart = () => { setIsLoading(true); setError(null); };
    const onCanPlay = () => { setIsLoading(false); };
    const onLoadedMetadata = () => { setDuration(Number.isFinite(audio.duration) ? audio.duration : 0); };
    const onTimeUpdate = () => { setCurrentTime(audio.currentTime); };
    const onPlay = () => { setIsPlaying(true); };
    const onPause = () => { setIsPlaying(false); };
    const onWaiting = () => { setIsLoading(true); };
    const onPlaying = () => { setIsLoading(false); };
    const onError = () => { setIsLoading(false); setIsPlaying(false); setError('Unable to play this audio file.'); };
    const onEnded = () => {
      const nextIndex = indexRef.current + 1;
      if (nextIndex < queueRef.current.length) {
        indexRef.current = nextIndex;
        void loadSong(queueRef.current[nextIndex], false);
      } else {
        setIsPlaying(false);
        setCurrentTime(0);
      }
    };

    audio.addEventListener('loadstart', onLoadStart);
    audio.addEventListener('canplay', onCanPlay);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('waiting', onWaiting);
    audio.addEventListener('playing', onPlaying);
    audio.addEventListener('error', onError);
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
      audio.removeEventListener('loadstart', onLoadStart);
      audio.removeEventListener('canplay', onCanPlay);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('waiting', onWaiting);
      audio.removeEventListener('playing', onPlaying);
      audio.removeEventListener('error', onError);
      audio.removeEventListener('ended', onEnded);
    };
  }, []);

  const loadSong = async (song: PlayableSong, autoplay: boolean) => {
    const audio = audioRef.current;
    if (!audio) return;
    setCurrentSong(song);
    setCurrentTime(0);
    setDuration(0);
    setError(null);
    audio.src = `${API_URL}/library/${encodeURIComponent(song.id)}/audio`;
    audio.volume = volume;
    audio.muted = isMuted;
    audio.load();
    if (autoplay) {
      try {
        await audio.play();
      } catch {
        setIsPlaying(false);
        setError('Playback was blocked. Press play to start the song.');
      }
    }
  };

  const playSong = async (song: PlayableSong, queue?: PlayableSong[], source?: string) => {
    const nextQueue = queue && queue.length ? queue : [song];
    queueRef.current = nextQueue;
    indexRef.current = Math.max(0, nextQueue.findIndex((item) => item.id === song.id));
    setCurrentSource(source || null);
    if (currentSong?.id === song.id && audioRef.current?.src) {
      try {
        await audioRef.current.play();
      } catch {
        setError('Playback was blocked. Press play to start the song.');
      }
      return;
    }
    await loadSong(song, true);
  };

  const playQueue = async (songs: PlayableSong[], source?: string) => {
    if (!songs.length) return;
    setCurrentSource(source || null);
    await playSong(songs[0], songs, source);
  };

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio || !currentSong) return;
    try {
      if (audio.paused) await audio.play();
      else audio.pause();
    } catch {
      setError('Playback was blocked. Press play to start the song.');
    }
  };

  const next = async () => {
    const nextIndex = indexRef.current + 1;
    if (nextIndex >= queueRef.current.length) return;
    indexRef.current = nextIndex;
    await loadSong(queueRef.current[nextIndex], true);
  };

  const previous = async () => {
    const audio = audioRef.current;
    if (audio && audio.currentTime > 3) {
      audio.currentTime = 0;
      setCurrentTime(0);
      return;
    }
    const previousIndex = indexRef.current - 1;
    if (previousIndex < 0) return;
    indexRef.current = previousIndex;
    await loadSong(queueRef.current[previousIndex], true);
  };

  const seek = (time: number) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = Math.max(0, Math.min(time, duration || time));
    setCurrentTime(audioRef.current.currentTime);
  };

  const setVolume = (nextVolume: number) => {
    const safeVolume = Math.max(0, Math.min(1, nextVolume));
    setVolumeState(safeVolume);
    if (audioRef.current) audioRef.current.volume = safeVolume;
    if (safeVolume > 0) setIsMuted(false);
  };

  const toggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    if (audioRef.current) audioRef.current.muted = nextMuted;
  };

  const syncQueue = (songs: PlayableSong[]) => {
    queueRef.current = songs;
    if (!currentSong) return;
    const nextIndex = songs.findIndex((song) => song.id === currentSong.id);
    if (nextIndex === -1) {
      audioRef.current?.pause();
      setCurrentSong(null);
      setIsPlaying(false);
      indexRef.current = -1;
    } else {
      indexRef.current = nextIndex;
    }
  };

  return (
    <MusicPlayerContext.Provider value={{ currentSong, currentSource, isPlaying, isLoading, currentTime, duration, volume, isMuted, error, playSong, playQueue, togglePlay, next, previous, seek, setVolume, toggleMute, syncQueue }}>
      {children}
      <audio ref={audioRef} preload="metadata" className="hidden" />
    </MusicPlayerContext.Provider>
  );
}

export function useMusicPlayer() {
  const context = useContext(MusicPlayerContext);
  if (!context) throw new Error('useMusicPlayer must be used inside MusicPlayerProvider');
  return context;
}
