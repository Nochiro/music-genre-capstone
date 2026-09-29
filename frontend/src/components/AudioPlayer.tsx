import React from 'react';
import { Play, Pause, Volume2 } from 'lucide-react';

interface AudioPlayerProps {
  file: File;
  onDurationChange?: (duration: number) => void;
}

export default function AudioPlayer({
  file,
  onDurationChange,
}: AudioPlayerProps) {
  const audioRef = React.useRef<HTMLAudioElement>(null);
  const objectUrlRef = React.useRef<string | null>(null);

  const [isPlaying, setIsPlaying] = React.useState(false);
  const [currentTime, setCurrentTime] = React.useState(0);
  const [duration, setDuration] = React.useState(0);
  const [error, setError] = React.useState<string | null>(null);

  // Create ONE object URL for this file.
  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;

    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setError(null);

    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.removeAttribute('src');
        audioRef.current.load();
      }

      URL.revokeObjectURL(url);
      objectUrlRef.current = null;
    };
  }, [file]);

  const togglePlay = async () => {
    const audio = audioRef.current;

    if (!audio) {
      return;
    }

    try {
      setError(null);

      if (audio.paused) {
        await audio.play();
        setIsPlaying(true);
      } else {
        audio.pause();
        setIsPlaying(false);
      }
    } catch (err) {
      console.error('Audio playback failed:', err);
      setIsPlaying(false);
      setError('Unable to play this audio file.');
    }
  };

  const handleTimeUpdate = () => {
    const audio = audioRef.current;

    if (audio) {
      setCurrentTime(audio.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    const audio = audioRef.current;

    if (!audio) {
      return;
    }

    const dur = audio.duration;

    if (Number.isFinite(dur)) {
      setDuration(dur);
      onDurationChange?.(dur);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleAudioError = () => {
    console.error('Browser could not load the audio file.');
    setIsPlaying(false);
    setError('The browser could not load this audio file.');
  };

  const handleSeek = (
    event: React.MouseEvent<HTMLDivElement>
  ) => {
    const audio = audioRef.current;

    if (!audio || !duration) {
      return;
    }

    const rect = event.currentTarget.getBoundingClientRect();
    const clickPosition = event.clientX - rect.left;
    const percentage = Math.max(
      0,
      Math.min(1, clickPosition / rect.width)
    );

    audio.currentTime = percentage * duration;
    setCurrentTime(audio.currentTime);
  };

  const formatTime = (seconds: number) => {
    if (!Number.isFinite(seconds) || seconds < 0) {
      return '0:00';
    }

    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progressPercent =
    duration > 0
      ? Math.min(100, (currentTime / duration) * 100)
      : 0;

  return (
    <div className="w-full glass-panel p-4 rounded-xl">
      <audio
        ref={audioRef}
        src={objectUrlRef.current ?? undefined}
        preload="metadata"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        onError={handleAudioError}
      />

      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={togglePlay}
          className="p-2 bg-gradient-to-r from-violet-500 to-cyan-500 hover:from-violet-600 hover:to-cyan-600 rounded-lg transition-all button-hover"
          aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
        >
          {isPlaying ? (
            <Pause className="w-5 h-5 text-white" />
          ) : (
            <Play className="w-5 h-5 text-white ml-0.5" />
          )}
        </button>

        <div className="flex-1">
          <div
            className="relative h-2 bg-slate-700 rounded-full overflow-hidden cursor-pointer"
            onClick={handleSeek}
          >
            <div
              className="h-full bg-gradient-to-r from-violet-400 to-cyan-400 transition-all"
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <div className="flex justify-between items-center mt-2 text-xs text-slate-400">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>

          {error && (
            <p className="text-xs text-red-400 mt-2">
              {error}
            </p>
          )}
        </div>

        <Volume2 className="w-4 h-4 text-slate-400" />
      </div>
    </div>
  );
}