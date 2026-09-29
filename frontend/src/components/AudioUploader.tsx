import React, { useRef } from 'react';
import { Upload, Music } from 'lucide-react';

interface AudioUploaderProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
  fileName?: string;
}

export default function AudioUploader({ onFileSelect, disabled, fileName }: AudioUploaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragActive, setIsDragActive] = React.useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const processFile = (file: File) => {
    if (file.type.startsWith('audio/') || file.type === 'application/octet-stream') {
      onFileSelect(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full fade-in">
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled}
      />

      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`
          relative group cursor-pointer
          border-2 border-dashed rounded-2xl p-12
          transition-all duration-300
          ${
            isDragActive
              ? 'upload-active'
              : 'border-slate-700 hover:border-violet-500/50 hover:bg-white/5'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <div className="flex flex-col items-center justify-center gap-4">
          <div
            className={`
              p-4 rounded-full transition-all duration-300
              ${
                isDragActive
                  ? 'bg-violet-500/20 scale-110'
                  : 'bg-slate-800 group-hover:bg-violet-500/10'
              }
            `}
          >
            <Upload
              className={`w-8 h-8 transition-all ${isDragActive ? 'text-violet-400' : 'text-slate-400'}`}
            />
          </div>

          <div className="text-center">
            <p className="text-lg font-semibold text-slate-100">
              {fileName ? 'File selected' : 'Drop your audio here'}
            </p>
            <p className="text-sm text-slate-400 mt-1">
              {fileName ? fileName : 'or click to browse • MP3, WAV supported'}
            </p>
          </div>

          {fileName && (
            <div className="flex items-center gap-2 mt-2 px-3 py-1 bg-violet-500/10 border border-violet-500/20 rounded-lg">
              <Music className="w-4 h-4 text-violet-400" />
              <span className="text-sm text-violet-300">{fileName}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
